"""
AIG-Specific Graph Generation using Trained GraphRNN Model
With optional truth table conditioning
"""

import argparse
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import os

from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP
from aig_dataset import NODE_TYPES, EDGE_TYPES
import generate
from generate import m_seq_to_adj_mat, sample_bernoulli, sample_softmax

# Node type color mapping for visualization
NODE_TYPE_COLORS = {
    NODE_TYPES['ZERO']: 'gray',
    NODE_TYPES['PI']: 'green',
    NODE_TYPES['AND']: 'blue',
    NODE_TYPES['PO']: 'red'
}


def rnn_edge_gen(edge_rnn, h, num_edges, adj_vec_size, sample_fun, attempts=None, truth_table=None):
    """
    Generates the edges coming from this node using RNN method, with optional truth table conditioning.

    Arguments:
        edge_rnn: EdgeRNN model to use for generation
        h: Hidden state computed by the NodeLevelRNN
        num_edges: Number of edges to generate.
        adj_vec_size: Size of the padded adjacency vector to output.
            This should corresponds to the input size of the NodeLeveRNN.
        attempts: Not implemented!
        truth_table: Optional truth table tensor to condition the generation

    Returns: Adjacency vector of size [1, 1, adj_vec_size]
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adj_vec = torch.zeros([1, 1, adj_vec_size, edge_rnn.edge_feature_len], device=device)

    edge_rnn.set_first_layer_hidden(h)

    # SOS token
    x = torch.ones([1, 1, edge_rnn.edge_feature_len], device=device)

    for i in range(num_edges):
        # calculate probability of this edge existing, with optional truth table
        if hasattr(edge_rnn, 'use_conditioning') and edge_rnn.use_conditioning and truth_table is not None:
            prob = edge_rnn(x, truth_table=truth_table, return_logits=True)
        else:
            prob = edge_rnn(x, return_logits=True)

        # sample from this probability and assign value to adjacency vector
        # assign the value of this edge into the input of the next iteration
        x[0, 0, :] = torch.tensor(sample_fun(prob[0, 0, :].detach()), device=device)
        adj_vec[0, 0, i, :] = x[0, 0, :]

    return adj_vec


def mlp_edge_gen(edge_mlp, h, num_edges, adj_vec_size, sample_fun, attempts=1, truth_table=None):
    """
    Generates the edges coming from this node using MLP method, with optional truth table conditioning.

    Arguments:
        edge_mlp: EdgeMLP model to use for generation
        h: Hidden state computed by the NodeLevelRNN
        num_edges: Number of edges to generate.
        adj_vec_size: Size of the padded adjacency vector to output.
            This should correspond to the input size of the NodeLeveRNN.
        attempts: Number of retries that should be attempted if no
            edge is sampled.
        truth_table: Optional truth table tensor to condition the generation

    Returns: Adjacency vector of size [1, 1, adj_vec_size]
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adj_vec = torch.zeros([1, 1, adj_vec_size, edge_mlp.edge_feature_len], device=device)

    # calculate probabilities of all edges from this node existing, with optional truth table
    if hasattr(edge_mlp, 'use_conditioning') and edge_mlp.use_conditioning and truth_table is not None:
        edge_probs = edge_mlp(h, truth_table=truth_table, return_logits=True)
    else:
        edge_probs = edge_mlp(h, return_logits=True)

    # update adj_vec with the sampled value from each edge probability
    for _ in range(attempts):
        for i in range(num_edges):
            adj_vec[0, 0, i, :] = torch.tensor(sample_fun(edge_probs[0, 0, i, :].detach()), device=device)
        # If we generated all zeros we will try again if there are
        # attempts left. If we have sampled at least one edge, we can go on.
        if (adj_vec[0, 0, :, :].data > 0).any():
            break

    return adj_vec


def generate_aig(num_nodes, node_model, edge_model, input_size, edge_gen_function, mode, truth_table=None):
    """
    Generates an And-Inverter Graph with specific constraints and optional truth table conditioning

    Args:
        num_nodes: Number of nodes to generate
        node_model: Graph-level RNN model
        edge_model: Edge-level model (MLP or RNN)
        input_size: Maximum number of previous nodes to connect
        edge_gen_function: Edge generation method
        mode: Generation mode (directed-multiclass)
        truth_table: Optional truth table tensor for conditioning [1, n_outputs * 2^n_inputs]

    Returns:
        Adjacency matrix and node types
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    node_model.eval()
    edge_model.eval()

    # Use softmax sampling for multi-class edge types
    sample_fun = sample_softmax if mode == 'directed-multiclass' else sample_bernoulli

    # Initialize adjacency vector and node types
    adj_vec = torch.ones([1, 1, input_size, node_model.edge_feature_len], device=device)
    list_adj_vecs = []
    node_types = []

    # Simple heuristic for node type generation
    def generate_node_type(current_nodes):
        """
        Simple node type generation heuristic
        - First few nodes are PIs
        - Middle nodes are AND gates
        - Last few nodes are POs
        """
        if len(current_nodes) < 2:
            return NODE_TYPES['PI']
        elif len(current_nodes) > num_nodes - 3:
            return NODE_TYPES['PO']
        else:
            return NODE_TYPES['AND']

    node_model.reset_hidden()

    for i in range(1, num_nodes):
        # Generate node type
        current_node_type = generate_node_type(node_types)
        node_types.append(current_node_type)

        # Generate graph state vector with optional truth table
        if hasattr(node_model, 'use_conditioning') and node_model.use_conditioning and truth_table is not None:
            h = node_model(adj_vec, truth_table=truth_table)
        else:
            h = node_model(adj_vec)

        # Generate edges for this node with optional truth table
        adj_vec = edge_gen_function(
            edge_model,
            h,
            num_edges=min(i, input_size),
            adj_vec_size=input_size,
            sample_fun=sample_fun,
            attempts=1,
            truth_table=truth_table
        )

        # Process adjacency vector
        if mode == 'directed-multiclass':
            # Turn one-hot into class index
            one_hot = adj_vec[0, 0, :min(num_nodes, input_size), :].cpu().detach().int().numpy()
            class_vec = np.zeros([min(num_nodes, input_size)])

            # Handle cases where there might be no non-zero values in one_hot[:i]
            nonzero_indices = one_hot[:i].nonzero()
            if len(nonzero_indices) > 0 and len(nonzero_indices[0]) > 0:
                class_vec[:i] = nonzero_indices[1]

            list_adj_vecs.append(class_vec)
        else:
            list_adj_vecs.append(adj_vec[0, 0, :min(num_nodes, input_size), 0].cpu().detach().int().numpy())

        # Early stopping if no more edges
        if np.array(list_adj_vecs[-1] == 0).all() and i > num_nodes // 2:
            break

    # Convert to adjacency matrix
    adj = m_seq_to_adj_mat(np.array(list_adj_vecs), m=input_size)

    # Post-processing for the adjacency matrix
    adj = adj + adj.T

    # Remove isolated nodes
    if mode == 'directed-multiclass':
        adj_filtered = adj.copy()
        non_zero_rows = ~np.all(adj_filtered == 0, axis=1)
        non_zero_cols = ~np.all(adj_filtered == 0, axis=0)
        non_zero_indices = non_zero_rows & non_zero_cols
        adj_filtered = adj_filtered[non_zero_indices][:, non_zero_indices]

        # Adjust node types accordingly
        node_types_filtered = [nt for nt, keep in zip(node_types, non_zero_indices) if keep]
        node_types = node_types_filtered
        adj = adj_filtered

    # Get the lower triangular part
    adj = np.tril(adj)

    # Handle directed multiclass edge types
    if mode == 'directed-multiclass':
        adj = (adj % 2) + (adj // 2).T

    return adj, node_types


def visualize_aig(adj_matrix, node_types, output_file='generated_aig.png', truth_table=None):
    """
    Visualize the generated AIG graph

    Args:
        adj_matrix: Adjacency matrix
        node_types: List of node types
        output_file: File to save visualization
        truth_table: Optional truth table used for conditioning (for the title)
    """
    # Create networkx graph
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Set node colors based on types
    node_colors = [NODE_TYPE_COLORS.get(nt, 'gray') for nt in node_types]

    # Set edge colors/styles based on edge types
    edge_colors = []
    edge_styles = []

    for u, v, data in G.edges(data=True):
        # In directed multiclass, edge type is encoded in the weight
        edge_type = int(adj_matrix[u, v])
        if edge_type == 1:  # Regular edge
            edge_colors.append('black')
            edge_styles.append('solid')
        elif edge_type == 2:  # Inverted edge
            edge_colors.append('red')
            edge_styles.append('dashed')
        else:
            edge_colors.append('gray')
            edge_styles.append('dotted')

    # Visualization
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=500
    )

    # Draw edges with appropriate styles
    for i, (u, v) in enumerate(G.edges()):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=1.5,
            edge_color=edge_colors[i],
            style=edge_styles[i],
            arrowsize=15
        )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight='bold'
    )

    # Add a legend for node types
    legend_elements = [
        plt.Line2D([0], [0], color=color, marker='o', linestyle='None',
                   markersize=10, label=name)
        for name, type_id in NODE_TYPES.items()
        for c_name, color in NODE_TYPE_COLORS.items()
        if c_name == type_id
    ]

    # Add legend for edge types
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='solid',
                           markersize=10, label='Regular Edge'))
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='dashed',
                           markersize=10, label='Inverted Edge'))

    plt.legend(handles=legend_elements, title='AIG Components')

    # Create title based on whether truth table was used
    if truth_table is not None:
        plt.title('Generated AIG Graph (Truth Table Conditioned)')
    else:
        plt.title('Generated AIG Graph (Unconditioned)')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    return G


def load_aig_model_from_config(model_path):
    """
    Load AIG-specific models from checkpoint

    Args:
        model_path: Path to model checkpoint

    Returns:
        Loaded models and configuration details
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state = torch.load(model_path, map_location=device)
    config = state['config']

    input_size = config['data']['m']

    # Check if the model supports truth table conditioning
    tt_size = None
    if 'truth_table_conditioning' in config['model'] and config['model']['truth_table_conditioning']:
        n_outputs = config['model'].get('n_outputs', 8)
        n_inputs = config['model'].get('n_inputs', 8)
        tt_size = n_outputs * (2 ** n_inputs)

    if config['model']['edge_model'] == 'rnn':
        node_model = GraphLevelRNN(
            input_size=config['data']['m'],
            output_size=config['model']['EdgeRNN']['hidden_size'],
            tt_size=tt_size,
            **config['model']['GraphRNN']
        ).to(device)
        edge_model = EdgeLevelRNN(
            tt_size=tt_size,
            **config['model']['EdgeRNN']
        ).to(device)
        edge_gen_function = rnn_edge_gen
    else:
        node_model = GraphLevelRNN(
            input_size=config['data']['m'],
            output_size=None,
            tt_size=tt_size,
            **config['model']['GraphRNN']
        ).to(device)
        edge_model = EdgeLevelMLP(
            input_size=config['model']['GraphRNN']['hidden_size'],
            output_size=config['data']['m'],
            tt_size=tt_size,
            **config['model']['EdgeMLP']
        ).to(device)
        edge_gen_function = mlp_edge_gen

    node_model.load_state_dict(state['node_model'])
    edge_model.load_state_dict(state['edge_model'])

    mode = config['model']['mode'] if 'mode' in config['model'] else 'directed-multiclass'

    return node_model, edge_model, input_size, edge_gen_function, mode, tt_size


def create_random_truth_table(n_outputs=8, n_inputs=8, device='cpu'):
    """
    Create a random truth table for conditioning

    Args:
        n_outputs: Number of output bits
        n_inputs: Number of input bits
        device: Device to place tensor on

    Returns:
        Flattened truth table tensor of shape [1, n_outputs * 2^n_inputs]
    """
    # Create random binary truth table - shape [n_outputs, 2^n_inputs]
    table = torch.randint(0, 2, (n_outputs, 2**n_inputs), dtype=torch.float)

    # Flatten to [1, n_outputs * 2^n_inputs] for conditioning
    flattened = table.reshape(1, -1).to(device)

    return flattened, table


def create_specific_truth_table(function_name, n_inputs=8, device='cpu'):
    """
    Create a specific truth table for a common Boolean function

    Args:
        function_name: Name of function ('AND', 'OR', 'XOR', 'MAJORITY', etc.)
        n_inputs: Number of input bits
        device: Device to place tensor on

    Returns:
        Flattened truth table tensor and original table
    """
    num_combinations = 2**n_inputs
    table = torch.zeros((1, num_combinations), dtype=torch.float)

    # Generate all possible input combinations
    input_combinations = []
    for i in range(num_combinations):
        # Convert i to binary and pad to n_inputs bits
        binary = format(i, f'0{n_inputs}b')
        inputs = [int(b) for b in binary]
        input_combinations.append(inputs)

    # Compute the output for each input combination
    for i, inputs in enumerate(input_combinations):
        if function_name == 'AND':
            # AND of all inputs
            output = all(inputs)
        elif function_name == 'OR':
            # OR of all inputs
            output = any(inputs)
        elif function_name == 'XOR':
            # XOR of all inputs
            output = sum(inputs) % 2 == 1
        elif function_name == 'MAJORITY':
            # Majority vote
            output = sum(inputs) > n_inputs/2
        elif function_name == 'PARITY':
            # Parity check (even parity)
            output = sum(inputs) % 2 == 0
        else:
            # Default to identity function (output = first input)
            output = inputs[0] == 1

        table[0, i] = float(output)

    # For functions like AND/OR, repeat the table for multiple outputs
    full_table = table.repeat(8, 1)  # 8 outputs

    # Flatten to [1, n_outputs * 2^n_inputs] for conditioning
    flattened = full_table.reshape(1, -1).to(device)

    return flattened, full_table


def calculate_aig_validity(G, node_types):
    """
    Calculate the validity of an AIG based on gate input requirements.

    A valid AIG has:
    - AND gates with exactly 2 inputs
    - PO (output) gates with exactly 1 input

    Args:
        G: NetworkX DiGraph object representing the AIG
        node_types: List of node types corresponding to each node in G

    Returns:
        float: Percentage of gates that have the correct number of inputs (0.0 to 1.0)
        dict: Detailed stats about gate validities
    """
    # Count gates that need validation (AND and PO gates)
    and_gates = [i for i, nt in enumerate(node_types) if nt == NODE_TYPES['AND']]
    po_gates = [i for i, nt in enumerate(node_types) if nt == NODE_TYPES['PO']]
    total_gates = len(and_gates) + len(po_gates)

    # Prepare detailed stats
    stats = {
        'and_gates': len(and_gates),
        'po_gates': len(po_gates),
        'valid_and_gates': 0,
        'valid_po_gates': 0,
        'and_validity': 0.0,
        'po_validity': 0.0,
        'overall_validity': 0.0
    }

    if total_gates == 0:
        # No gates to validate (unusual, but possible)
        return 1.0, stats

    # Count correctly connected gates
    valid_gates = 0

    # Check AND gates (should have exactly 2 inputs)
    for gate in and_gates:
        # Count incoming edges to this gate
        inputs = G.in_degree(gate)
        if inputs == 2:
            valid_gates += 1
            stats['valid_and_gates'] += 1

    # Check PO gates (should have exactly 1 input)
    for gate in po_gates:
        # Count incoming edges to this gate
        inputs = G.in_degree(gate)
        if inputs == 1:
            valid_gates += 1
            stats['valid_po_gates'] += 1

    # Calculate percentage of valid gates
    if len(and_gates) > 0:
        stats['and_validity'] = stats['valid_and_gates'] / len(and_gates)

    if len(po_gates) > 0:
        stats['po_validity'] = stats['valid_po_gates'] / len(po_gates)

    stats['overall_validity'] = valid_gates / total_gates

    return stats['overall_validity'], stats


def main():
    """
    Main script for generating AIGs with optional truth table conditioning
    """
    parser = argparse.ArgumentParser(description='AIG Graph Generation with Truth Table Conditioning')
    parser.add_argument('model_path', help='Path to trained model checkpoint')
    parser.add_argument('-n', '--nodes', type=int, default=60,
                        help='Number of nodes to generate')
    parser.add_argument('-g', '--graphs', type=int, default=3,
                        help='Number of graphs to generate')
    parser.add_argument('-o', '--output_dir', default='generated_aigs',
                        help='Directory to save generated graphs')
    parser.add_argument('--condition', action='store_true',
                        help='Use truth table conditioning for generation')
    parser.add_argument('--function', choices=['RANDOM', 'AND', 'OR', 'XOR', 'MAJORITY', 'PARITY'],
                        default='RANDOM', help='Boolean function for truth table (if conditioning)')
    parser.add_argument('--validity-threshold', type=float, default=0.0,
                        help='Minimum validity threshold (0.0-1.0). Keep generating until threshold is met')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    node_model, edge_model, input_size, edge_gen_function, mode, tt_size = load_aig_model_from_config(args.model_path)

    # Check if truth table conditioning is available
    has_conditioning = tt_size is not None and hasattr(node_model, 'use_conditioning') and node_model.use_conditioning
    if args.condition and not has_conditioning:
        print("Warning: Model does not support truth table conditioning. Generating unconditioned graphs.")
        args.condition = False

    # Create device
    device = next(node_model.parameters()).device

    # Create truth table if conditioning is enabled
    truth_table = None
    if args.condition and has_conditioning:
        # Parse truth table parameters from config
        n_outputs = 8  # Default
        n_inputs = 8   # Default

        # Generate truth table based on specified function
        if args.function == 'RANDOM':
            truth_table, table_original = create_random_truth_table(n_outputs, n_inputs, device)
            function_desc = "random"
        else:
            truth_table, table_original = create_specific_truth_table(args.function, n_inputs, device)
            function_desc = args.function.lower()

        print(f"Generated {function_desc} truth table for conditioning")

    # Generate multiple graphs
    generated_graphs = []
    for i in range(args.graphs):
        # Track attempts to meet validity threshold if specified
        max_attempts = 10  # Limit number of generation attempts
        best_validity = 0.0
        best_graph = None

        for attempt in range(max_attempts):
            # Generate with or without conditioning
            adj_matrix, node_types = generate_aig(
                args.nodes,
                node_model,
                edge_model,
                input_size,
                edge_gen_function,
                mode,
                truth_table=truth_table if args.condition else None
            )

            # Create graph for validation
            G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

            # Calculate validity
            validity, _ = calculate_aig_validity(G, node_types)

            # Check if threshold is met or if this is the best so far
            if validity >= args.validity_threshold:
                # Save this graph and exit attempt loop
                best_graph = (G, adj_matrix, node_types)
                best_validity = validity
                print(f"Graph {i}: Valid graph (validity: {validity:.2%}) generated on attempt {attempt+1}")
                break
            elif validity > best_validity:
                # Track as best so far
                best_graph = (G, adj_matrix, node_types)
                best_validity = validity

        # Use best graph found
        if best_graph is not None:
            G, adj_matrix, node_types = best_graph
        else:
            print(f"Warning: Could not meet validity threshold after {max_attempts} attempts.")

        # Visualize and save graph
        condition_prefix = "conditioned_" if args.condition else "unconditioned_"
        output_file = os.path.join(args.output_dir, f'{condition_prefix}aig_{i}.png')
        G = visualize_aig(adj_matrix, node_types, output_file, truth_table=truth_table if args.condition else None)

        generated_graphs.append((G, adj_matrix, node_types))

    # Print statistics
    condition_status = "with truth table conditioning" if args.condition else "without conditioning"
    print(f"\nGenerated {args.graphs} AIGs {condition_status}:")
    for i, (G, adj_matrix, node_types) in enumerate(generated_graphs):
        print(f"\nGraph {i}:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print("  Node Type Distribution:")
        type_counts = {
            'ZERO': node_types.count(NODE_TYPES['ZERO']) if NODE_TYPES['ZERO'] in node_types else 0,
            'PI': node_types.count(NODE_TYPES['PI']) if NODE_TYPES['PI'] in node_types else 0,
            'AND': node_types.count(NODE_TYPES['AND']) if NODE_TYPES['AND'] in node_types else 0,
            'PO': node_types.count(NODE_TYPES['PO']) if NODE_TYPES['PO'] in node_types else 0
        }
        for type_name, count in type_counts.items():
            print(f"    {type_name}: {count}")

        # Calculate and print validity
        validity, validity_stats = calculate_aig_validity(G, node_types)
        print("  Validity Assessment:")
        print(f"    Overall Validity: {validity:.2%}")
        print(f"    AND Gates: {validity_stats['valid_and_gates']}/{validity_stats['and_gates']} valid ({validity_stats['and_validity']:.2%})")
        print(f"    PO Gates: {validity_stats['valid_po_gates']}/{validity_stats['po_gates']} valid ({validity_stats['po_validity']:.2%})")

    # If truth table was used, save it as well
    if args.condition and truth_table is not None:
        # Create a visualization of the truth table
        plt.figure(figsize=(10, 6))
        plt.imshow(table_original.cpu().numpy(), cmap='Blues', aspect='auto')
        plt.colorbar(label='Output Value')
        plt.title(f'Truth Table ({args.function})')
        plt.xlabel('Input Combination')
        plt.ylabel('Output Bit')
        truth_table_file = os.path.join(args.output_dir, f'truth_table_{args.function.lower()}.png')
        plt.savefig(truth_table_file)
        plt.close()

        print(f"\nTruth table visualization saved to {truth_table_file}")


if __name__ == '__main__':
    main()