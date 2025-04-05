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
import networkx as nx
import numpy as np
from aig_dataset import NODE_TYPES, EDGE_TYPES # Make sure constants are imported


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


def generate_aig(num_nodes, node_model, edge_model, input_size, edge_gen_function, mode, config, truth_table=None):
    """
    Generates an And-Inverter Graph and returns it as a NetworkX DiGraph.
    Includes node types and edge types as attributes.

    Args:
        num_nodes: Target number of nodes.
        node_model: Graph-level RNN model.
        edge_model: Edge-level model (MLP or RNN).
        input_size: 'm' value (max lookback).
        edge_gen_function: Function to generate edges (rnn_edge_gen or mlp_edge_gen).
        mode: Generation mode ('directed-multiclass').
        config: The model configuration dictionary.
        truth_table: Optional truth table tensor for conditioning.

    Returns:
        nx.DiGraph: The generated AIG with node/edge type attributes.
                    Returns None if generation fails or results in an empty graph.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    node_model.eval()
    edge_model.eval()

    # Determine sampling function based on mode
    use_edge_features = config['model']['GraphRNN'].get('edge_feature_len', 1) > 1
    if use_edge_features and mode == 'directed-multiclass':
        sample_fun = sample_softmax
    else:
        # Fallback or handle binary case if needed
        sample_fun = sample_bernoulli # Or raise error if only multiclass is expected

    # Initialize graph generation state
    adj_vec = torch.ones([1, 1, input_size, node_model.edge_feature_len], device=device) # SOS token input
    list_adj_vecs = []
    generated_node_types = [] # Store generated node types

    # --- Node Type Generation ---
    # Determine if node type prediction is enabled
    predict_node_types = config['model'].get('predict_node_types', False)

    node_model.reset_hidden()

    actual_num_nodes = 0
    for i in range(num_nodes): # Iterate up to num_nodes
        # --- Generate graph state vector (RNN hidden state) ---
        node_output = node_model(adj_vec, truth_table=truth_table if truth_table is not None else None)

        # Unpack node_output based on whether node types are predicted
        h = None # Hidden state for edge generation
        node_type_logits = None
        current_node_type = None

        if predict_node_types and isinstance(node_output, tuple):
            h, node_type_logits = node_output # h might be projection or hidden state
            # Sample node type from logits for the current step (index 0 as batch size is 1)
            # Use argmax for deterministic generation, or sample from softmax for stochastic
            current_node_type = torch.argmax(node_type_logits[0, 0, :]).item()
        else:
            h = node_output
            # Use a heuristic if not predicting types (e.g., mostly AND)
            # Or get type info from dataset if generating based on existing graph size? - No, generate freely.
            # Fallback: Assume AND for internal nodes if type prediction is off
            if i == 0: # First node maybe PI? Or handle differently?
                 current_node_type = NODE_TYPES['PI'] # Placeholder heuristic
            else:
                 current_node_type = NODE_TYPES['AND'] # Placeholder heuristic

        generated_node_types.append(current_node_type)
        actual_num_nodes += 1

        # --- Generate edges for this node ---
        # Only generate edges if it's not the first node (index 0)
        if i > 0:
            # Number of potential predecessors is min(current_index, input_size 'm')
            num_edges_to_generate = min(i, input_size)

            adj_vec_generated = edge_gen_function(
                edge_model,
                h, # Pass the hidden state
                num_edges=num_edges_to_generate,
                adj_vec_size=input_size, # M value
                sample_fun=sample_fun,
                attempts=1,
                truth_table=truth_table # Pass TT for conditioning edge model too
            )

            # --- Store the generated adjacency vector for this step ---
            # We need to store the connection information (which edge type to which previous node)
            # The shape is [1, 1, input_size, num_edge_classes]
            # Convert one-hot to class index for multi-class
            if use_edge_features and mode == 'directed-multiclass':
                 # Slice to get relevant connections, convert one-hot to class index
                adj_slice = adj_vec_generated[0, 0, :num_edges_to_generate, :].cpu().detach()
                # Find the index of the '1' in the last dimension
                edge_types_vec = torch.argmax(adj_slice, dim=-1).numpy() # Shape [num_edges_to_generate]
                list_adj_vecs.append(edge_types_vec)
            else: # Binary case (not expected for AIG)
                 list_adj_vecs.append(adj_vec_generated[0, 0, :num_edges_to_generate, 0].cpu().detach().numpy())

            # Check for early stopping (if model outputs only "no edge" for a while)
            # Check the *last* generated vector
            if len(list_adj_vecs) > 0 and np.all(list_adj_vecs[-1] == EDGE_TYPES['NONE']) and i > num_nodes // 2:
                 print(f"INFO: Early stopping at node {i+1} due to no edges.")
                 break # Stop generation

            # Prepare input for the next iteration (the generated edges)
            adj_vec = adj_vec_generated
        else:
            # For the very first node (i=0), add a placeholder empty connection list
            # Or handle the sequence start differently if GraphRNN expects non-empty first step
            list_adj_vecs.append(np.array([])) # No predecessors for the first node
            # Use the initial SOS token for the next step's adj_vec
            adj_vec = torch.ones([1, 1, input_size, node_model.edge_feature_len], device=device)


    # --- Construct NetworkX Graph ---
    if actual_num_nodes <= 1:
        print("Warning: Generated graph has <= 1 node. Returning None.")
        return None

    G = nx.DiGraph()
    node_mapping = list(range(actual_num_nodes)) # Simple integer node IDs

    # Add nodes with types
    for i in range(actual_num_nodes):
        G.add_node(node_mapping[i], type=generated_node_types[i])

    # Add edges based on list_adj_vecs
    for target_idx in range(1, actual_num_nodes): # Start from the second node
        # Connections for node target_idx (which is at index target_idx-1 in list_adj_vecs)
        edge_types_vec = list_adj_vecs[target_idx-1]
        num_potential_preds = len(edge_types_vec)

        # Iterate through potential predecessors based on 'm' lookback and sequence order
        for k in range(num_potential_preds):
            edge_type = int(edge_types_vec[k])

            # Skip "No Edge" type
            if edge_type == EDGE_TYPES['NONE']:
                continue

            # Determine the source node index in the original sequence
            # The k-th entry in edge_types_vec corresponds to the connection
            # from node target_idx to node target_idx - num_potential_preds + k
            # (since GraphRNN reverses the order in the adjacency vector input)
            # Let's re-verify GraphRNN adj vec format: adj_vec[k] connects current node i to node i-m+k
            # So, edge_types_vec[k] connects target_idx to target_idx - num_potential_preds + k
            # Wait, the input is reversed: padded[::-1].
            # Let's assume `list_adj_vecs[i-1]` contains edge types to nodes [i-m, ..., i-1] relative to node i.
            # The k-th element corresponds to node i-m+k.
            # Let m = input_size. num_potential_preds = min(target_idx, m)
            # Start index in original sequence: start = target_idx - num_potential_preds
            # k-th element connects target_idx to node start + k

            source_idx_relative_to_start = k
            source_node_absolute_idx = (target_idx - num_potential_preds) + source_idx_relative_to_start

            if 0 <= source_node_absolute_idx < target_idx:
                source_node = node_mapping[source_node_absolute_idx]
                target_node = node_mapping[target_idx]
                # Add edge with type attribute
                G.add_edge(source_node, target_node, type=edge_type)

    # Add graph attributes (optional, if needed from config)
    G.graph['n_inputs'] = config['model'].get('n_inputs')
    G.graph['n_outputs'] = config['model'].get('n_outputs')
    G.graph['name'] = 'generated_aig'

    return G


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