'''Code to use trained GraphRNN to generate a new AIG graph.'''

import argparse
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
import yaml  # For loading config

# Assuming model definitions are in 'model.py'
# Make sure NODE_TYPES and EDGE_TYPES are accessible if needed later,
# but generation itself only uses edge types implicitly via edge_feature_len=3
# from aig_dataset import EDGE_TYPES # Only needed if you use names later

# --- Model Loading ---
# Adapted from load_aig_model_from_config / load_model_from_config
def load_aig_generator_model(model_path):
    """Load Node and Edge models from a checkpoint."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model state from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    try:
        state = torch.load(model_path, map_location=device)
        config = state['config']
    except Exception as e:
        raise RuntimeError(f"Error loading model state: {e}")

    # Import necessary model classes (ensure these are defined in model.py)
    from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP

    input_size = config['data']['m']
    # Assume edge_feature_len is correctly set in the config used for training
    edge_feature_len = config['model']['GraphRNN'].get('edge_feature_len', 3)
    if edge_feature_len != 3:
         print(f"Warning: Expected edge_feature_len=3 for AIGs (None, Regular, Inverted), found {edge_feature_len} in config.")

    node_model_args = config['model']['GraphRNN']
    node_model_args['edge_feature_len'] = edge_feature_len
    # Ensure node type prediction and conditioning are OFF for generation
    node_model_args['predict_node_types'] = False
    node_model_args['use_conditioning'] = False
    node_model_args['tt_size'] = None

    if config['model']['edge_model'] == 'rnn':
        edge_model_args = config['model']['EdgeRNN']
        edge_model_args['edge_feature_len'] = edge_feature_len
        edge_model_args['use_conditioning'] = False
        edge_model_args['tt_size'] = None

        node_model = GraphLevelRNN(
            input_size=input_size,
            output_size=edge_model_args['hidden_size'], # RNN edge model needs output size
            **node_model_args
        ).to(device)
        edge_model = EdgeLevelRNN(**edge_model_args).to(device)
        edge_gen_function = rnn_edge_gen
        print("Using RNN edge model.")
    else: # Assume MLP
        edge_model_args = config['model']['EdgeMLP']
        edge_model_args['edge_feature_len'] = edge_feature_len
        edge_model_args['use_conditioning'] = False
        edge_model_args['tt_size'] = None

        node_model = GraphLevelRNN(
            input_size=input_size,
            output_size=None, # MLP edge model uses hidden state directly
            **node_model_args
        ).to(device)
        edge_model = EdgeLevelMLP(
            input_size=node_model_args['hidden_size'], # MLP takes hidden state from GraphRNN
            output_size=input_size, # MLP predicts for 'm' possible predecessors
            **edge_model_args
        ).to(device)
        edge_gen_function = mlp_edge_gen
        print("Using MLP edge model.")

    try:
        node_model.load_state_dict(state['node_model'])
        edge_model.load_state_dict(state['edge_model'])
        print("Model weights loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading model state dict: {e}")

    use_bfs = config['data'].get('use_bfs', True)  # Default to True if missing

    # --- START FIX ---
    if use_bfs:
        input_size = config['data']['m']
        print(f"INFO: Loading model for BFS mode. Input size (m): {input_size}")
    else:
        # Need max_node_count for TopSort.
        # OPTION 1: Pass max_node_count as an argument
        # OPTION 2: Determine it by loading the dataset here (less efficient)
        try:
            # Example: you might need to load the dataset to get this info
            from aig_dataset import AIGDataset
            # NOTE: This requires the dataset path from config
            temp_dataset = AIGDataset(graph_file=config['data']['graph_file'], m=None, training=False, use_bfs=False)
            max_node_count = temp_dataset.max_node_count
            if max_node_count <= 1: raise ValueError("Max node count <= 1")
            input_size = max_node_count - 1
            print(f"INFO: Loading model for TopSort mode. Input size (max_nodes-1): {input_size}")
        except Exception as e:
            raise RuntimeError(f"Could not determine max_node_count for TopSort mode: {e}")

    # Mode should be 'directed-multiclass' for AIGs
    mode = config['model'].get('mode', 'directed-multiclass')
    if mode != 'directed-multiclass':
        print(f"Warning: Expected mode 'directed-multiclass', found '{mode}' in config.")

    return node_model, edge_model, input_size, edge_gen_function, mode, edge_feature_len


# --- Sampling Function ---
def sample_softmax(x):
    """Samples a one-hot vector from softmax probabilities of input logits x."""
    # Ensure input is a torch tensor for softmax
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    # Detach from computation graph before converting to numpy
    probabilities = torch.softmax(x.detach(), dim=0).cpu().numpy()
    num_classes = probabilities.shape[0]
    # Ensure probabilities sum to 1 (handle potential floating point issues)
    probabilities = probabilities / np.sum(probabilities)
    chosen_class = np.random.choice(range(num_classes), p=probabilities)
    one_hot = np.zeros(num_classes)
    one_hot[chosen_class] = 1
    return one_hot

# --- Edge Generation Functions (Simplified: No Conditioning) ---
def rnn_edge_gen(edge_rnn, h, num_edges, adj_vec_size, sample_fun, edge_feature_len, attempts=None):
    """Generates edges using RNN method (No conditioning)."""
    device = h.device # Get device from hidden state tensor
    adj_vec = torch.zeros([1, 1, adj_vec_size, edge_feature_len], device=device)

    edge_rnn.set_first_layer_hidden(h)

    # SOS token
    x = torch.ones([1, 1, edge_feature_len], device=device)

    for i in range(num_edges):
        # Calculate logits (RNN output before activation)
        # Assuming edge_rnn.forward can take return_logits=True
        try:
             logits = edge_rnn(x, return_logits=True)
        except TypeError:
             # Fallback if model doesn't support return_logits
             print("Warning: EdgeRNN doesn't support return_logits=True, using direct output.")
             logits = edge_rnn(x)

        # Sample from logits and assign one-hot vector to x and adj_vec
        # sample_fun returns numpy array, convert back to tensor
        sampled_one_hot = torch.tensor(sample_fun(logits[0, 0, :]), dtype=torch.float, device=device)
        x[0, 0, :] = sampled_one_hot
        adj_vec[0, 0, i, :] = sampled_one_hot

    return adj_vec

def mlp_edge_gen(edge_mlp, h, num_edges, adj_vec_size, sample_fun, edge_feature_len, attempts=1):
    """Generates edges using MLP method (No conditioning)."""
    device = h.device # Get device from hidden state tensor
    adj_vec = torch.zeros([1, 1, adj_vec_size, edge_feature_len], device=device)

    # Calculate logits (MLP output before activation)
    # Assuming edge_mlp.forward can take return_logits=True
    try:
        edge_logits = edge_mlp(h, return_logits=True) # Shape: [1, 1, adj_vec_size, edge_feature_len]
    except TypeError:
        print("Warning: EdgeMLP doesn't support return_logits=True, using direct output.")
        edge_logits = edge_mlp(h)


    # Update adj_vec with the sampled value from each edge logit distribution
    for _ in range(attempts):
        for i in range(num_edges):
             # Sample using logits for the i-th potential predecessor
             # sample_fun returns numpy array, convert back to tensor
             sampled_one_hot = torch.tensor(sample_fun(edge_logits[0, 0, i, :]), dtype=torch.float, device=device)
             adj_vec[0, 0, i, :] = sampled_one_hot
        # If we generated all zeros (only 'No Edge' sampled), try again if attempts left.
        # Check if any class other than 0 (assuming EDGE_TYPES['NONE'] == 0) was sampled.
        if (torch.argmax(adj_vec[0, 0, :num_edges, :], dim=-1) > 0).any():
            break

    return adj_vec

# --- Core AIG Generation Function ---
def generate_aig_structure(num_nodes, node_model, edge_model, input_size, edge_gen_function, mode, edge_feature_len):
    """
    Generates an And-Inverter Graph structure (nodes and edges with types).

    Args:
        num_nodes: Target number of nodes.
        node_model: Graph-level RNN model.
        edge_model: Edge-level model (MLP or RNN).
        input_size: 'm' value (max lookback).
        edge_gen_function: Function to generate edges (rnn_edge_gen or mlp_edge_gen).
        mode: Generation mode ('directed-multiclass').
        edge_feature_len: Number of edge types (should be 3 for AIGs).

    Returns:
        nx.DiGraph: The generated AIG with edge 'type' attributes (1=Regular, 2=Inverted).
                    Returns None if generation fails or results in an empty graph.
    """
    if mode != 'directed-multiclass':
        print(f"Warning: This generator is designed for 'directed-multiclass' mode, but found '{mode}'. Results may be incorrect.")

    device = next(node_model.parameters()).device # Get device from model

    node_model.eval()
    edge_model.eval()

    # Use softmax sampling for multiclass edges
    sample_fun = sample_softmax

    # Initialize with SOS token (batch_size=1, seq_len=1)
    # Shape: [1, 1, input_size, edge_feature_len]
    adj_vec_input = torch.ones([1, 1, input_size, edge_feature_len], device=device)

    # Store the generated edge type *indices* for each node
    list_edge_type_indices = []

    node_model.reset_hidden()
    actual_num_nodes = 0

    with torch.no_grad(): # Ensure no gradients are computed during generation
        for i in range(num_nodes): # Iterate up to num_nodes
            # --- Generate graph state vector (RNN hidden state) ---
            # Node model should not predict node types or use conditioning here
            h = node_model(adj_vec_input) # h shape depends on edge model type

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
                    edge_feature_len=edge_feature_len,
                    attempts=1, # Can increase if needed
                ) # Shape: [1, 1, input_size, edge_feature_len]

                # --- Store the generated edge type indices for this step ---
                # Convert the generated one-hot vectors to class indices (0, 1, or 2)
                # Slice to get relevant connections: [1, 1, num_edges_to_generate, edge_feature_len]
                adj_slice = adj_vec_generated[0, 0, :num_edges_to_generate, :]
                # Find the index of the '1' in the last dimension -> [num_edges_to_generate]
                edge_indices_vec = torch.argmax(adj_slice, dim=-1).cpu().numpy()
                list_edge_type_indices.append(edge_indices_vec)

                # Check for early stopping (if model outputs only "no edge" class=0 for a while)
                if len(list_edge_type_indices) > 0 and np.all(list_edge_type_indices[-1] == 0) and i > num_nodes // 2:
                     print(f"INFO: Early stopping at node {i+1} due to only generating 'No Edge'.")
                     actual_num_nodes -= 1 # Decrement count as this node wasn't fully added
                     break # Stop generation

                # Prepare input for the next iteration (the generated edges)
                adj_vec_input = adj_vec_generated
            else:
                # For the very first node (i=0), add a placeholder empty connection list
                list_edge_type_indices.append(np.array([], dtype=int)) # No predecessors for the first node
                # Use the initial SOS token for the next step's input
                adj_vec_input = torch.ones([1, 1, input_size, edge_feature_len], device=device)


    # --- Construct NetworkX Graph ---
    if actual_num_nodes <= 1:
        print("Warning: Generated graph has <= 1 node. Returning None.")
        return None

    G = nx.DiGraph()
    # Add nodes first (simple integer IDs)
    for i in range(actual_num_nodes):
        G.add_node(i) # Add nodes without type attribute

    # Add edges based on list_edge_type_indices
    # list_edge_type_indices[k] contains edge info *for* node k+1
    for target_idx in range(1, actual_num_nodes): # Start from the second node (index 1)
        edge_indices_vec = list_edge_type_indices[target_idx-1] # Indices for edges *to* target_idx
        num_potential_preds = len(edge_indices_vec)

        # Iterate through potential predecessors based on 'm' lookback and sequence order
        # The k-th entry in edge_indices_vec corresponds to the connection attempt
        # from potential source node (target_idx - num_potential_preds + k) to target_idx.
        # This needs to precisely match the dataset creation logic.
        for k in range(num_potential_preds):
            edge_type = int(edge_indices_vec[k]) # Should be 0, 1, or 2

            # Skip "No Edge" type (assuming index 0 is NO_EDGE)
            if edge_type == 0: # Corresponds to EDGE_TYPES['NONE']
                continue

            # Determine the source node index
            # Source node index = first possible predecessor index + k
            first_possible_pred_idx = target_idx - num_potential_preds
            source_node_idx = first_possible_pred_idx + k

            # Ensure source node index is valid and precedes target node
            if 0 <= source_node_idx < target_idx:
                # Add the directed edge with its type (1 or 2)
                G.add_edge(source_node_idx, target_idx, type=edge_type)
            # else:
                # print(f"Debug: Invalid source index {source_node_idx} for target {target_idx} (k={k}, num_preds={num_potential_preds})")


    # Optional: Remove isolated nodes (nodes with no incoming or outgoing edges)
    # isolated_nodes = list(nx.isolates(G))
    # if isolated_nodes:
    #     print(f"Removing {len(isolated_nodes)} isolated nodes: {isolated_nodes}")
    #     G.remove_nodes_from(isolated_nodes)

    if G.number_of_nodes() == 0:
        print("Warning: Graph became empty after removing isolated nodes.")
        return None

    return G

def visualize_aig_structure(G, output_file='generated_aig_structure.png'):
    """Visualize the generated AIG structure."""
    if G is None:
        print("Cannot visualize None graph.")
        return

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42) # Use a layout algorithm

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)

    # Draw edges with styles based on 'type' attribute
    regular_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 1]
    inverted_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 2]
    other_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') not in [1, 2]]

    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=1.5, edge_color='black', style='solid', arrowsize=15)
    nx.draw_networkx_edges(G, pos, edgelist=inverted_edges, width=1.5, edge_color='red', style='dashed', arrowsize=15)
    if other_edges:
        print(f"Warning: Found edges with unexpected types: {other_edges}")
        nx.draw_networkx_edges(G, pos, edgelist=other_edges, width=1.0, edge_color='gray', style='dotted', arrowsize=15)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', linestyle='solid', label='Regular Edge (type 1)'),
        plt.Line2D([0], [0], color='red', linestyle='dashed', label='Inverted Edge (type 2)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title('Generated AIG Structure')
    plt.axis('off') # Turn off axis
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Generated graph visualization saved to {output_file}")


# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate AIG structure from a trained GraphRNN model")
    parser.add_argument('model_path', help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('-n', '--nodes', type=int, default=50, help='Target number of nodes for the generated graph')
    parser.add_argument('-o', '--output', type=str, default='generated_aig.png', help='Output file name for the visualization')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed for generation')

    args = parser.parse_args()

    if args.seed is not None:
        print(f"Setting random seed to: {args.seed}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    try:
        # Load models and config
        node_model, edge_model, input_size, edge_gen_func, mode, edge_feature_len = load_aig_generator_model(args.model_path)

        # Generate the graph structure
        print(f"Generating AIG structure with target nodes: {args.nodes}...")
        generated_graph = generate_aig_structure(
            num_nodes=args.nodes,
            node_model=node_model,
            edge_model=edge_model,
            input_size=input_size,
            edge_gen_function=edge_gen_func,
            mode=mode,
            edge_feature_len=edge_feature_len
        )

        if generated_graph:
            print(f"Successfully generated graph with {generated_graph.number_of_nodes()} nodes and {generated_graph.number_of_edges()} edges.")
            # Visualize the generated graph
            visualize_aig_structure(generated_graph, args.output)
        else:
            print("Graph generation failed or resulted in an empty graph.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")