'''Code to use trained GraphRNN to generate a new AIG graph.'''

import argparse
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
import yaml  # For loading config
from model import *

# Assuming model definitions are in 'model.py'
# Make sure NODE_TYPES and EDGE_TYPES are accessible if needed later,
# but generation itself only uses edge types implicitly via edge_feature_len=3
# from aig_dataset import EDGE_TYPES # Only needed if you use names later

# --- Model Loading ---
# Adapted from load_aig_model_from_config / load_model_from_config
def load_aig_generator_model(model_path):
    """
    Load Node and Edge models from a checkpoint.
    MODIFIED: Also returns the max_level used by the node model's embedding.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model state from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    try:
        state = torch.load(model_path, map_location=device)
        config = state['config']
    except Exception as e:
        raise RuntimeError(f"Error loading model state: {e}")

    # Determine input_size based on config (BFS or TopSort)
    use_bfs = config['data'].get('use_bfs', False) # Use False default for TopSort AIGs
    if use_bfs:
        input_size = config['data'].get('m')
        if input_size is None:
            raise ValueError("Config 'data.m' is required when use_bfs is true.")
        print(f"INFO: Loading model for BFS mode. Input size (m): {input_size}")
    else:
        # Need max_node_count for TopSort. Determine from dataset.
        dataset_path = config['data'].get('graph_file')
        if not dataset_path or not os.path.exists(dataset_path):
             raise FileNotFoundError(f"Dataset file '{dataset_path}' specified in config not found.")
        try:
            from aig_dataset import AIGDataset # Import locally
            temp_dataset = AIGDataset(graph_file=dataset_path, training=False)
            max_node_count = temp_dataset.max_node_count
            if max_node_count <= 1: raise ValueError("Max node count from dataset <= 1")
            input_size = max_node_count - 1
            print(f"INFO: Loading model for TopSort mode. Input size (max_nodes-1): {input_size}")
            # --- Store dataset's max level ---
            max_level_from_dataset = temp_dataset.max_level
            print(f"INFO: Max level from dataset: {max_level_from_dataset}")
        except Exception as e:
             raise RuntimeError(f"Could not determine max_node_count/max_level for TopSort mode from {dataset_path}: {e}")

    # Model Setup
    edge_feature_len = config['model']['GraphRNN'].get('edge_feature_len', 3)
    if edge_feature_len != 3:
         print(f"Warning: Expected edge_feature_len=3 for AIGs, found {edge_feature_len} in config.")

    # --- Get max_level used during training from config ---
    # It was added to the GraphRNN args in main.py's setup_models
    max_level_config = config['model']['GraphRNN'].get('max_level')
    # Use the one from config if available, otherwise fallback to dataset's
    max_level_for_model = max_level_config if max_level_config is not None else max_level_from_dataset
    if max_level_for_model is None:
         print("Warning: max_level not found in config or dataset. Level embedding might not work.")
         max_level_for_model = 15 # Default or raise error? Defaulting to 0 for now.

    print(f"INFO: Using max_level={max_level_for_model} for model initialization/clamping.")

    node_model_args = config['model']['GraphRNN'].copy()
    node_model_args['input_size'] = input_size # Use determined size
    node_model_args['edge_feature_len'] = edge_feature_len
    node_model_args['max_level'] = max_level_for_model # Pass it during init
    # Ensure node type prediction and conditioning are OFF for generation
    node_model_args['predict_node_types'] = False
    node_model_args['use_conditioning'] = False
    node_model_args['tt_size'] = None

    if config['model']['edge_model'] == 'rnn':
        edge_model_args = config['model']['EdgeRNN'].copy()
        edge_model_args['edge_feature_len'] = edge_feature_len
        edge_model_args['tt_size'] = None # Conditioning off

        node_model = GraphLevelRNN(
            output_size=edge_model_args['hidden_size'], # RNN edge model needs output size
            **node_model_args
        ).to(device)
        edge_model = EdgeLevelRNN(**edge_model_args).to(device)
        edge_gen_function = rnn_edge_gen
        print("Using RNN edge model.")
    else: # Assume MLP
        edge_model_args = config['model']['EdgeMLP'].copy()
        edge_model_args['edge_feature_len'] = edge_feature_len
        edge_model_args['tt_size'] = None # Conditioning off
        edge_model_args['output_size'] = input_size # MLP output matches input size
        edge_model_args['input_size'] = node_model_args['hidden_size'] # MLP input is node hidden state

        node_model = GraphLevelRNN(
            output_size=None, # MLP edge model uses hidden state directly
            **node_model_args
        ).to(device)
        edge_model = EdgeLevelMLP(**edge_model_args).to(device)
        edge_gen_function = mlp_edge_gen
        print("Using MLP edge model.")

    # Load state dicts
    try:
        node_model.load_state_dict(state['node_model'])
        edge_model.load_state_dict(state['edge_model'])
        print("Model weights loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading model state dict: {e}")

    mode = config['model'].get('mode', 'directed-multiclass')
    if mode != 'directed-multiclass':
        print(f"Warning: Expected mode 'directed-multiclass', found '{mode}' in config.")

    # --- Return max_level_for_model ---
    return node_model, edge_model, input_size, edge_gen_function, mode, edge_feature_len, max_level_for_model



# --- Sampling Function ---
# def sample_softmax(x):
#     """Samples a one-hot vector from softmax probabilities of input logits x."""
#     # Ensure input is a torch tensor for softmax
#     if not isinstance(x, torch.Tensor):
#         x = torch.tensor(x)
#     # Detach from computation graph before converting to numpy
#     probabilities = torch.softmax(x.detach(), dim=0).cpu().numpy()
#     num_classes = probabilities.shape[0]
#     # Ensure probabilities sum to 1 (handle potential floating point issues)
#     probabilities = probabilities / np.sum(probabilities)
#     chosen_class = np.random.choice(range(num_classes), p=probabilities)
#     one_hot = np.zeros(num_classes)
#     one_hot[chosen_class] = 1
#     return one_hot
def sample_softmax(x, temperature=1.2): # Set back to 1.0 for standard softmax
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    # When temp=1.0, this is just standard softmax
    probabilities = torch.softmax(x.detach() / temperature, dim=0).cpu().numpy()
    num_classes = probabilities.shape[0]
    # Renormalization might still be needed due to floating point
    probabilities = probabilities / np.sum(probabilities)
    chosen_class = np.random.choice(range(num_classes), p=probabilities)
    one_hot = np.zeros(num_classes)
    one_hot[chosen_class] = 1
    return one_hot

# --- Edge Generation Functions (Simplified: No Conditioning) ---
def rnn_edge_gen(edge_rnn, h, num_edges, adj_vec_size, sample_fun, edge_feature_len, attempts=1):
    """Generates edges using RNN method (No conditioning)."""
    device = h.device  # Get device from hidden state tensor

    best_adj_vec = None
    best_edge_count = 0

    # Try multiple attempts to generate edges
    for attempt in range(attempts):
        adj_vec = torch.zeros([1, 1, adj_vec_size, edge_feature_len], device=device)
        edge_rnn.set_first_layer_hidden(h)  # Reset hidden state for each attempt

        # SOS token
        x = torch.ones([1, 1, edge_feature_len], device=device)

        for i in range(num_edges):
            # Calculate logits (RNN output before activation)
            try:
                logits = edge_rnn(x, return_logits=True)
            except TypeError:
                # Fallback if model doesn't support return_logits
                print("Warning: EdgeRNN doesn't support return_logits=True, using direct output.")
                logits = edge_rnn(x)

            # Sample from logits and assign one-hot vector to x and adj_vec
            sampled_one_hot = torch.tensor(sample_fun(logits[0, 0, :]), dtype=torch.float, device=device)
            x[0, 0, :] = sampled_one_hot
            adj_vec[0, 0, i, :] = sampled_one_hot

        # Count how many non-zero (edge class 1 or 2) edges were generated
        edge_count = torch.sum(torch.argmax(adj_vec[0, 0, :num_edges, :], dim=-1) > 0).item()

        # If this attempt has more edges than the best so far, keep it
        if best_adj_vec is None or edge_count > best_edge_count:
            best_adj_vec = adj_vec.clone()
            best_edge_count = edge_count

        # If we've found at least one edge, we can stop trying
        if edge_count > 0:
            break

    return best_adj_vec  # Return the best attempt

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
def generate_aig_structure(num_nodes, node_model, edge_model, input_size, edge_gen_function, mode, edge_feature_len, node_model_max_level):
    """
    Generates an And-Inverter Graph structure (nodes and edges with types).
    MODIFIED: Calculates and uses node levels during generation.

    Args:
        num_nodes: Target number of nodes.
        node_model: Graph-level RNN model.
        edge_model: Edge-level model (MLP or RNN).
        input_size: 'm' value (max lookback).
        edge_gen_function: Function to generate edges (rnn_edge_gen or mlp_edge_gen).
        mode: Generation mode ('directed-multiclass').
        edge_feature_len: Number of edge types (should be 3 for AIGs).
        node_model_max_level: Max level value used in node_model's embedding.

    Returns:
        nx.DiGraph: The generated AIG with edge 'type' attributes (1=Regular, 2=Inverted).
                    Returns None if generation fails or results in an empty graph.
    """
    if mode != 'directed-multiclass':
        print(f"Warning: This generator is designed for 'directed-multiclass' mode, but found '{mode}'. Results may be incorrect.")

    device = next(node_model.parameters()).device # Get device from model

    node_model.eval()
    edge_model.eval()

    sample_fun = sample_softmax # Use softmax sampling for multiclass edges

    # Initialize with SOS token
    adj_vec_input = torch.ones([1, 1, input_size, edge_feature_len], device=device)
    list_edge_type_indices = []
    node_model.reset_hidden()

    # --- NEW: Track generated graph and node levels ---
    G_generated_so_far = nx.DiGraph()
    node_levels_so_far = { -1: -1 } # Use -1 for pre-SOS state, node 0 will have level 0
    # -------------------------------------------------

    actual_num_nodes = 0
    consecutive_no_edge_nodes = 0

    with torch.no_grad():
        for i in range(num_nodes): # Generate node i (0-indexed)
            # --- Determine level for the *input* step (node i) ---
            # The input adj_vec_input represents connections *to* node i-1.
            # The RNN state update processes this input to represent node i-1's state.
            # So, the level embedding should correspond to node i-1.
            level_for_input_node = node_levels_so_far.get(i - 1, -1) # Get level of node i-1, default -1
            # For the very first step (i=0), input is SOS, level is 0.
            current_input_level = 0 if i == 0 else level_for_input_node + 1 # Level of node i

            # Clamp level to be within the embedding range [0, max_level]
            clamped_level = max(0, min(current_input_level, node_model_max_level))
            current_level_tensor = torch.tensor([[clamped_level]], dtype=torch.long, device=device)
            # --- End Level Calculation for Input ---

            # --- Generate graph state vector (RNN hidden state for node i) ---
            # Pass the level tensor corresponding to the node being processed
            h = node_model(adj_vec_input, levels=current_level_tensor)
            # h now represents the state *after* processing node i's connections (or SOS)
            # It will be used to generate edges for node i+1
            # --- End State Generation ---

            # Add node i to the tracked graph
            G_generated_so_far.add_node(i)
            node_levels_so_far[i] = current_input_level # Store the calculated level for node i
            actual_num_nodes += 1


            # --- Generate edges *for* node i+1 (based on hidden state h from node i) ---
            # Note: The loop generates node i, then generates edges for i+1.
            # We need num_nodes iterations to generate nodes 0 to num_nodes-1.
            # The edge generation happens *after* node i is processed.

            if i < num_nodes - 1: # Only generate edges if not the last target node
                num_edges_to_generate = min(i + 1, input_size) # Edges for node i+1 connect to nodes 0..i

                adj_vec_generated = edge_gen_function(
                    edge_model,
                    h, # Hidden state after processing node i
                    num_edges=num_edges_to_generate,
                    adj_vec_size=input_size, # M value
                    sample_fun=sample_fun,
                    edge_feature_len=edge_feature_len,
                    attempts=5,
                )

                # --- Store edge indices and update G_generated_so_far ---
                adj_slice = adj_vec_generated[0, 0, :num_edges_to_generate, :]
                edge_indices_vec = torch.argmax(adj_slice, dim=-1).cpu().numpy()
                list_edge_type_indices.append(edge_indices_vec) # Store edges for node i+1

                # Add edges to the tracked graph G_generated_so_far
                target_node_for_edges = i + 1
                max_pred_level_for_next_node = -1
                for k in range(num_edges_to_generate):
                    edge_type = int(edge_indices_vec[k])
                    if edge_type > 0: # If it's a Regular (1) or Inverted (2) edge
                         # Source node index calculation (remains the same logic)
                         source_node_idx = (target_node_for_edges - num_edges_to_generate) + k
                         if 0 <= source_node_idx < target_node_for_edges:
                             G_generated_so_far.add_edge(source_node_idx, target_node_for_edges, type=edge_type)
                             # Track max predecessor level for the *next* node's level calculation
                             max_pred_level_for_next_node = max(max_pred_level_for_next_node, node_levels_so_far.get(source_node_idx, -1))

                # Calculate and store level for the *next* node (i+1)
                level_of_next_node = max_pred_level_for_next_node + 1
                node_levels_so_far[i+1] = level_of_next_node
                # --- End Edge Storage and Level Calc for Next Node ---


                # Check for early stopping (based on edges generated *for node i+1*)
                if np.all(edge_indices_vec == 0):
                    consecutive_no_edge_nodes += 1
                    if consecutive_no_edge_nodes >= 3 and (i+1) > num_nodes // 3: # Check against i+1
                        print(
                            f"INFO: Early stopping at step {i+1} (node {i+1}) after {consecutive_no_edge_nodes} consecutive 'No Edge' steps."
                        )
                        # No need to decrement actual_num_nodes here, just break
                        break
                else:
                    consecutive_no_edge_nodes = 0

                # Prepare input for the next iteration (processing node i+1)
                adj_vec_input = adj_vec_generated

            else: # Last iteration (i == num_nodes - 1), node generated, no more edges needed
                 pass


    # --- Construct Final NetworkX Graph ---
    # G_generated_so_far already holds the graph structure and edge types
    if G_generated_so_far.number_of_nodes() <= 1:
        print("Warning: Generated graph has <= 1 node after generation/stopping. Returning None.")
        return None

    # Add level attribute to nodes (optional, but useful)
    for node_id, level in node_levels_so_far.items():
        if node_id >= 0 and node_id in G_generated_so_far: # Check if node exists
            G_generated_so_far.nodes[node_id]['level'] = level


    # Optional: Remove isolated nodes
    isolated_nodes = list(nx.isolates(G_generated_so_far))
    if isolated_nodes:
        print(f"Removing {len(isolated_nodes)} isolated nodes: {isolated_nodes}")
        G_generated_so_far.remove_nodes_from(isolated_nodes)

    if G_generated_so_far.number_of_nodes() == 0:
        print("Warning: Graph became empty after removing isolated nodes.")
        return None

    return G_generated_so_far

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