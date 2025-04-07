"""
Generation module for AIG (And-Inverter Graphs) using trained models.
"""

import numpy as np
import torch
import networkx as nx
import os
from torch.distributions import Categorical

# Import necessary model classes and setup function
from model import *  # Import all model classes
from aig_dataset import _calculate_levels, EDGE_TYPES, NUM_EDGE_FEATURES

def aig_seq_to_nx(adj_seq_list: list, edge_feature_len: int) -> nx.DiGraph:
    """
    Converts the sequence of adjacency vectors (output from generation)
    into a NetworkX DiGraph for an AIG.

    Args:
        adj_seq_list: List of numpy arrays, where each array represents the
                      incoming edge types for a node (shape: [m, edge_feature_len]).
                      Assumes topological order (node 1 connects to node 0, node 2 to 0,1 etc.)
        edge_feature_len: Number of edge features (should be 3 for AIG).

    Returns:
        A NetworkX DiGraph representing the AIG.
    """
    if not adj_seq_list:
        return nx.DiGraph()

    num_nodes = len(adj_seq_list) + 1  # +1 for the implicit node 0 (SOS/PI placeholder)
    g = nx.DiGraph()
    g.add_nodes_from(range(num_nodes))  # Nodes 0 to num_nodes-1

    effective_m = adj_seq_list[0].shape[0]  # Get m from the shape

    for target_node_idx, adj_vec in enumerate(adj_seq_list, start=1):
        # adj_vec shape: [m, edge_feature_len]
        # Represents connections *to* target_node_idx
        num_possible_preds = min(target_node_idx, effective_m)

        # The adj_vec is reversed relative to node indices 0..i-1
        # adj_vec[0] corresponds to connection from node target_node_idx - 1
        # adj_vec[k] corresponds to connection from node target_node_idx - 1 - k
        for k in range(num_possible_preds):
            source_node_idx = target_node_idx - 1 - k
            if source_node_idx < 0: continue  # Should not happen with correct slicing

            edge_type_probs = adj_vec[k, :]  # Probabilities/logits for this edge
            # Assuming the adj_vec contains the *sampled* one-hot representation
            edge_type = np.argmax(edge_type_probs).item()  # Get the sampled class index

            if edge_type != EDGE_TYPES["NONE"]:
                g.add_edge(source_node_idx, target_node_idx, type=edge_type)

    # Basic node type inference based on topology (can be refined in validation)
    for n in g.nodes():
         in_deg = g.in_degree(n)
         out_deg = g.out_degree(n)
         if in_deg == 0:
             g.nodes[n]['inferred_type'] = 'PI'  # Or ZERO
         elif in_deg == 2:
             g.nodes[n]['inferred_type'] = 'AND'
         elif out_deg == 0 and in_deg > 0:  # Sink nodes (might be POs)
             g.nodes[n]['inferred_type'] = 'PO'
         else:  # Could be intermediate, buffer, or invalid fan-in
              g.nodes[n]['inferred_type'] = 'UNKNOWN'  # Or based on in_deg == 1?

    return g


def rnn_edge_gen_aig(edge_rnn, h, num_edges_to_sample, effective_m, temperature=1.0, device='cpu'):
    """
    Generates AIG edges for one node using RNN method with categorical sampling.

    Args:
        edge_rnn: Trained EdgeLevel RNN/LSTM model.
        h: Hidden state from the NodeLevel model for the current node step.
           Shape typically [1, 1, node_hidden_size] for MLP input,
           or needs to be set via set_first_layer_hidden for EdgeRNN/LSTM.
        num_edges_to_sample: The number of potential incoming edges to sample (min(current_node_idx, effective_m)).
        effective_m: The maximum number of predecessors considered (max_nodes - 1).
        temperature: Sampling temperature (higher -> more random).
        device: Torch device.

    Returns:
        torch.Tensor: Sampled adjacency vector [1, 1, effective_m, NUM_EDGE_FEATURES] (one-hot encoded).
    """
    adj_vec_sampled = torch.zeros([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
    # One-hot encode the "NONE" type for padding
    adj_vec_sampled[:, :, :, EDGE_TYPES["NONE"]] = 1.0

    # --- Handle setting initial hidden state based on model type ---
    # This depends on how the NodeModel output `h` is structured and if EdgeRNN/LSTM is used.
    # Assuming EdgeLevelRNN/LSTM classes have `set_first_layer_hidden`
    if hasattr(edge_rnn, 'set_first_layer_hidden') and callable(getattr(edge_rnn, 'set_first_layer_hidden')):
         try:
            # We need the hidden state corresponding *to this node* from the node model's sequence output
            # Assuming h passed here is the *correct* hidden state slice for this step.
             edge_rnn.set_first_layer_hidden(h.squeeze(0))  # Remove batch dim if necessary
         except Exception as e:
             print(f"Warning: Could not set hidden state for EdgeRNN/LSTM: {e}")
             # Decide how to proceed: maybe zero hidden state or raise error
             return adj_vec_sampled  # Return default if hidden state fails
    # --- End Hidden State Handling ---

    # SOS token (one-hot encoded for NONE type, assuming SOS implies no actual edge)
    # Shape: [batch=1, seq=1, features=3]
    x = torch.zeros([1, 1, NUM_EDGE_FEATURES], device=device)
    x[:, :, EDGE_TYPES["NONE"]] = 1.0  # Start with SOS = NONE

    sampled_edges = []
    for i in range(num_edges_to_sample):
        # Get logits for the next edge type
        # EdgeRNN expects packed sequences usually, but here we run step-by-step
        # Shape of logits: [batch=1, seq=1, features=3]
        logits = edge_rnn(x, return_logits=True)  # Pass x, internal hidden state advances

        # Apply temperature and sample
        scaled_logits = logits.squeeze(1) / temperature  # Shape: [1, 3]
        dist = Categorical(logits=scaled_logits)
        sampled_type_idx = dist.sample()  # Tensor containing index [0, 1, or 2]

        # Create next input (one-hot encoding of sampled type)
        x.zero_()
        x[0, 0, sampled_type_idx.item()] = 1.0
        sampled_edges.append(x.clone().squeeze().cpu().numpy())  # Store sampled one-hot vector

    # Place sampled edges into the correct (reversed) positions in adj_vec_sampled
    if sampled_edges:
        sampled_stack = torch.tensor(np.stack(sampled_edges[::-1]), device=device)  # Reverse list before stacking
        num_sampled = sampled_stack.shape[0]
        adj_vec_sampled[0, 0, :num_sampled, :] = sampled_stack  # Fill from the start (represents nearest predecessors)

    return adj_vec_sampled


def mlp_edge_gen_aig(edge_mlp, h, num_edges_to_sample, effective_m, temperature=1.0, device='cpu'):
    """
    Generates AIG edges for one node using MLP method with categorical sampling.

    Args:
        edge_mlp: Trained EdgeLevel MLP model.
        h: Output from the NodeLevel model for the current node step.
           Shape typically [1, 1, node_hidden_size].
        num_edges_to_sample: The number of potential incoming edges to sample (min(current_node_idx, effective_m)).
        effective_m: The maximum number of predecessors considered (max_nodes - 1).
        temperature: Sampling temperature.
        device: Torch device.

    Returns:
        torch.Tensor: Sampled adjacency vector [1, 1, effective_m, NUM_EDGE_FEATURES] (one-hot encoded).
    """
    adj_vec_sampled = torch.zeros([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
    # One-hot encode the "NONE" type for padding
    adj_vec_sampled[:, :, :, EDGE_TYPES["NONE"]] = 1.0

    # Get logits for all potential edges at once
    # edge_mlp output shape: [batch=1, seq=1, output_m, features=3]
    # Assuming h is shaped [1, 1, node_hidden_size]
    all_logits = edge_mlp(h, return_logits=True)
    # Squeeze batch and sequence dims: [output_m, features=3]
    logits_for_sampling = all_logits.squeeze(0).squeeze(0)

    # Sample edges for the relevant predecessors
    if num_edges_to_sample > 0:
        # Select logits for the edges we need to sample
        # Logits are ordered corresponding to potential predecessors 0 to m-1
        # We need the *last* `num_edges_to_sample` logits (reversed order)
        relevant_logits = logits_for_sampling[:num_edges_to_sample, :]  # Corresponds to preds i-1 down to i-num_edges

        # Apply temperature and sample
        scaled_logits = relevant_logits / temperature
        dist = Categorical(logits=scaled_logits)
        sampled_type_indices = dist.sample()  # Shape: [num_edges_to_sample]

        # Create one-hot encodings
        one_hot_sampled = torch.nn.functional.one_hot(sampled_type_indices, num_classes=NUM_EDGE_FEATURES).float()

        # Place into the adjacency vector (correct positions)
        # Needs to be reversed to match RNN generation order if needed, but MLP generates all at once.
        # The adj_vec represents inputs from node i-1, i-2,... down to i-m
        # The logits[:num_edges] correspond to these same nodes.
        adj_vec_sampled[0, 0, :num_edges_to_sample, :] = one_hot_sampled

    return adj_vec_sampled


def generate_aig(num_nodes_target, node_model, edge_model, effective_m, max_level_model,
                 edge_gen_fn, device,
                 temperature=1.0, max_steps=None, eos_patience=10):
    """
    Generates a single And-Inverter Graph (AIG).

    Args:
        num_nodes_target: Desired number of nodes (generation might stop earlier/later).
        node_model: Trained NodeLevel model.
        edge_model: Trained EdgeLevel model.
        effective_m: Max predecessors considered (max_nodes_train - 1).
        max_level_model: Max level the node_model was trained with (for level embedding).
        edge_gen_fn: The edge generation function to use (rnn_edge_gen_aig or mlp_edge_gen_aig).
        device: Torch device.
        temperature: Sampling temperature.
        max_steps: Optional maximum generation steps (nodes to add).
        eos_patience: Stop if no actual edges (type 1 or 2) are added for this many consecutive steps.

    Returns:
        nx.DiGraph: The generated AIG as a NetworkX graph.
        int: The maximum level reached in the generated graph.
    """
    node_model.eval()
    edge_model.eval()

    # --- Initialization ---
    adj_vec_current = torch.zeros([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
    adj_vec_current[:, :, :, EDGE_TYPES["NONE"]] = 1.0
    list_adj_vecs_sampled = []
    node_model.reset_hidden()
    current_levels = {0: 0}
    max_level_gen = 0
    uses_level_embedding = hasattr(node_model, 'level_embedding') and node_model.level_embedding is not None
    no_real_edge_steps_in_a_row = 0  # Initialize patience counter

    # Determine max generation steps
    if max_steps is None:
        # Make max_steps slightly larger to allow patience to trigger before hard limit
        max_steps = int(num_nodes_target * 2.5) + eos_patience
    if num_nodes_target <= 1:
        return nx.DiGraph(), 0

    # --- Generation Loop ---
    for i in range(1, max_steps):
        current_node_idx = i
        # --- Node Level Hidden State Calculation ---
        level_for_input = None
        if uses_level_embedding:
            max_pred_level = -1
            # Avoid repeatedly building the graph if performance is critical
            # Only need predecessors of current_node_idx
            # However, building it is simpler for now
            temp_g_so_far = aig_seq_to_nx(list_adj_vecs_sampled, NUM_EDGE_FEATURES)
            # Need to add the node we are about to generate to check predecessors
            # This assumes node indices are contiguous from 0
            if not temp_g_so_far.has_node(current_node_idx):
                 temp_g_so_far.add_node(current_node_idx)  # Add node to check preds even if no edges yet

            # Check predecessors using the structure *before* adding edges for this step
            preds_of_current = []
            if current_node_idx in temp_g_so_far:  # Check if node exists before getting preds
                 # Logic based on list_adj_vecs_sampled index implies connectivity
                 num_preds_possible = min(current_node_idx, effective_m)
                 for k in range(num_preds_possible):
                      source_node_idx = current_node_idx - 1 - k
                      # Check the *previous* step's output if available
                      if list_adj_vecs_sampled:
                           # We need to look at the connectivity *to* current_node_idx
                           # which was determined by the vector added at step i-1
                           # Let's stick to the simpler graph build for level calc for now
                           pass  # Simplified level calc below uses graph structure

            # Simplified level calculation: max level of nodes < current_node_idx connected to it
            # Re-calculate levels on the temp graph at each step (less efficient but simpler)
            if temp_g_so_far.number_of_nodes() > 1 and nx.is_directed_acyclic_graph(temp_g_so_far):
                 temp_levels, _ = _calculate_levels(temp_g_so_far)
                 current_levels = temp_levels  # Update levels based on current graph
                 max_pred_level = -1
                 # Find max level among nodes that *could* connect to current_node_idx
                 num_preds_possible = min(current_node_idx, effective_m)
                 for k in range(num_preds_possible):
                     source_node_idx = current_node_idx - 1 - k
                     if source_node_idx in current_levels:
                          max_pred_level = max(max_pred_level, current_levels.get(source_node_idx, -1))
            else:
                 # Fallback if not DAG or too small
                 max_pred_level = current_levels.get(current_node_idx-1, -1)  # Approx level based on previous node

            current_node_level = max_pred_level + 1
            # Only add to current_levels if positive, avoid overwriting node 0?
            if current_node_idx > 0:
                current_levels[current_node_idx] = current_node_level
            max_level_gen = max(max_level_gen, current_node_level)

            level_clamped = min(current_node_level, max_level_model)
            level_for_input = torch.tensor([[level_clamped]], dtype=torch.long, device=device)

        # --- Node model forward pass ---
        node_output = node_model(adj_vec_current, levels=level_for_input)
        h_node = node_output[0] if isinstance(node_output, tuple) else node_output

        # --- Edge Sampling ---
        num_edges_to_sample = min(current_node_idx, effective_m)
        adj_vec_next_sampled = edge_gen_fn(edge_model, h_node, num_edges_to_sample, effective_m, temperature, device)

        # --- Store and Prepare for Next Step ---
        adj_vec_np = adj_vec_next_sampled.squeeze(0).squeeze(0).cpu().numpy()
        list_adj_vecs_sampled.append(adj_vec_np)
        adj_vec_current = adj_vec_next_sampled

        # --- Termination Checks ---
        # 1. Check for actual edges added in this step
        has_real_edge = False
        if num_edges_to_sample > 0:  # Only check if we actually sampled something
            sampled_part = adj_vec_np[:num_edges_to_sample, :]
            has_real_edge = np.any(sampled_part[:, EDGE_TYPES["REGULAR"]] == 1.0) or \
                            np.any(sampled_part[:, EDGE_TYPES["INVERTED"]] == 1.0)

        if has_real_edge:
            no_real_edge_steps_in_a_row = 0  # Reset patience counter
        else:
            no_real_edge_steps_in_a_row += 1  # Increment patience counter

        # 2. Check explicit EOS signal (all NONE in sampled region)
        is_eos = False
        if num_edges_to_sample > 0:
            is_eos = np.all(adj_vec_np[:num_edges_to_sample, EDGE_TYPES["NONE"]] == 1.0)
        elif num_edges_to_sample == 0:  # First step (i=1), num_edges_to_sample=0
             is_eos = False  # Cannot be EOS on first step

        if is_eos:
            print(f"INFO: EOS signal detected at step {i}. Stopping generation.")
            list_adj_vecs_sampled.pop()  # Remove the EOS vector
            break

        # 3. Check Patience
        if no_real_edge_steps_in_a_row >= eos_patience:
            print(f"INFO: Patience ({eos_patience}) exceeded. No real edges added for {no_real_edge_steps_in_a_row} steps. Stopping at step {i}.")
            # Do NOT pop the last vector here, it wasn't a clean EOS
            break

        # 4. Check target node count
        # Stop *after* adding the target node index
        if current_node_idx >= num_nodes_target - 1:
            # Let loop continue if target is reached exactly, break on next iteration if needed
            pass  # Allow loop to potentially finish via EOS/Patience/MaxSteps

        # 5. Max steps reached (handled by loop limit)
        if i == max_steps - 1:
             print(f"INFO: Reached max_steps ({max_steps}). Stopping generation.")

    # --- Final Graph Construction ---
    final_graph = aig_seq_to_nx(list_adj_vecs_sampled, NUM_EDGE_FEATURES)
    final_levels, final_max_level = {}, -1
    if final_graph.number_of_nodes() > 0:
         try:
             if nx.is_directed_acyclic_graph(final_graph):
                  final_levels, final_max_level = _calculate_levels(final_graph)
                  final_graph.graph['levels'] = final_levels
             else:
                  print("Warning: Generated graph is not a DAG. Max level calculation might be inaccurate.")
                  final_levels = {n: 0 for n in final_graph.nodes()}
                  final_max_level = -2  # Use negative code for non-DAG
         except Exception as e:
              print(f"Warning: Error calculating final levels: {e}")
              final_max_level = -3  # Use negative code for error

    # Print final node count relative to target
    final_node_count = final_graph.number_of_nodes()
    print(f"Generated graph with {final_node_count} nodes (target: {num_nodes_target}) and max level {final_max_level}.")

    return final_graph, final_max_level


def load_model_and_config(model_path, device):
    """Loads model state dict and config from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    if 'config' not in state:
        raise ValueError(f"Checkpoint {model_path} does not contain 'config'.")
    config = state['config']
    return state, config