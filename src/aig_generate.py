"""
Generation module for AIG (And-Inverter Graphs) using trained models.
"""

import os
import torch
import numpy as np
import networkx as nx
from torch.distributions import Categorical

# Try to import necessary constants
try:
    from aig_dataset import _calculate_levels, EDGE_TYPES, NUM_EDGE_FEATURES
except ImportError:
    print("Warning: Failed to import from aig_dataset. Using default constants.")
    # Default values as fallback
    EDGE_TYPES = {"NONE": 0, "REGULAR": 1, "INVERTED": 2}
    NUM_EDGE_FEATURES = 3

    def _calculate_levels(g):
        """Fallback implementation for calculate_levels if import fails."""
        if not g or g.number_of_nodes() == 0:
            return {}, -1

        levels = {}
        for node in nx.topological_sort(g):
            pred_levels = [levels.get(pred, 0) for pred in g.predecessors(node)]
            levels[node] = max(pred_levels + [-1]) + 1

        max_level = max(levels.values()) if levels else -1
        return levels, max_level


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

    Fixed version with better error handling and fallbacks.

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
    # Special case: If at the first step (num_edges_to_sample=0),
    # just return a tensor with NONE edges
    if num_edges_to_sample <= 0:
        adj_vec_sampled = torch.zeros([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
        adj_vec_sampled[:, :, :, EDGE_TYPES["NONE"]] = 1.0
        return adj_vec_sampled

    # Initialize output tensor (default to NONE edges)
    adj_vec_sampled = torch.zeros([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
    adj_vec_sampled[:, :, :, EDGE_TYPES["NONE"]] = 1.0

    # Handle setting initial hidden state based on model type
    hidden_state_set = False
    if hasattr(edge_rnn, 'set_first_layer_hidden') and callable(getattr(edge_rnn, 'set_first_layer_hidden')):
        try:
            if h.dim() == 3:  # shape [1, 1, hidden_size]
                edge_rnn.set_first_layer_hidden(h.squeeze(0))  # Try squeezing one dim
            else:  # shape [1, hidden_size] or [hidden_size]
                edge_rnn.set_first_layer_hidden(h)
            hidden_state_set = True
        except Exception as e:
            print(f"Warning: Could not set hidden state for EdgeRNN/LSTM: {e}")

    if not hidden_state_set:
        print("Warning: EdgeRNN hidden state not set, generation might fail")

    try:
        # SOS token (one-hot encoded for NONE type)
        # Shape: [batch=1, seq=1, features=3]
        x = torch.zeros([1, 1, NUM_EDGE_FEATURES], device=device)
        x[:, :, EDGE_TYPES["NONE"]] = 1.0  # Start with SOS = NONE

        sampled_edges = []
        force_valid_edge = True  # Force at least one valid edge to avoid all-NONE

        for i in range(num_edges_to_sample):
            try:
                # Get logits for the next edge type
                logits = edge_rnn(x, return_logits=True)

                # Check for NaN or Inf values
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"Warning: NaN/Inf detected in logits at step {i}")
                    # Set to a reasonable fallback
                    logits = torch.zeros([1, 1, NUM_EDGE_FEATURES], device=device)
                    logits[0, 0, EDGE_TYPES["REGULAR"]] = 1.0

                # Apply temperature and sample
                scaled_logits = logits.squeeze(1) / temperature  # Shape: [1, 3]
                dist = Categorical(logits=scaled_logits)
                sampled_type_idx = dist.sample()  # Tensor containing index [0, 1, or 2]

                # Force a valid edge (REGULAR or INVERTED) for at least the first edge
                if force_valid_edge and i == 0 and sampled_type_idx.item() == EDGE_TYPES["NONE"]:
                    sampled_type_idx = torch.tensor([EDGE_TYPES["REGULAR"]], device=device)
                    force_valid_edge = False

                # Create next input (one-hot encoding of sampled type)
                x.zero_()
                x[0, 0, sampled_type_idx.item()] = 1.0
                sampled_edges.append(x.clone().squeeze().cpu().numpy())  # Store sampled one-hot vector

            except Exception as e:
                print(f"Error in edge sampling at step {i}: {e}")
                # Provide a fallback value for this step
                fallback = np.zeros(NUM_EDGE_FEATURES)
                if i == 0:  # Force a valid edge for position 0 if error
                    fallback[EDGE_TYPES["REGULAR"]] = 1.0
                else:
                    fallback[EDGE_TYPES["NONE"]] = 1.0
                sampled_edges.append(fallback)

        # Place sampled edges into the correct (reversed) positions in adj_vec_sampled
        if sampled_edges:
            try:
                sampled_stack = torch.tensor(np.stack(sampled_edges[::-1]), device=device)  # Reverse before stacking
                num_sampled = sampled_stack.shape[0]
                adj_vec_sampled[0, 0, :num_sampled, :] = sampled_stack  # Fill from the start (represents nearest predecessors)
            except Exception as e:
                print(f"Error stacking sampled edges: {e}")
                # Ensure first edge is REGULAR on error
                if num_edges_to_sample > 0:
                    adj_vec_sampled[0, 0, 0, EDGE_TYPES["NONE"]] = 0.0
                    adj_vec_sampled[0, 0, 0, EDGE_TYPES["REGULAR"]] = 1.0

    except Exception as e:
        print(f"Error in rnn_edge_gen_aig: {e}")
        # On error, ensure we have at least one REGULAR edge in the output
        if num_edges_to_sample > 0:
            adj_vec_sampled[0, 0, 0, EDGE_TYPES["NONE"]] = 0.0
            adj_vec_sampled[0, 0, 0, EDGE_TYPES["REGULAR"]] = 1.0

    return adj_vec_sampled


def mlp_edge_gen_aig(edge_mlp, h, num_edges_to_sample, effective_m, temperature=1.0, device='cpu'):
    """
    Generates AIG edges for one node using MLP method with categorical sampling.

    Fixed version that handles edge cases better.

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
    # Special case: If at the first step (num_edges_to_sample=0),
    # just return a tensor with NONE edges
    if num_edges_to_sample <= 0:
        adj_vec_sampled = torch.zeros([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
        adj_vec_sampled[:, :, :, EDGE_TYPES["NONE"]] = 1.0
        return adj_vec_sampled

    # Initialize output tensor (default to NONE edges)
    adj_vec_sampled = torch.zeros([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
    adj_vec_sampled[:, :, :, EDGE_TYPES["NONE"]] = 1.0

    try:
        # Get logits for all potential edges at once
        # edge_mlp output shape: [batch=1, seq=1, output_m, features=3]
        all_logits = edge_mlp(h, return_logits=True)

        # Debug info
        if all_logits is None:
            print("ERROR: edge_mlp returned None")
            return adj_vec_sampled

        # Squeeze batch and sequence dims: [output_m, features=3]
        logits_for_sampling = all_logits.squeeze(0).squeeze(0)

        # Ensure we have enough logits for sampling
        if logits_for_sampling.shape[0] < num_edges_to_sample:
            print(f"WARNING: Not enough logits for sampling: {logits_for_sampling.shape[0]} < {num_edges_to_sample}")
            # Use what we have
            num_edges_to_sample = min(num_edges_to_sample, logits_for_sampling.shape[0])

        # Select logits for the edges we need to sample
        relevant_logits = logits_for_sampling[:num_edges_to_sample, :]

        # Check for NaN/Inf values in logits
        if torch.isnan(relevant_logits).any() or torch.isinf(relevant_logits).any():
            print(f"WARNING: NaN or Inf values detected in logits. Using default edges.")
            # Force some reasonable values in edge[0] to avoid all NONE
            if num_edges_to_sample > 0:
                # Make a manual probability distribution favoring REGULAR edges
                adj_vec_sampled[0, 0, 0, EDGE_TYPES["REGULAR"]] = 1.0
                adj_vec_sampled[0, 0, 0, EDGE_TYPES["NONE"]] = 0.0
            return adj_vec_sampled

        # Apply temperature and try to sample
        try:
            # Apply temperature scaling
            scaled_logits = relevant_logits / temperature

            # Initialize distribution for sampling
            dist = Categorical(logits=scaled_logits)

            # Sample edge types - catch any other potential errors
            sampled_type_indices = dist.sample()  # Shape: [num_edges_to_sample]

            # Create one-hot encodings
            one_hot_sampled = torch.nn.functional.one_hot(
                sampled_type_indices,
                num_classes=NUM_EDGE_FEATURES
            ).float()

            # Place into the adjacency vector (correct positions)
            adj_vec_sampled[0, 0, :num_edges_to_sample, :] = one_hot_sampled

            # Ensure we got at least one actual edge (not all NONE)
            # This is especially important for step 1
            if num_edges_to_sample > 0 and torch.all(one_hot_sampled[:, EDGE_TYPES["NONE"]] == 1.0):
                # Force at least one edge in the first position to be REGULAR
                # This helps avoid immediate EOS signals
                adj_vec_sampled[0, 0, 0, EDGE_TYPES["NONE"]] = 0.0
                adj_vec_sampled[0, 0, 0, EDGE_TYPES["REGULAR"]] = 1.0
                print("Adding forced REGULAR edge to avoid all-NONE output")

        except Exception as e:
            print(f"Error during sampling: {e}. Using default edges.")
            # Force a reasonable result
            if num_edges_to_sample > 0:
                adj_vec_sampled[0, 0, 0, EDGE_TYPES["REGULAR"]] = 1.0
                adj_vec_sampled[0, 0, 0, EDGE_TYPES["NONE"]] = 0.0

    except Exception as e:
        print(f"Error in mlp_edge_gen_aig: {e}")
        # Return default (already initialized above)

    return adj_vec_sampled


def generate_aig(num_nodes_target, node_model, edge_model, effective_m, max_level_model,
                 edge_gen_fn, device, temperature=1.0, max_steps=None, eos_patience=10, debug=False):
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
        eos_patience: Stop if no real edges (type 1 or 2) are added for this many consecutive steps.
        debug: Enable additional debug output.

    Returns:
        nx.DiGraph: The generated AIG as a NetworkX graph.
        int: The maximum level reached in the generated graph.
    """
    node_model.eval()
    edge_model.eval()

    # --- Initialization ---
    # Important: Initialize with SOS token structure (ones)
    adj_vec_current = torch.ones([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
    # Clear and set the NONE channel to 1.0 for initial SOS token
    adj_vec_current.zero_()
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

    if debug:
        print(f"DEBUG: Starting generation with effective_m={effective_m}, target_nodes={num_nodes_target}")
        print(f"DEBUG: Node model type: {type(node_model).__name__}, Edge model type: {type(edge_model).__name__}")
        print(f"DEBUG: Uses level embedding: {uses_level_embedding}, Max level model: {max_level_model}")
        print(f"DEBUG: Temperature: {temperature}, EOS patience: {eos_patience}")

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
        try:
            node_output = node_model(adj_vec_current, levels=level_for_input)
            h_node = node_output[0] if isinstance(node_output, tuple) else node_output

            if debug and i == 1:
                print(f"DEBUG: First node hidden state shape: {h_node.shape}")
                print(f"DEBUG: First node hidden state mean: {h_node.mean().item():.6f}, var: {h_node.var().item():.6f}")
                print(f"DEBUG: First node hidden state contains NaN: {torch.isnan(h_node).any().item()}")

        except Exception as e:
            print(f"ERROR in node model forward pass: {e}")
            # Return empty graph on error
            return nx.DiGraph(), -1

        # --- Edge Sampling ---
        num_edges_to_sample = min(current_node_idx, effective_m)

        if debug and i == 1:
            print(f"DEBUG: Step {i}, sampling {num_edges_to_sample} edges")

        try:
            adj_vec_next_sampled = edge_gen_fn(edge_model, h_node, num_edges_to_sample, effective_m, temperature, device)

            if debug and i == 1:
                print(f"DEBUG: First adj_vec_next_sampled shape: {adj_vec_next_sampled.shape}")
                edge_dist = adj_vec_next_sampled[0, 0, :num_edges_to_sample].sum(dim=0).cpu().numpy() if num_edges_to_sample > 0 else "N/A"
                print(f"DEBUG: Edge type distribution in first step: {edge_dist}")

        except Exception as e:
            print(f"ERROR in edge generation: {e}")
            # Return current graph on error
            if not list_adj_vecs_sampled:
                return nx.DiGraph(), -1
            else:
                return aig_seq_to_nx(list_adj_vecs_sampled, NUM_EDGE_FEATURES), max_level_gen

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

        # SPECIAL DEBUG CHECK FOR STEP 1
        if is_eos and i == 1:
            print(f"WARNING: EOS detected at step 1! This indicates a potential issue.")
            # For step 1, let's create a simpler graph as fallback
            if debug:
                print(f"DEBUG: Trying to continue despite first-step EOS...")
                is_eos = False  # Override EOS for debugging

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