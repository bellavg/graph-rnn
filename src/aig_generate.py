"""
Generation module for AIG (And-Inverter Graphs) using trained models.
"""

# src/aig_generate.py
import os
import torch
import numpy as np
import networkx as nx
from torch.distributions import Categorical
import torch.nn.functional as F # <--- Added Import
import logging # <--- Added Import
import traceback
# --- Setup logger if not already configured ---
logger = logging.getLogger("aig_generator")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Try to import necessary constants
try:
    # Adjust relative path if needed, or ensure src is in PYTHONPATH
    from aig_dataset import _calculate_levels, EDGE_TYPES, NUM_EDGE_FEATURES
except ImportError:
    logger.warning("Failed to import from aig_dataset. Using default constants.")
    # Default values as fallback
    EDGE_TYPES = {"NONE": 0, "REGULAR": 1, "INVERTED": 2}
    NUM_EDGE_FEATURES = 3

    def _calculate_levels(g):
        """Fallback implementation for calculate_levels if import fails."""
        if not g or g.number_of_nodes() == 0: return {}, -1
        levels = {}
        try:
            for node in nx.topological_sort(g):
                pred_levels = [levels.get(pred, -1) for pred in g.predecessors(node)]
                levels[node] = max(pred_levels) + 1 if pred_levels else 0
            max_level = max(levels.values()) if levels else -1
        except nx.NetworkXUnfeasible: # Handle non-DAG case
             logger.warning("Graph is not a DAG during level calculation fallback.")
             return {n:0 for n in g.nodes()}, -2
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

    if not adj_seq_list: # Check again after node add
        return g

    effective_m = adj_seq_list[0].shape[0]  # Get m from the shape

    for target_node_idx, adj_vec in enumerate(adj_seq_list, start=1):
        # adj_vec shape: [m, edge_feature_len]
        # Represents connections *to* target_node_idx
        num_possible_preds = min(target_node_idx, effective_m)

        # The adj_vec is reversed relative to node indices 0..i-1
        # adj_vec[k] corresponds to connection from node target_node_idx - 1 - k
        for k in range(num_possible_preds):
            source_node_idx = target_node_idx - 1 - k
            if source_node_idx < 0: continue  # Should not happen with correct slicing

            # Check if adj_vec has the expected dimension
            if adj_vec.shape[1] != edge_feature_len:
                 logger.error(f"Mismatch: adj_vec features ({adj_vec.shape[1]}) != edge_feature_len ({edge_feature_len}) at target {target_node_idx}")
                 continue

            try:
                edge_type_probs = adj_vec[k, :]  # Probabilities/logits for this edge
                # Assuming the adj_vec contains the *sampled* one-hot representation
                edge_type = np.argmax(edge_type_probs).item()  # Get the sampled class index
            except IndexError:
                 logger.error(f"IndexError accessing adj_vec[{k}] for target {target_node_idx} (shape: {adj_vec.shape})")
                 continue

            if edge_type != EDGE_TYPES["NONE"]:
                g.add_edge(source_node_idx, target_node_idx, type=edge_type)

    # Basic node type inference based on topology (can be refined in validation)
    # Removed as it's better done in the evaluation script after cleaning
    # for n in g.nodes(): ...

    return g

def rnn_edge_gen_aig(edge_rnn, h, num_edges_to_sample, effective_m,
                     temperature=1.0, device='cpu', debug=False, current_node_idx_for_log=None):
    """
    Generates AIG edges for one node using RNN method with categorical sampling.
    Includes optional probability logging.

    Args:
        # ... (other args) ...
        debug: Enable debug logging.
        current_node_idx_for_log: The index of the node being generated (for logging context).

    Returns:
        torch.Tensor: Sampled adjacency vector [1, 1, effective_m, NUM_EDGE_FEATURES] (one-hot encoded).
    """
    if num_edges_to_sample <= 0:
        adj_vec_sampled = torch.zeros([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
        adj_vec_sampled[:, :, :, EDGE_TYPES["NONE"]] = 1.0
        return adj_vec_sampled

    adj_vec_sampled = torch.zeros([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
    adj_vec_sampled[:, :, :, EDGE_TYPES["NONE"]] = 1.0

    hidden_state_set = False
    if hasattr(edge_rnn, 'set_first_layer_hidden') and callable(getattr(edge_rnn, 'set_first_layer_hidden')):
        try:
            if h.dim() == 3: edge_rnn.set_first_layer_hidden(h.squeeze(0))
            else: edge_rnn.set_first_layer_hidden(h)
            hidden_state_set = True
        except Exception as e:
            logger.warning(f"Could not set hidden state for EdgeRNN/LSTM: {e}")

    if not hidden_state_set: logger.warning("EdgeRNN hidden state not set")

    # --- Probability Logging Setup ---
    # Log only for a few initial graphs if debug is enabled extensively elsewhere,
    # or control via a separate flag/counter if needed.
    log_probs_this_call = debug and (current_node_idx_for_log is not None and current_node_idx_for_log < 5) # Log first 5 nodes
    if log_probs_this_call:
        logger.debug(f"--- Logging Edge Probs (Node {current_node_idx_for_log}) ---")

    try:
        x = torch.zeros([1, 1, NUM_EDGE_FEATURES], device=device)
        x[:, :, EDGE_TYPES["NONE"]] = 1.0 # Start with SOS = NONE
        sampled_edges = []

        for k in range(num_edges_to_sample): # Loop over potential predecessors
            try:
                logits = edge_rnn(x, return_logits=True)

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger.warning(f"NaN/Inf detected in logits at edge step k={k}. Using default.")
                    logits = torch.zeros_like(logits)
                    logits[0, 0, EDGE_TYPES["REGULAR"]] = 1.0 # Default to REGULAR

                scaled_logits = logits.squeeze(1) / temperature

                # --- ADD PROBABILITY LOGGING ---
                if log_probs_this_call:
                    probs = F.softmax(scaled_logits, dim=-1).squeeze().cpu().numpy()
                    # Node index this edge connects FROM
                    source_node_log_idx = (current_node_idx_for_log - 1 - k) if current_node_idx_for_log is not None else "N/A"
                    logger.debug(f"  k={k} (from node {source_node_log_idx}): Probs=[N:{probs[0]:.3f}, R:{probs[1]:.3f}, I:{probs[2]:.3f}]")
                # --- END LOGGING ---

                dist = torch.distributions.Categorical(logits=scaled_logits)
                sampled_type_idx = dist.sample()

                # Optional: Force first edge to be non-NONE?
                # if k == 0 and sampled_type_idx.item() == EDGE_TYPES["NONE"] and num_edges_to_sample > 0:
                #     sampled_type_idx = torch.tensor([EDGE_TYPES["REGULAR"]], device=device)
                #     logger.debug(f"  k=0 forced to REGULAR")

                x.zero_() # Prepare next input x
                x[0, 0, sampled_type_idx.item()] = 1.0
                # Store sampled one-hot vector: shape [NUM_EDGE_FEATURES]
                sampled_edges.append(x.clone().squeeze(0).squeeze(0).cpu().numpy())

            except Exception as e_inner:
                logger.error(f"Error in edge sampling step k={k}: {e_inner}")
                fallback = np.zeros(NUM_EDGE_FEATURES)
                fallback[EDGE_TYPES["NONE"]] = 1.0 # Default to NONE on error
                sampled_edges.append(fallback)
                # Optionally break if one step fails: break

        # Place sampled edges into the correct (reversed) positions
        if sampled_edges:
            try:
                # Stack requires consistent shapes, squeeze() above ensures shape [3]
                # Reverse the list of sampled vectors before stacking
                sampled_stack = torch.tensor(np.stack(sampled_edges[::-1]), device=device, dtype=torch.float32)
                num_sampled = sampled_stack.shape[0]
                # Ensure sampled_stack is 2D: [num_sampled, NUM_EDGE_FEATURES]
                if sampled_stack.dim() == 1: # Handle case of single sampled edge
                    sampled_stack = sampled_stack.unsqueeze(0)

                if num_sampled > 0 and sampled_stack.shape[1] == NUM_EDGE_FEATURES:
                    adj_vec_sampled[0, 0, :num_sampled, :] = sampled_stack
                elif num_sampled > 0:
                    logger.error(f"Shape mismatch after stacking: {sampled_stack.shape}, expected [, {NUM_EDGE_FEATURES}]")

            except Exception as e_stack:
                logger.error(f"Error stacking/assigning sampled edges: {e_stack}")
                # Fallback: Ensure first edge is REGULAR on error if possible
                if num_edges_to_sample > 0:
                    adj_vec_sampled.zero_()
                    adj_vec_sampled[0, 0, 0, EDGE_TYPES["REGULAR"]] = 1.0

    except Exception as e_outer:
        logger.error(f"Error in rnn_edge_gen_aig outer loop: {e_outer}")
        # Fallback on major error
        if num_edges_to_sample > 0:
             adj_vec_sampled.zero_()
             adj_vec_sampled[0, 0, 0, EDGE_TYPES["REGULAR"]] = 1.0

    return adj_vec_sampled


def mlp_edge_gen_aig(edge_mlp, h, num_edges_to_sample, effective_m, temperature=1.0, device='cpu', debug=False, current_node_idx_for_log=None):
    """
    Generates AIG edges for one node using MLP method with categorical sampling.
    (Includes placeholder args for debug/logging consistency)
    """
    if num_edges_to_sample <= 0:
        adj_vec_sampled = torch.zeros([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
        adj_vec_sampled[:, :, :, EDGE_TYPES["NONE"]] = 1.0
        return adj_vec_sampled

    adj_vec_sampled = torch.zeros([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
    adj_vec_sampled[:, :, :, EDGE_TYPES["NONE"]] = 1.0

    try:
        all_logits = edge_mlp(h, return_logits=True) # Expects h shape [1, 1, node_hidden_size]
        if all_logits is None:
            logger.error("ERROR: edge_mlp returned None")
            return adj_vec_sampled

        logits_for_sampling = all_logits.squeeze(0).squeeze(0) # Shape: [output_m, features=3]

        if logits_for_sampling.shape[0] < num_edges_to_sample:
            logger.warning(f"Not enough logits ({logits_for_sampling.shape[0]}) for sampling {num_edges_to_sample}. Using available.")
            num_edges_to_sample = logits_for_sampling.shape[0]

        if num_edges_to_sample <= 0: # Check again after potential reduction
             return adj_vec_sampled

        relevant_logits = logits_for_sampling[:num_edges_to_sample, :]

        if torch.isnan(relevant_logits).any() or torch.isinf(relevant_logits).any():
            logger.warning(f"NaN or Inf values detected in MLP logits. Using default edges.")
            adj_vec_sampled[0, 0, 0, EDGE_TYPES["REGULAR"]] = 1.0
            adj_vec_sampled[0, 0, 0, EDGE_TYPES["NONE"]] = 0.0
            return adj_vec_sampled

        try:
            scaled_logits = relevant_logits / temperature
            dist = torch.distributions.Categorical(logits=scaled_logits)
            sampled_type_indices = dist.sample() # Shape: [num_edges_to_sample]

            one_hot_sampled = torch.nn.functional.one_hot(
                sampled_type_indices, num_classes=NUM_EDGE_FEATURES
            ).float()

            adj_vec_sampled[0, 0, :num_edges_to_sample, :] = one_hot_sampled

            # Ensure not all NONE if num_edges_to_sample > 0
            if num_edges_to_sample > 0 and torch.all(one_hot_sampled[:, EDGE_TYPES["NONE"]] == 1.0):
                adj_vec_sampled[0, 0, 0, EDGE_TYPES["NONE"]] = 0.0
                adj_vec_sampled[0, 0, 0, EDGE_TYPES["REGULAR"]] = 1.0
                logger.debug("MLP: Forced first edge to REGULAR to avoid all-NONE.")

        except Exception as e_sample:
            logger.error(f"Error during MLP sampling: {e_sample}. Using default edges.")
            if num_edges_to_sample > 0:
                adj_vec_sampled[0, 0, 0, EDGE_TYPES["REGULAR"]] = 1.0
                adj_vec_sampled[0, 0, 0, EDGE_TYPES["NONE"]] = 0.0

    except Exception as e_outer:
        logger.error(f"Error in mlp_edge_gen_aig: {e_outer}")
        # Return default (initialized to NONEs)

    return adj_vec_sampled



def generate_aig(num_nodes_target, node_model, edge_model, effective_m, max_level_model,
                 edge_gen_fn, device,
                 temperatures: list, # Accept list
                 max_steps=None, eos_patience=10, debug=False):
    """
    Generates a single And-Inverter Graph (AIG). Tries multiple temperatures.
    """
    node_model.eval()
    edge_model.eval()

    best_graph = None
    best_max_level = -1
    best_temp = -1.0

    if not temperatures: # Handle empty temperature list
        logger.error("Temperature list is empty. Cannot generate graph.")
        return nx.DiGraph(), -1

    logger.info(f"Starting generation for target N={num_nodes_target} with Temps={temperatures}")

    for temp_idx, temperature in enumerate(temperatures):
        logger.debug(f"--- Attempting Temp={temperature} ---")

        # Reset state for each temperature attempt
        adj_vec_current = torch.zeros([1, 1, effective_m, NUM_EDGE_FEATURES], device=device)
        adj_vec_current[:, :, :, EDGE_TYPES["NONE"]] = 1.0 # SOS Token

        list_adj_vecs_sampled = []
        node_model.reset_hidden()
        current_levels = {0: 0} # Level of implicit PI node 0
        max_level_gen = 0
        uses_level_embedding = hasattr(node_model, 'level_embedding') and node_model.level_embedding is not None
        no_real_edge_steps_in_a_row = 0

        # --- Determine max generation steps ---
        gen_limit = max_steps
        if gen_limit is None:
            gen_limit = max(num_nodes_target + eos_patience + 5, int(num_nodes_target * 1.5))
            logger.debug(f"Setting max_steps automatically to {gen_limit}")

        if num_nodes_target <= 1: return nx.DiGraph(), 0

        # --- Generation Loop ---
        try:
            for i in range(1, gen_limit):
                current_node_idx = i # Generating edges FOR node i (connecting FROM nodes 0 to i-1)

                # --- Node Level Hidden State ---
                level_for_input = None
                if uses_level_embedding:
                    # Calculate level based on predecessors added so far
                    max_pred_level = -1
                    num_preds_possible = min(current_node_idx, effective_m)
                    for k in range(num_preds_possible):
                        source_node_idx = current_node_idx - 1 - k
                        max_pred_level = max(max_pred_level, current_levels.get(source_node_idx, -1))

                    current_node_level = max_pred_level + 1
                    current_levels[current_node_idx] = current_node_level # Store level of node we are about to add
                    max_level_gen = max(max_level_gen, current_node_level)
                    level_clamped = min(current_node_level, max_level_model)
                    level_for_input = torch.tensor([[level_clamped]], dtype=torch.long, device=device)

                # --- Node model forward pass ---
                node_output = node_model(adj_vec_current, levels=level_for_input)
                h_node = node_output[0] if isinstance(node_output, tuple) else node_output

                # --- Edge Sampling ---
                num_edges_to_sample = min(current_node_idx, effective_m)
                adj_vec_next_sampled = edge_gen_fn(edge_model, h_node, num_edges_to_sample, effective_m,
                                                 temperature, device, debug, current_node_idx) # Pass current node idx for logging

                # --- Store and Check Termination ---
                adj_vec_np = adj_vec_next_sampled.squeeze().cpu().numpy()
                list_adj_vecs_sampled.append(adj_vec_np)
                adj_vec_current = adj_vec_next_sampled # Use sampled as input for next step

                # Termination checks
                has_real_edge = False
                if num_edges_to_sample > 0:
                    sampled_part = adj_vec_np[:num_edges_to_sample, :]
                    has_real_edge = np.any(sampled_part[:, EDGE_TYPES["REGULAR"]] == 1.0) or \
                                    np.any(sampled_part[:, EDGE_TYPES["INVERTED"]] == 1.0)

                if has_real_edge: no_real_edge_steps_in_a_row = 0
                else: no_real_edge_steps_in_a_row += 1

                is_eos = False
                if num_edges_to_sample > 0:
                     # Check if *all* sampled entries are NONE
                     is_eos = np.all(np.argmax(sampled_part, axis=-1) == EDGE_TYPES["NONE"])

                # Stop conditions
                if is_eos and i > 1: # Don't stop on EOS at step 1
                    logger.info(f"EOS signal detected at step {i} (temp={temperature}).")
                    list_adj_vecs_sampled.pop() # Remove the EOS vector
                    break
                if no_real_edge_steps_in_a_row >= eos_patience:
                    logger.info(f"Patience ({eos_patience}) exceeded at step {i} (temp={temperature}).")
                    break
                # Stop when we have generated *enough* nodes (i.e., added node num_nodes_target)
                # Loop generates node i, so stop after generating node num_nodes_target
                if current_node_idx >= num_nodes_target:
                    logger.info(f"Reached target node count ({num_nodes_target+1} nodes total including node 0) at step {i} (temp={temperature}).")
                    break
                if i == gen_limit - 1:
                    logger.info(f"Reached max_steps ({gen_limit}) (temp={temperature}).")
            # --- End Generation Loop for this temp ---

            # --- Construct and Evaluate Graph for this temperature ---
            temp_graph = aig_seq_to_nx(list_adj_vecs_sampled, NUM_EDGE_FEATURES)
            temp_max_level = -1
            if temp_graph.number_of_nodes() > 0:
                try:
                    if nx.is_directed_acyclic_graph(temp_graph):
                        _, temp_max_level = _calculate_levels(temp_graph)
                    else: temp_max_level = -2
                except Exception as e_level:
                    logger.error(f"Error calculating level for temp {temperature}: {e_level}")
                    temp_max_level = -3

            logger.debug(f"  Temp {temperature} finished: N={temp_graph.number_of_nodes()}, L={temp_max_level}")

            # --- Update Best Graph ---
            is_better = False
            if temp_graph is not None and temp_graph.number_of_nodes() > 0:
                 # Prioritize graphs closer to target size, then higher level
                 current_best_nodes = best_graph.number_of_nodes() if best_graph else 0
                 nodes_closer_to_target = abs(temp_graph.number_of_nodes() - num_nodes_target) < abs(current_best_nodes - num_nodes_target)
                 nodes_equal_distance = abs(temp_graph.number_of_nodes() - num_nodes_target) == abs(current_best_nodes - num_nodes_target)

                 if best_graph is None:
                      is_better = True
                 elif nodes_closer_to_target:
                      is_better = True
                 elif nodes_equal_distance and temp_max_level > best_max_level:
                      is_better = True
                 # Optional: If levels equal, maybe prefer higher node count?
                 elif nodes_equal_distance and temp_max_level == best_max_level and temp_graph.number_of_nodes() > current_best_nodes:
                      is_better = True


            if is_better:
                logger.debug(f"  Updating best graph (from Temp {temperature}, N={temp_graph.number_of_nodes()}, L={temp_max_level})")
                best_graph = temp_graph
                best_max_level = temp_max_level
                best_temp = temperature

        except Exception as e_gen:
            logger.error(f"Major error during generation loop (Temp={temperature}): {e_gen}")
            logger.error(traceback.format_exc())
            continue # Try next temperature

    # --- End Temperature Loop ---

    if best_graph is None:
        logger.warning("Failed to generate any graph.")
        return nx.DiGraph(), -1

    logger.info(f"Selected best graph: N={best_graph.number_of_nodes()}, L={best_max_level} (from Temp={best_temp})")
    return best_graph, best_max_level


def load_model_and_config(model_path, device):
    """Loads model state dict and config from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
    try:
        state = torch.load(model_path, map_location=device)
    except Exception as e:
        raise IOError(f"Failed to load torch checkpoint {model_path}: {e}")

    if 'config' not in state:
        raise ValueError(f"Checkpoint {model_path} does not contain 'config'.")
    config = state['config']

    # Basic validation of state dict keys needed later
    required_keys = ['node_model', 'edge_model']
    missing_keys = [key for key in required_keys if key not in state]
    if missing_keys:
         # Attempt to find alternative keys
         alt_keys_found = False
         alt_map = {'node_model': 'node_net', 'edge_model': 'edge_net'} # Add more if needed
         for req_key, alt_key in alt_map.items():
              if req_key in missing_keys and alt_key in state:
                   logger.warning(f"Using alternative key '{alt_key}' for missing '{req_key}' in state_dict.")
                   state[req_key] = state[alt_key]
                   missing_keys.remove(req_key)
                   alt_keys_found = True

         if missing_keys: # Check again after trying alternatives
              raise ValueError(f"Checkpoint {model_path} missing state_dict keys: {missing_keys}")

    return state, config