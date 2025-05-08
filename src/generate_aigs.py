'''Code to use trained GraphRNN to generate a new graph, handling node prediction.'''

import argparse
import numpy as np
import torch
import networkx as nx
import logging
import sys
import os
from typing import Dict, Any, List, Tuple, Optional, Callable

# --- Import from project files ---
try:
    from model import GraphLevelRNN, EdgeLevelRNN
    MODEL_CLASSES_LOADED = True
except ImportError as e:
    print(f"Error importing models from model.py: {e}. Ensure GraphLevelRNN and EdgeLevelRNN are defined.")
    sys.exit(1)

# --- Logger Setup ---
logger = logging.getLogger("generate_aigs")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Configuration Import ---
# Assumes aig_config.py is in the same directory or accessible in PYTHONPATH
import aig_config as aig_config

# --- Constants ---
EDGE_TYPES_INTERNAL = {"NONE": 0, "REGULAR": 1, "INVERTED": 2}
# NUM_EDGE_FEATURES should be 3 because the model predicts one of three classes: None, Regular, Inverted.
NUM_EDGE_FEATURES = 3

INT_TO_EDGE_TYPE_STR = {
    EDGE_TYPES_INTERNAL["REGULAR"]: aig_config.EDGE_TYPE_KEYS[0],
    EDGE_TYPES_INTERNAL["INVERTED"]: aig_config.EDGE_TYPE_KEYS[1],
}
INT_TO_NODE_TYPE_STR = {i: key for i, key in enumerate(aig_config.NODE_TYPE_KEYS)}
NODE_STR_TO_INT = {name: i for i, name in INT_TO_NODE_TYPE_STR.items()} # For robust default type assignment
NODE_UNKNOWN_STR = "UNKNOWN_NODE"

if len(INT_TO_NODE_TYPE_STR) != aig_config.NUM_NODE_FEATURES:
    logger.warning(
        f"Mismatch: INT_TO_NODE_TYPE_STR len ({len(INT_TO_NODE_TYPE_STR)}) "
        f"!= aig_config.NUM_NODE_FEATURES ({aig_config.NUM_NODE_FEATURES})"
    )

# --- Sampling Functions (Assumed correct) ---
def sample_bernoulli(p):
    p_val = p.item() if isinstance(p, torch.Tensor) else p
    return int(np.random.random() < p_val)

def sample_temperature(logits: torch.Tensor, temperature: float, top_k: int = 0, top_p: float = 0.0) -> int:
    if logits.numel() == 0: return EDGE_TYPES_INTERNAL["NONE"]
    if temperature <= 1e-6: return torch.argmax(logits).item()
    logits = logits / temperature
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float('Inf'))
    if top_k > 0:
        top_k_val = min(top_k, logits.size(-1))
        if top_k_val > 0:
            indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, -float('Inf'))
    probabilities = torch.softmax(logits, dim=-1)
    try:
        if not torch.all(probabilities >= 0): probabilities = torch.clamp(probabilities, min=0)
        prob_sum = torch.sum(probabilities)
        if prob_sum < 1e-6:
            valid_logits_mask = logits > -float('Inf')
            if torch.any(valid_logits_mask): probabilities = torch.ones_like(logits) * valid_logits_mask.float(); probabilities = probabilities / torch.sum(probabilities)
            else: probabilities = torch.ones_like(logits) / logits.numel()
        return torch.multinomial(probabilities, 1).item()
    except RuntimeError: return torch.argmax(logits).item()

# --- Edge Generation Helper (Assumed correct) ---
def rnn_edge_gen(
        edge_rnn: EdgeLevelRNN, h_context: torch.Tensor, num_edges_to_generate: int,
        adj_vec_padding_size: int, edge_feature_len_arg: int, sample_fun_arg: Callable,
        mode_arg: str, temperature_arg: float, top_k_arg: int, top_p_arg: float,
        attempts_arg: Optional[int]) -> torch.Tensor:
    device = h_context.device
    adj_indices_vec = torch.full((adj_vec_padding_size,), EDGE_TYPES_INTERNAL["NONE"], dtype=torch.long, device=device)
    x_edge_input = torch.zeros([1, 1, edge_feature_len_arg], device=device)
    x_edge_input[0, 0, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
    if hasattr(edge_rnn, 'hidden') and edge_rnn.hidden is None:
        logger.error("rnn_edge_gen: edge_rnn.hidden is None."); return adj_indices_vec.unsqueeze(0).unsqueeze(0)
    for i in range(num_edges_to_generate):
        try:
            edge_logits = edge_rnn(x_edge_input, x_lens=torch.tensor([1], device='cpu'), return_logits=True)
            sampled_class_index = sample_fun_arg(edge_logits.squeeze(), temperature_arg, top_k_arg, top_p_arg)
            adj_indices_vec[i] = sampled_class_index
            next_x_one_hot = torch.zeros_like(x_edge_input)
            if 0 <= sampled_class_index < edge_feature_len_arg: next_x_one_hot[0, 0, sampled_class_index] = 1.0
            else: logger.warning(f"RNN Gen: Invalid edge index {sampled_class_index}. Using NONE."); next_x_one_hot[0, 0, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
            x_edge_input = next_x_one_hot
        except Exception as e: logger.error(f"RNN edge gen step {i} error: {e}", exc_info=True); return adj_indices_vec.unsqueeze(0).unsqueeze(0)
    return adj_indices_vec.unsqueeze(0).unsqueeze(0)

# --- Graph Building (Assumed correct) ---
def build_graph_from_indices(
        list_adj_indices: List[np.ndarray], list_predicted_node_types: List[int],
        m: int, final_num_nodes: int) -> nx.DiGraph:
    G = nx.DiGraph()
    if final_num_nodes <= 0: return G
    if len(list_predicted_node_types) != final_num_nodes:
        logger.error(f"Node type list len ({len(list_predicted_node_types)}) != final nodes ({final_num_nodes})."); return G
    for node_idx in range(final_num_nodes):
        predicted_type_int = list_predicted_node_types[node_idx]
        node_type_str = INT_TO_NODE_TYPE_STR.get(predicted_type_int, NODE_UNKNOWN_STR)
        if node_type_str == NODE_UNKNOWN_STR: logger.warning(f"Node {node_idx}: Invalid type index {predicted_type_int}.")
        G.add_node(node_idx, type=node_type_str)
    for j, class_indices_for_target_node in enumerate(list_adj_indices):
        target_node_idx = j + 1
        if target_node_idx not in G: continue
        num_potential_sources = min(target_node_idx, m)
        for k_rev in range(num_potential_sources):
            source_node_idx = (target_node_idx - 1) - k_rev
            if source_node_idx < 0 or source_node_idx not in G: continue
            edge_class_int = class_indices_for_target_node[k_rev]
            edge_type_str = INT_TO_EDGE_TYPE_STR.get(edge_class_int)
            if edge_type_str: G.add_edge(source_node_idx, target_node_idx, type=edge_type_str)
    return G

# --- Main Generation Function ---
def generate(
        node_model: GraphLevelRNN,
        edge_model: EdgeLevelRNN,
        input_size: int,  # m, max predecessors
        edge_gen_function: Callable,
        mode: str,
        edge_feature_len: int, # Should be NUM_EDGE_FEATURES (3)
        predict_node_types: bool,
        num_node_types: Optional[int], # e.g., 4 for AIG
        max_nodes: int,
        min_nodes: int,
        patience: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        edge_sample_attempts: int = 1
) -> Optional[nx.DiGraph]:

    # --- Validation and Setup ---
    if not isinstance(node_model, GraphLevelRNN): logger.warning("Node_model type mismatch")
    if not isinstance(edge_model, EdgeLevelRNN): logger.warning("Edge_model type mismatch")
    if predict_node_types and num_node_types is None:
        logger.error("Node prediction enabled but num_node_types not provided."); return None
    if edge_feature_len != NUM_EDGE_FEATURES:
        logger.warning(f"edge_feature_len mismatch: Expected {NUM_EDGE_FEATURES}, got {edge_feature_len}.")

    if hasattr(node_model, 'eval'): node_model.eval()
    if hasattr(edge_model, 'eval'): edge_model.eval()
    device = next(node_model.parameters()).device
    edge_model.to(device)

    sample_fun = sample_temperature
    min_nodes = max(1, min_nodes)
    if max_nodes < min_nodes: max_nodes = min_nodes
    patience = max(1, patience)

    # --- Generation Loop ---
    # `adj_vec_for_node_model_input` stores the connections *into* the previous node (node i-1)
    adj_vec_for_node_model_input = torch.zeros([1, 1, input_size, NUM_EDGE_FEATURES], device=device)
    adj_vec_for_node_model_input[:, :, :, EDGE_TYPES_INTERNAL["NONE"]] = 1.0 # SOS for first step

    list_adj_indices_for_all_nodes = []  # Stores edge indices for edges *into* node i+1
    list_predicted_node_types = []       # Stores type index for node i

    if hasattr(node_model, 'reset_hidden'): node_model.reset_hidden()
    if hasattr(edge_model, 'reset_hidden'): edge_model.reset_hidden()

    generated_nodes_count = 0 # Tracks number of nodes whose types and incoming edges are finalized
    no_edge_streak = 0

    # --- Level Calculation Setup ---
    current_node_levels = {} # Stores level of each generated node_idx
    model_max_allowable_level = node_model.max_level if hasattr(node_model, 'max_level') and node_model.max_level is not None else float('inf')
    uses_level_embedding = hasattr(node_model, 'level_embedding') and node_model.level_embedding is not None

    logger.info(
        f"Starting generation: max_nodes={max_nodes}, min_nodes={min_nodes}, patience={patience}, "
        f"temp={temperature}, top_k={top_k}, top_p={top_p}, predict_nodes={predict_node_types}, use_levels={uses_level_embedding}")

    with torch.no_grad():
        # --- Loop for generating nodes 0 to max_nodes-1 ---
        # `current_node_idx` is the index of the node we are currently generating (0, 1, 2, ...)
        for current_node_idx in range(max_nodes):

            # --- Determine Level for the current_node_idx ---
            # Calculate the level of the current node *before* calling NodeModel,
            # based on the levels of its potential predecessors (nodes 0 to current_node_idx - 1).
            # This requires knowing which predecessors *will* connect, which we only know *after* edge generation.
            # This seems like the core difficulty.

            # *** REVISED LEVEL LOGIC V3: Align with older script `aig_generate.py` ***
            # Calculate the level of the current node based on the levels of nodes 0..i-1
            # *before* calling the node model for node i.
            # This assumes the level embedding should represent the structural level of the node being generated.
            current_node_calculated_level = 0
            if current_node_idx > 0:
                # Find max level among predecessors of current_node_idx
                # Predecessors are determined by the *last* generated edge vector: list_adj_indices_for_all_nodes[-1]
                max_predecessor_level = -1
                if list_adj_indices_for_all_nodes: # Check if edge vectors exist
                    last_edge_indices_np = list_adj_indices_for_all_nodes[-1]
                    num_potential_preds = len(last_edge_indices_np)
                    for k_rev in range(num_potential_preds):
                        if last_edge_indices_np[k_rev] != EDGE_TYPES_INTERNAL["NONE"]: # If edge exists
                            # Source node for the edge into node current_node_idx-1
                            source_node_for_prev = (current_node_idx - 1 - 1) - k_rev
                            # We need the level of the node that *will* connect to current_node_idx
                            # This still seems problematic.

                            # Let's try the logic from the older `aig_generate.py` directly:
                            # Calculate level based on predecessors 0..i-1, assuming connectivity based on previous steps.
                            # This might be an approximation if the model needs exact current connectivity.

                            # Re-calculating level based on nodes 0..i-1 that *could* connect
                            max_pred_level_for_current = -1
                            num_preds_possible_for_current = min(current_node_idx, input_size)
                            for k in range(num_preds_possible_for_current):
                                source_node = current_node_idx - 1 - k # Potential source node index
                                if source_node in current_node_levels:
                                     max_pred_level_for_current = max(max_pred_level_for_current, current_node_levels[source_node])
                            current_node_calculated_level = max_pred_level_for_current + 1
                else: # First node (node 1) after node 0
                     current_node_calculated_level = current_node_levels.get(0, 0) + 1 # Depends only on node 0

            else: # current_node_idx is 0
                current_node_calculated_level = 0

            current_node_levels[current_node_idx] = current_node_calculated_level # Store calculated level for this node
            clamped_level_for_node_model = min(current_node_calculated_level, model_max_allowable_level)
            level_input_tensor = torch.tensor([[clamped_level_for_node_model]], dtype=torch.long, device=device) if uses_level_embedding else None

            logger.debug(f"Generating node {current_node_idx} (NodeModel input level: {clamped_level_for_node_model if uses_level_embedding else 'N/A'})...")

            # --- Node Model: Get context (h) and predict type for current_node_idx ---
            # Input `adj_vec_for_node_model_input` represents edges into node `current_node_idx - 1`
            node_model_output = node_model(adj_vec_for_node_model_input, levels=level_input_tensor)

            predicted_type_for_current_node = 0
            if predict_node_types:
                if not isinstance(node_model_output, tuple) or len(node_model_output) != 2:
                    logger.error(f"Node model (for node {current_node_idx}) didn't return (h, logits)."); return None
                h_context_for_current_node, logits_for_current_node_type = node_model_output
                predicted_type_for_current_node = sample_fun(logits_for_current_node_type.squeeze(), temperature, top_k, top_p)
            else:
                h_context_for_current_node = node_model_output
                predicted_type_for_current_node = NODE_STR_TO_INT.get(aig_config.NODE_TYPE_KEYS[0 if current_node_idx == 0 else 2], 0) # Default CONST0 for node 0, AND for others

            list_predicted_node_types.append(predicted_type_for_current_node)
            # --- End Node Model processing for current node ---

            # --- Edge Prediction: Edges *into* current_node_idx ---
            num_edges_to_predict = min(current_node_idx, input_size)
            edge_indices_for_current_node_np = np.array([], dtype=int)

            if num_edges_to_predict > 0:
                if hasattr(edge_model, 'set_first_layer_hidden'):
                    try:
                        edge_model.set_first_layer_hidden(h_context_for_current_node)
                    except Exception as e:
                        logger.error(f"Error setting edge hidden state for node {current_node_idx}: {e}"); return None

                adj_indices_tensor = edge_gen_function(
                    edge_model, h_context_for_current_node, num_edges_to_predict, input_size,
                    edge_feature_len, sample_fun, mode, temperature, top_k, top_p, edge_sample_attempts
                )
                edge_indices_for_current_node_np = adj_indices_tensor[0, 0, :num_edges_to_predict].cpu().numpy()

            # Store edge indices. Note: Node 0 has no incoming edges from previous nodes.
            if current_node_idx > 0:
                list_adj_indices_for_all_nodes.append(edge_indices_for_current_node_np)

            generated_nodes_count += 1 # We have now finalized node `current_node_idx`
            # --- End Edge Prediction ---

            # --- Patience Stopping Check ---
            if generated_nodes_count >= min_nodes:
                # Check edges generated for the *current* node (node `current_node_idx`)
                all_none_edges = np.all(edge_indices_for_current_node_np == EDGE_TYPES_INTERNAL["NONE"]) if edge_indices_for_current_node_np.size > 0 else True

                # Increment streak only if the current node (which is > 0) received no edges
                if current_node_idx > 0 and all_none_edges:
                    no_edge_streak += 1
                elif current_node_idx > 0 and not all_none_edges : # Reset if any edge is made to a node > 0
                    no_edge_streak = 0

                logger.debug(f"  Node {current_node_idx}: Edges {edge_indices_for_current_node_np.tolist() if edge_indices_for_current_node_np.size > 0 else '[]'}. Streak: {no_edge_streak}/{patience}")

                if no_edge_streak >= patience and current_node_idx > 0 : # Don't stop due to node 0 having no incoming edges
                    logger.info(f"Stopping early at node {current_node_idx}: reached patience={patience}.")
                    break
            # --- End Patience Check ---

            # --- Prepare `adj_vec_for_node_model_input` for the *next* iteration (for node current_node_idx + 1) ---
            # This vector represents connections *into* `current_node_idx`
            adj_vec_for_node_model_input.zero_()
            adj_vec_for_node_model_input[:, :, :, EDGE_TYPES_INTERNAL["NONE"]] = 1.0

            num_actual_indices = len(edge_indices_for_current_node_np)
            slice_len_for_next_input = min(num_actual_indices, input_size)
            for k_one_hot in range(slice_len_for_next_input):
                class_idx = edge_indices_for_current_node_np[k_one_hot]
                if 0 <= class_idx < NUM_EDGE_FEATURES:
                    adj_vec_for_node_model_input[0, 0, k_one_hot, EDGE_TYPES_INTERNAL["NONE"]] = 0.0
                    adj_vec_for_node_model_input[0, 0, k_one_hot, class_idx] = 1.0
            # --- End Input Prep ---

            # No need for explicit check `generated_nodes_count >= max_nodes` here,
            # the loop condition `range(max_nodes)` handles it.

    # --- Post-processing After Loop ---
    if generated_nodes_count == 0:
        logger.warning("Generation resulted in 0 nodes."); return None
    # Check if enough nodes were generated *before* patience stop
    if generated_nodes_count < min_nodes:
        logger.warning(f"Generated nodes ({generated_nodes_count}) < min_nodes ({min_nodes}). Returning None.")
        return None

    logger.info(f"Building NetworkX graph for {generated_nodes_count} generated nodes...")
    # list_adj_indices_for_all_nodes has edge info for nodes 1 to N-1
    # list_predicted_node_types has type info for nodes 0 to N-1
    final_graph = build_graph_from_indices(
        list_adj_indices_for_all_nodes,
        list_predicted_node_types,
        input_size, # m
        generated_nodes_count # n
    )
    logger.debug(f"Graph building successful. Nodes: {final_graph.number_of_nodes()}, Edges: {final_graph.number_of_edges()}")
    return final_graph


if __name__ == '__main__':
    print("This script defines generation functions. Run get_aigs.py or a similar control script to execute.")

