'''Code to use trained GraphRNN to generate a new graph, handling node prediction.'''

import argparse
import numpy as np
import torch
import networkx as nx
import logging  # Use logging
import sys
import os
import time
import json
import pickle
import inspect
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

# --- Configuration Import with Fallback ---
try:
    # Try importing from the project structure if available
    from . import aig_config as aig_config

    logger.info("Successfully imported aig_config.")
except ImportError:
    logger.warning("Could not import from .aig_config. Using default internal values.")


    # Define fallbacks directly if import fails
    class aig_config:  # Simple class to hold fallback values
        NODE_TYPE_KEYS = ["CONST0", "PI", "AND", "PO"]
        NUM_NODE_FEATURES = 4
        EDGE_TYPE_KEYS = ["regular", "inverted"]
        NUM_EDGE_FEATURES_AIG = 2  # Regular, Inverted (excluding None)

# --- Constants ---
EDGE_TYPES_INTERNAL = {"NONE": 0, "REGULAR": 1, "INVERTED": 2}
NUM_EDGE_FEATURES = 3  # Number of edge classes model predicts (None, Reg, Inv)

# Map internal indices (1, 2) to string keys from aig_config
INT_TO_EDGE_TYPE_STR = {
    EDGE_TYPES_INTERNAL["REGULAR"]: aig_config.EDGE_TYPE_KEYS[0],  # 'regular'
    EDGE_TYPES_INTERNAL["INVERTED"]: aig_config.EDGE_TYPE_KEYS[1],  # 'inverted'
}

# Map internal indices (0, 1, 2, 3) to string keys from aig_config
INT_TO_NODE_TYPE_STR = {
    i: node_key for i, node_key in enumerate(aig_config.NODE_TYPE_KEYS)
}
# Validate mapping length against config
if len(INT_TO_NODE_TYPE_STR) != aig_config.NUM_NODE_FEATURES:
    logger.warning(
        f"Mismatch between derived INT_TO_NODE_TYPE_STR ({len(INT_TO_NODE_TYPE_STR)}) "
        f"and aig_config.NUM_NODE_FEATURES ({aig_config.NUM_NODE_FEATURES})"
    )

NODE_UNKNOWN_STR = "UNKNOWN_NODE"


# --- Sampling Functions ---
def sample_bernoulli(p):
    """ Samples 0 or 1 based on probability p. """
    p_val = p.item() if isinstance(p, torch.Tensor) else p
    return int(np.random.random() < p_val)


# --- MODIFIED: sample_temperature returns INDEX ---
def sample_temperature(logits: torch.Tensor, temperature: float, top_k: int = 0, top_p: float = 0.0) -> int:
    """ Samples an index from logits using temperature, top-k, or top-p sampling. """
    if logits.numel() == 0:
        logger.error("Cannot sample from empty logits tensor.")
        return 0  # Default to index 0 (e.g., 'NONE' edge or 'CONST0' node)

    if temperature <= 1e-6:  # Use argmax for zero or near-zero temperature
        return torch.argmax(logits).item()

    logits = logits / temperature

    # Top-p filtering
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # Use scatter_ to map back to original indices
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices,
                                                                                sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float('Inf'))

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Ensure top_k is not larger than num_classes
        if top_k > 0:  # Proceed only if top_k is valid
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, -float('Inf'))

    probabilities = torch.softmax(logits, dim=-1)

    # Multinomial sampling
    try:
        # Ensure probabilities are valid for multinomial
        if not torch.all(probabilities >= 0):
            logger.warning("Negative probabilities encountered after filtering/softmax. Clamping.")
            probabilities = torch.clamp(probabilities, min=0)

        # Check if sum is too small
        prob_sum = torch.sum(probabilities)
        if prob_sum < 1e-6:
            logger.warning(f"Probabilities sum to {prob_sum.item()} after filtering. Using uniform distribution.")
            probabilities = torch.ones_like(logits) / logits.numel()

        sampled_index = torch.multinomial(probabilities, 1).item()
    except RuntimeError as e:
        logger.error(
            f"Runtime error during torch.multinomial: {e}. Logits: {logits.cpu().numpy()}, Probs: {probabilities.cpu().numpy()}. Falling back to argmax.")
        # Fallback to argmax of original scaled logits if sampling fails
        sampled_index = torch.argmax(logits).item()  # Use potentially filtered logits

    return sampled_index


# --- END MODIFIED ---


# --- Edge Generation Helper Function (RNN Only) ---
def rnn_edge_gen(
        edge_rnn: EdgeLevelRNN,
        h_context: torch.Tensor,  # Renamed from node_output_context for clarity
        num_edges_to_generate: int,
        adj_vec_padding_size: int,
        edge_feature_len: int,  # Should be NUM_EDGE_FEATURES (e.g., 3)
        sample_fun: Callable,  # Should be sample_temperature
        mode: str,  # Unused in this specific function, but kept for signature consistency
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        attempts: Optional[int] = None  # Unused placeholder
) -> torch.Tensor:  # Returns tensor of indices
    """ Generates edge indices (0, 1, or 2) using an EdgeLevelRNN model. """
    try:
        device = h_context.device
    except AttributeError:  # Handle case where h_context might not be a tensor initially? Unlikely here.
        device = next(edge_rnn.parameters()).device

    # Stores the sampled class index for each potential edge
    adj_indices_vec = torch.full((adj_vec_padding_size,), EDGE_TYPES_INTERNAL["NONE"], dtype=torch.long, device=device)

    # Initial SOS token (one-hot for class 0 - NoEdge)
    x_edge_input = torch.zeros([1, 1, edge_feature_len], device=device)
    x_edge_input[0, 0, EDGE_TYPES_INTERNAL["NONE"]] = 1.0

    # Set hidden state (assuming it's handled correctly before calling this)
    # No internal initialization here, rely on set_first_layer_hidden being called externally.
    if hasattr(edge_rnn, 'hidden') and edge_rnn.hidden is None:
        logger.error("rnn_edge_gen called but edge_rnn.hidden is None. External init failed?")
        # Return vector of NONEs
        return adj_indices_vec.unsqueeze(0).unsqueeze(0)  # Add batch/seq dims back

    # Generation loop
    for i in range(num_edges_to_generate):
        try:
            # Call EdgeLevelRNN forward pass
            edge_logits = edge_rnn(x_edge_input, x_lens=torch.tensor([1], device='cpu'), return_logits=True)
            # Output edge_logits shape: [1, 1, edge_feature_len]

            # Sample the edge type index using the provided sample_fun
            sampled_class_index = sample_fun(
                edge_logits.squeeze(),  # Remove batch/seq dims -> [edge_feature_len]
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )  # sample_fun now returns an index

            # Update the adjacency vector for this step
            adj_indices_vec[i] = sampled_class_index

            # Prepare input for the next edge prediction step (one-hot encoding of sampled index)
            next_x_one_hot = torch.zeros_like(x_edge_input)
            if 0 <= sampled_class_index < edge_feature_len:
                next_x_one_hot[0, 0, sampled_class_index] = 1.0
            else:
                logger.warning(f"RNN Gen: Invalid edge class index {sampled_class_index} sampled. Using NONE.")
                next_x_one_hot[0, 0, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
            x_edge_input = next_x_one_hot

        except Exception as e:
            logger.error(f"Error during edge generation step {i}: {e}", exc_info=True)
            # Return partially generated vector on error
            return adj_indices_vec.unsqueeze(0).unsqueeze(0)  # Add batch/seq dims back

    # Return the vector of sampled indices, adding batch/seq dims for consistency if needed elsewhere
    # Shape [adj_vec_padding_size] -> [1, 1, adj_vec_padding_size]
    return adj_indices_vec.unsqueeze(0).unsqueeze(0)


# --- Graph Building Function (Unchanged) ---
def build_graph_from_indices(
        list_adj_indices: List[np.ndarray],
        list_predicted_node_types: List[int],
        m: int,
        final_num_nodes: int
) -> nx.DiGraph:
    """ Builds a NetworkX DiGraph from generated edge index lists and predicted node type indices. """
    G = nx.DiGraph()
    if final_num_nodes <= 0: return G

    if len(list_predicted_node_types) != final_num_nodes:
        logger.error(f"Node type list length ({len(list_predicted_node_types)}) != final nodes ({final_num_nodes}).")
        return G

    logger.debug(f"Assigning predicted node types for {final_num_nodes} nodes...")
    for node_idx in range(final_num_nodes):
        predicted_type_idx = list_predicted_node_types[node_idx]
        node_type_str = INT_TO_NODE_TYPE_STR.get(predicted_type_idx, NODE_UNKNOWN_STR)
        if node_type_str == NODE_UNKNOWN_STR: logger.warning(
            f"Node {node_idx}: Invalid type index {predicted_type_idx}.")
        G.add_node(node_idx, type=node_type_str)  # Assign string type attribute

    logger.debug(f"Adding edges based on {len(list_adj_indices)} adjacency vectors...")
    for target_node_idx, class_indices in enumerate(list_adj_indices,
                                                    start=1):  # list_adj_indices is for nodes 1 to n-1
        if target_node_idx not in G: continue

        num_connections_possible = min(target_node_idx, m)
        actual_indices_len = len(class_indices)  # This is the vector for edges INTO target_node_idx
        len_to_process = min(actual_indices_len, num_connections_possible)

        # class_indices[k] represents edge from (target_node_idx - 1 - k) to target_node_idx
        for k in range(len_to_process):
            source_node_idx = (target_node_idx - 1) - k
            if source_node_idx < 0 or source_node_idx not in G: continue

            edge_class_int = class_indices[k]
            edge_type_str = INT_TO_EDGE_TYPE_STR.get(edge_class_int)

            if edge_type_str:  # Add edge only if not NONE (index 0)
                G.add_edge(source_node_idx, target_node_idx, type=edge_type_str)
                logger.debug(f"  Added edge: {source_node_idx} -> {target_node_idx} (type: {edge_type_str})")

    return G


# --- Main Generation Function ---
def generate(
        node_model: GraphLevelRNN,
        edge_model: EdgeLevelRNN,
        input_size: int,  # m
        edge_gen_function: Callable,  # Should be rnn_edge_gen
        mode: str,
        edge_feature_len: int,  # Should be NUM_EDGE_FEATURES
        predict_node_types: bool,
        num_node_types: Optional[int],
        # Generation parameters
        max_nodes: int,
        min_nodes: int,
        patience: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        edge_sample_attempts: int = 1  # Unused by rnn_edge_gen
) -> Optional[nx.DiGraph]:
    """ Generates a DAG using GraphLevelRNN and EdgeLevelRNN models. """
    # --- Validation and Setup ---
    if not isinstance(node_model, GraphLevelRNN): logger.warning(f"generate expects node_model type GraphLevelRNN")
    if not isinstance(edge_model, EdgeLevelRNN): logger.warning(f"generate expects edge_model type EdgeLevelRNN")
    if edge_gen_function is not rnn_edge_gen: logger.warning(f"generate expects edge_gen_function=rnn_edge_gen")
    if predict_node_types and num_node_types is None: logger.error(
        "Node prediction enabled but num_node_types not provided."); return None
    if edge_feature_len != NUM_EDGE_FEATURES: logger.warning(
        f"edge_feature_len mismatch: Expected {NUM_EDGE_FEATURES}, got {edge_feature_len}.")

    if hasattr(node_model, 'eval'): node_model.eval()
    if hasattr(edge_model, 'eval'): edge_model.eval()
    try:
        device = next(node_model.parameters()).device
    except:
        device = torch.device('cpu'); node_model.to(device)
    try:
        edge_model.to(device)
    except Exception as e:
        logger.error(f"Failed to move edge_model to device {device}: {e}")

    # --- MODIFIED: Use sample_temperature ---
    sample_fun = sample_temperature
    # --- END MODIFIED ---

    min_nodes = max(1, min_nodes)
    if max_nodes < min_nodes: max_nodes = min_nodes; logger.warning(
        f"max_nodes < min_nodes. Setting max_nodes = {max_nodes}.")
    patience = max(1, patience)
    # --- End Validation ---

    # --- Generation Loop ---
    # Initial input: SOS token (all NONE edges)
    adj_vec_input_node = torch.zeros([1, 1, input_size, NUM_EDGE_FEATURES], device=device)
    adj_vec_input_node[:, :, :, EDGE_TYPES_INTERNAL["NONE"]] = 1.0

    list_adj_indices = []  # Stores np arrays of sampled edge indices (0, 1, 2) for each node
    list_predicted_node_types = []  # Stores predicted node type index for each node

    if hasattr(node_model, 'reset_hidden'): node_model.reset_hidden()
    if hasattr(edge_model, 'reset_hidden'): edge_model.reset_hidden()

    generated_nodes_count = 0
    no_edge_streak = 0

    logger.info(
        f"Starting generation: max_nodes={max_nodes}, min_nodes={min_nodes}, patience={patience}, "
        f"temp={temperature}, top_k={top_k}, top_p={top_p}, predict_nodes={predict_node_types}")

    with torch.no_grad():
        # --- Predict Node 0 Type (Assume PI or CONST0) ---
        # Run node model once with initial SOS input to get context for node 0 edges (if needed)
        # and predict type for node 0.
        initial_node_output = node_model(adj_vec_input_node)
        initial_predicted_node_type_idx = 0  # Default to CONST0

        if predict_node_types:
            if not isinstance(initial_node_output, tuple) or len(initial_node_output) != 2:
                logger.error("Initial node model call didn't return (output, logits). Cannot predict node 0 type.")
                return None
            _, initial_node_logits = initial_node_output
            initial_node_type_idx = sample_fun(
                initial_node_logits.squeeze(), temperature, top_k, top_p
            )
            logger.debug(
                f"  Node 0: Predicted type logits: {initial_node_logits.squeeze().cpu().numpy()}, Sampled index: {initial_node_type_idx}")
        else:
            # Assign default type if not predicting (e.g., PI if node 0 is always input)
            initial_predicted_node_type_idx = aig_config.NODE_TYPE_TO_INT.get("PI", 1)  # Example default

        list_predicted_node_types.append(initial_predicted_node_type_idx)
        generated_nodes_count = 1  # Count node 0
        # Note: Node 0 has no incoming edges, so list_adj_indices remains empty for step 0.

        # --- Loop for Nodes 1 to max_nodes-1 ---
        for i in range(1, max_nodes):
            current_node_idx = i
            logger.debug(f"Generating node {current_node_idx}...")

            # --- Node Model Forward Pass & Type Prediction ---
            predicted_node_type_idx = 0  # Default
            try:
                # Input is the one-hot encoding of edges generated for the *previous* node (i-1)
                node_model_output = node_model(adj_vec_input_node)  # Pass edge input from previous step

                if predict_node_types:
                    if not isinstance(node_model_output, tuple) or len(node_model_output) != 2:
                        logger.error("Node model didn't return (output, logits) when predict_node_types=True.")
                        return None
                    h, node_type_logits = node_model_output
                    # Logits for the current node i are the last in the sequence output
                    current_node_logits = node_type_logits[0, -1, :]  # Assuming batch=1, take last sequence item
                    predicted_node_type_idx = sample_fun(
                        current_node_logits, temperature, top_k, top_p
                    )
                    logger.debug(
                        f"  Node {i}: Predicted type logits: {current_node_logits.cpu().numpy()}, Sampled index: {predicted_node_type_idx}")
                else:
                    h = node_model_output
                    # Assign default type if not predicting (e.g., AND)
                    predicted_node_type_idx = aig_config.NODE_TYPE_TO_INT.get("AND", 2)  # Example default

                list_predicted_node_types.append(predicted_node_type_idx)

                # Initialize/Set edge hidden state using node output 'h'
                if hasattr(edge_model, 'set_first_layer_hidden') and callable(edge_model.set_first_layer_hidden):
                    try:
                        edge_model.set_first_layer_hidden(h)
                    except Exception as e:
                        logger.error(f"Error setting edge hidden state for node {i}: {e}"); return None

            except Exception as e:
                logger.error(f"Error during NodeModel forward/Edge init for node {i}: {e}", exc_info=True)
                return None
            # --- End Node Model ---

            # --- Edge Prediction ---
            num_edges_to_generate = min(current_node_idx, input_size)
            current_indices_np = np.array([], dtype=int)

            if num_edges_to_generate > 0:
                logger.debug(f"  Generating {num_edges_to_generate} edges into node {i}...")
                try:
                    # Call edge gen function - it returns indices directly now
                    adj_indices_vec_tensor = edge_gen_function(
                        edge_model, h, num_edges_to_generate, input_size, sample_fun, mode,
                        temperature, top_k, top_p, edge_sample_attempts
                    )  # Returns shape [1, 1, input_size] with indices

                    # Extract relevant indices and convert to numpy
                    current_indices_np = adj_indices_vec_tensor[0, 0, :num_edges_to_generate].cpu().numpy()

                except Exception as e:
                    logger.error(f"Error during EdgeModel generation for node {i}: {e}", exc_info=True)
                    return None

            list_adj_indices.append(current_indices_np)  # Append indices for node i
            generated_nodes_count += 1
            # --- End Edge Prediction ---

            # --- Patience Stopping Check ---
            if generated_nodes_count >= min_nodes:
                only_no_edge_predicted = np.all(
                    current_indices_np <= EDGE_TYPES_INTERNAL["NONE"]) if current_indices_np.size > 0 else True
                if only_no_edge_predicted:
                    no_edge_streak += 1
                else:
                    no_edge_streak = 0
                logger.debug(f"  Node {i}: Edge streak: {no_edge_streak}/{patience}")
                if no_edge_streak >= patience:
                    logger.info(f"Stopping early at node {i}: reached patience={patience}.")
                    break
            # --- End Patience Check ---

            # --- Prepare input for the NEXT node iteration ---
            # Convert the sampled *indices* back to one-hot for the node model input
            next_adj_vec_input = torch.zeros([1, 1, input_size, NUM_EDGE_FEATURES], device=device)
            next_adj_vec_input[:, :, :, EDGE_TYPES_INTERNAL["NONE"]] = 1.0  # Default to NONE
            num_indices = len(current_indices_np)
            len_slice = min(num_indices, input_size)
            for k in range(len_slice):
                class_idx = current_indices_np[k]
                if 0 <= k < input_size:  # Check index is within bounds for input vec
                    if 0 <= class_idx < NUM_EDGE_FEATURES:  # Check index is valid edge type
                        next_adj_vec_input[0, 0, k, EDGE_TYPES_INTERNAL["NONE"]] = 0.0  # Clear NONE flag
                        next_adj_vec_input[0, 0, k, class_idx] = 1.0  # Set one-hot flag
                    # else: keep the default NONE if class_idx is invalid (shouldn't happen)

            adj_vec_input = next_adj_vec_input  # Use this for the next iteration
            # --- End Input Prep ---

    # --- Post-processing After Loop ---
    if generated_nodes_count == 0: logger.warning("Generation resulted in 0 nodes."); return None
    if generated_nodes_count < min_nodes:
        logger.warning(f"Generated nodes ({generated_nodes_count}) < min_nodes ({min_nodes}). Returning None.")
        return None

    logger.info(f"Building NetworkX graph for {generated_nodes_count} generated nodes...")
    try:
        # list_adj_indices contains vectors for nodes 1 to n-1
        # list_predicted_node_types contains types for nodes 0 to n-1
        final_graph = build_graph_from_indices(
            list_adj_indices,
            list_predicted_node_types,
            input_size,  # m
            generated_nodes_count  # n
        )
        logger.debug(
            f"Graph building successful. Nodes: {final_graph.number_of_nodes()}, Edges: {final_graph.number_of_edges()}")
        return final_graph
    except Exception as e:
        logger.error(f"Error building NetworkX graph: {e}", exc_info=True)
        return None


# Example of how to use if run standalone (for testing)
if __name__ == '__main__':
    print("This script defines generation functions. Run get_aigs.py or a similar control script to execute.")
    # Add basic tests here if desired, e.g., creating dummy models and calling generate

