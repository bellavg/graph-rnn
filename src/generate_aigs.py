# generate_aigs.py (Modified for Node Type Prediction & GRU/RNN Only)
import numpy as np
import torch
import logging
import os # Added for path joining
import pickle # Added for saving graphs
from typing import Optional, Tuple, Callable, Any, Dict, List
# --- MODIFIED: Import only necessary models ---
from model import GraphLevelRNN, EdgeLevelRNN
# --- END MODIFIED ---
import networkx as nx
import sys # Added for sys.exit

# --- Logger Setup ---
logger = logging.getLogger("generate_aigs")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Import the AIG configuration for type strings ---
try:
    import aig_config as aig_config
except ImportError:
    try:
        from . import aig_config
        logger.info("Imported aig_config using relative path.")
    except ImportError:
        logger.error("Failed to import AIG configuration from 'aig_config.py' or '.aig_config'. Ensure it's accessible.")
        sys.exit(1)


# --- Constants ---
EDGE_TYPES_INTERNAL = {"NONE": 0, "REGULAR": 1, "INVERTED": 2}
NUM_EDGE_FEATURES = 3 # Number of edge classes model predicts (None, Reg, Inv)

INT_TO_EDGE_TYPE_STR = {
    EDGE_TYPES_INTERNAL["REGULAR"]: aig_config.EDGE_TYPE_KEYS[0],
    EDGE_TYPES_INTERNAL["INVERTED"]: aig_config.EDGE_TYPE_KEYS[1],
}

INT_TO_NODE_TYPE_STR = {
    i: node_key for i, node_key in enumerate(aig_config.NODE_TYPE_KEYS)
}
if len(INT_TO_NODE_TYPE_STR) != aig_config.NUM_NODE_FEATURES:
     logger.warning(f"Mismatch between derived INT_TO_NODE_TYPE_STR ({len(INT_TO_NODE_TYPE_STR)}) and aig_config.NUM_NODE_FEATURES ({aig_config.NUM_NODE_FEATURES})")

NODE_UNKNOWN_STR = "UNKNOWN_NODE"

# --- Sampling Functions (Unchanged) ---
def sample_bernoulli(p):
    """ Samples 0 or 1 based on probability p. """
    p_val = p.item() if isinstance(p, torch.Tensor) else p
    return int(np.random.random() < p_val)

def sample_softmax(logits: torch.Tensor,
                   temperature: float = 1.0,
                   top_k: int = 0,
                   top_p: float = 0.0) -> int:
    """
    Samples an index from logits using temperature, top-k, or top-p sampling.
    Returns the sampled class index.
    """
    if logits.numel() == 0:
        logger.error("Cannot sample from empty logits tensor.")
        return 0

    if temperature <= 0:
        return torch.argmax(logits).item()

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p > 0.0 and top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        prob_sum = torch.sum(probs)
        if prob_sum > 1e-6:
             probs = probs / prob_sum
        else:
             logger.warning("top_p filtered all probabilities. Falling back to uniform.")
             probs = torch.ones_like(logits) / logits.numel()

    elif top_k > 0:
        top_k = min(top_k, probs.size(-1))
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(-1, top_k_indices, 1.0)
        probs = probs * mask
        prob_sum = torch.sum(probs)
        if prob_sum > 1e-6:
             probs = probs / prob_sum
        else:
             logger.warning("top_k filtered all probabilities. Falling back to uniform.")
             probs = torch.ones_like(logits) / logits.numel()

    probs_np = probs.detach().cpu().numpy()
    probs_np = np.maximum(probs_np, 0)
    prob_sum_np = np.sum(probs_np)
    if prob_sum_np > 1e-6:
        probs_np = probs_np / prob_sum_np
    else:
        logger.warning(f"Probabilities sum to {prob_sum_np} after filtering. Using uniform.")
        probs_np = np.ones(len(probs_np)) / len(probs_np)

    try:
        sampled_index = np.random.choice(range(len(probs_np)), p=probs_np)
    except ValueError as e:
        logger.error(f"Value error during np.random.choice: {e}. Probs: {probs_np}. Falling back to argmax.")
        sampled_index = torch.argmax(logits * temperature).item()

    return sampled_index


# --- MODIFIED: Edge Generation Helper Function (RNN Only) ---
def rnn_edge_gen(edge_rnn: EdgeLevelRNN, # Type hint specific model
                 node_output_context,
                 num_edges, adj_vec_size, sample_fun, mode,
                 temperature=1.0, top_k=0, top_p=0.0, attempts=None): # attempts is unused here
    """ Generates edge indices (0, 1, or 2) using an EdgeLevelRNN model. """
    # Determine device
    try:
        # Prefer context device if available and it's a tensor
        if isinstance(node_output_context, torch.Tensor):
            device = node_output_context.device
        else:
            device = next(edge_rnn.parameters()).device
    except Exception as e:
        logger.error(f"Could not determine device: {e}. Defaulting to CPU.")
        device = torch.device('cpu')

    # Stores the sampled class index for each potential edge
    adj_indices_vec = torch.full((1, 1, adj_vec_size), EDGE_TYPES_INTERNAL["NONE"], dtype=torch.long, device=device)

    # Initial SOS token (one-hot for class 0 - NoEdge)
    x = torch.zeros([1, 1, NUM_EDGE_FEATURES], device=device)
    x[0, 0, EDGE_TYPES_INTERNAL["NONE"]] = 1.0 # Use float for model input

    # Initialize hidden state if needed
    # Assumes edge_rnn is an instance of EdgeLevelRNN or similar with .hidden attribute
    if hasattr(edge_rnn, 'hidden') and edge_rnn.hidden is None:
        batch_size = 1
        num_layers = getattr(edge_rnn, 'num_layers', 1)
        hidden_size = getattr(edge_rnn, 'hidden_size', 128) # Provide a default
        # Initialize GRU hidden state
        edge_rnn.hidden = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        logger.debug("RNN edge_gen: Initialized GRU edge hidden state.")
    # NOTE: Removed LSTM init logic

    # Generation loop
    for i in range(num_edges):
        if not hasattr(edge_rnn, 'hidden') or edge_rnn.hidden is None:
            logger.error(f"RNN edge_gen: Hidden state is None before forward pass for edge {i}. Cannot proceed.")
            return adj_indices_vec

        try:
            # Call EdgeLevelRNN forward pass
            # Assuming it doesn't need node_output_context directly in forward
            raw_output = edge_rnn(x, return_logits=True)
        except TypeError:
            logger.debug("Edge RNN forward doesn't accept return_logits, assuming output is logits.")
            try:
                raw_output = edge_rnn(x)
            except Exception as e:
                logger.error(f"Error during edge_rnn forward pass: {e}")
                return adj_indices_vec # Return partial results

        # Validate output shape
        if raw_output is None or raw_output.dim() < 3:
             logger.error(f"Unexpected output shape from edge_rnn: {raw_output.shape if raw_output is not None else 'None'}")
             return adj_indices_vec

        # Get logits for the current edge prediction
        logits = raw_output[0, 0, :] # Assuming [batch=1, seq=1, num_classes] output

        # Sample the edge type index
        sampled_class_index = sample_fun(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        adj_indices_vec[0, 0, i] = sampled_class_index

        # Prepare input for the next edge prediction step
        next_x_one_hot = torch.zeros_like(x)
        if 0 <= sampled_class_index < NUM_EDGE_FEATURES: # Check against 3 edge types
            next_x_one_hot[0, 0, sampled_class_index] = 1.0
        else:
            logger.warning(f"RNN Gen: Invalid edge class index {sampled_class_index} sampled. Using NONE.")
            next_x_one_hot[0, 0, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
        x = next_x_one_hot

    return adj_indices_vec
# --- END MODIFIED ---

# --- REMOVED mlp_edge_gen function ---

# --- MODIFIED: Graph Building (Unchanged from previous version, already uses predicted types) ---
def build_graph_from_indices(
        list_adj_indices: List[np.ndarray],
        list_predicted_node_types: List[int], # List of predicted node type indices
        m: int,
        final_num_nodes: int
    ) -> nx.DiGraph:
    """
    Builds a NetworkX DiGraph from generated edge index lists and predicted node type indices.
    Edges are added with STRING 'type' attributes.
    Nodes are added with PREDICTED STRING 'type' attributes.
    """
    G = nx.DiGraph()
    if final_num_nodes <= 0:
        return G

    # Step 1: Add nodes WITH predicted types
    if len(list_predicted_node_types) != final_num_nodes:
        logger.error(f"Mismatch between predicted node types ({len(list_predicted_node_types)}) and final node count ({final_num_nodes}). Cannot build graph accurately.")
        return G # Return empty graph on error

    logger.debug(f"Assigning predicted node types for {final_num_nodes} nodes...")
    for node_idx in range(final_num_nodes):
        predicted_type_idx = list_predicted_node_types[node_idx]
        node_type_str = INT_TO_NODE_TYPE_STR.get(predicted_type_idx, NODE_UNKNOWN_STR)
        if node_type_str == NODE_UNKNOWN_STR:
            logger.warning(f"Node {node_idx}: Predicted type index {predicted_type_idx} is invalid. Assigning UNKNOWN.")
        G.add_node(node_idx, type=node_type_str)

    # Step 2: Add edges with string types
    for target_node_idx, class_indices in enumerate(list_adj_indices, start=1):
        if target_node_idx not in G:
            logger.warning(f"Target node {target_node_idx} not found in graph during edge addition. Skipping edges.")
            continue

        num_connections_possible = min(target_node_idx, m)
        actual_indices_len = len(class_indices)
        len_to_process = min(actual_indices_len, num_connections_possible)

        for k in range(len_to_process):
            source_node_idx = (target_node_idx - 1) - k
            if source_node_idx < 0: continue
            if source_node_idx not in G:
                logger.warning(f"Source node {source_node_idx} not found for edge to {target_node_idx}. Skipping edge.")
                continue

            edge_class_int = class_indices[k]
            edge_type_str = INT_TO_EDGE_TYPE_STR.get(edge_class_int)

            if edge_type_str: # Add edge only if not NONE
                G.add_edge(source_node_idx, target_node_idx, type=edge_type_str)

    return G


# --- MODIFIED: Generate Function (Signature unchanged, logic updated) ---
def generate(
        node_model: GraphLevelRNN, # Type hint specific model
        edge_model: EdgeLevelRNN, # Type hint specific model
        input_size: int,
        edge_gen_function: Callable, # Should be rnn_edge_gen
        mode: str,
        edge_feature_len: int,
        predict_node_types: bool,
        num_node_types: Optional[int],
        # Generation parameters
        max_nodes: int,
        min_nodes: int,
        patience: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        edge_sample_attempts: int = 1 # Note: unused by rnn_edge_gen currently
) -> Optional[nx.DiGraph]:
    """
    Generates a DAG using GraphLevelRNN and EdgeLevelRNN models.
    Optionally predicts node types during generation.
    Uses dynamic stopping based on patience mechanism.
    Returns: NetworkX DiGraph object with STRING edge types and PREDICTED node types,
             or None on failure.
    """
    # --- Validation and Setup (mostly unchanged) ---
    if not isinstance(node_model, GraphLevelRNN):
        logger.warning(f"generate expects node_model of type GraphLevelRNN, got {type(node_model)}")
    if not isinstance(edge_model, EdgeLevelRNN):
        logger.warning(f"generate expects edge_model of type EdgeLevelRNN, got {type(edge_model)}")
    if edge_gen_function is not rnn_edge_gen:
        logger.warning(f"generate expects edge_gen_function=rnn_edge_gen for these models, got {edge_gen_function.__name__}")
        # Decide whether to proceed or return None

    if predict_node_types and num_node_types is None:
        logger.error("Generation failed: Node prediction enabled but num_node_types not provided.")
        return None

    if hasattr(node_model, 'eval'): node_model.eval()
    if hasattr(edge_model, 'eval'): edge_model.eval()

    try: device = next(node_model.parameters()).device
    except StopIteration:
        logger.warning("Could not determine device from node_model. Using CPU.")
        device = torch.device('cpu')
        if hasattr(node_model, 'to'): node_model.to(device)
    try:
        if hasattr(edge_model, 'to'): edge_model.to(device)
    except Exception as e: logger.error(f"Failed to move edge_model to device {device}: {e}")

    sample_fun = sample_softmax

    if edge_feature_len != NUM_EDGE_FEATURES: logger.warning(f"edge_feature_len mismatch: Expected {NUM_EDGE_FEATURES}, got {edge_feature_len}.")
    min_nodes = max(1, min_nodes)
    if max_nodes < min_nodes: max_nodes = min_nodes; logger.warning(f"max_nodes < min_nodes. Setting max_nodes = {max_nodes}.")
    patience = max(1, patience)
    # --- End Validation ---

    # --- Generation Loop (logic for node prediction unchanged from previous version) ---
    adj_vec_input = torch.zeros([1, 1, input_size, NUM_EDGE_FEATURES], device=device)
    adj_vec_input[:, :, :, EDGE_TYPES_INTERNAL["NONE"]] = 1.0

    list_adj_indices = []
    list_predicted_node_types = []

    if hasattr(node_model, 'reset_hidden'): node_model.reset_hidden()
    if hasattr(edge_model, 'reset_hidden'): edge_model.reset_hidden()

    # Initialize edge model hidden state explicitly here if reset_hidden doesn't do it
    if hasattr(edge_model, 'hidden') and edge_model.hidden is None:
         batch_size = 1
         num_layers_edge = getattr(edge_model, 'num_layers', 1)
         hidden_size_edge = getattr(edge_model, 'hidden_size', 128)
         edge_model.hidden = torch.zeros(num_layers_edge, batch_size, hidden_size_edge, device=device)
         logger.debug("Initialized edge hidden state before loop.")


    generated_nodes_count = 0
    no_edge_streak = 0

    logger.info(
        f"Starting generation: max_nodes={max_nodes}, min_nodes={min_nodes}, patience={patience}, "
        f"temp={temperature}, top_k={top_k}, top_p={top_p}, predict_nodes={predict_node_types}")

    with torch.no_grad():
        for i in range(max_nodes):
            current_node_idx = i
            logger.debug(f"Generating node {current_node_idx}...")

            # --- Node Model Forward Pass & Type Prediction ---
            predicted_node_type_idx = 0 # Default
            try:
                node_model_output = node_model(adj_vec_input) # Pass edge input

                if predict_node_types:
                    if not isinstance(node_model_output, tuple) or len(node_model_output) != 2:
                        logger.error("Node model didn't return (output, logits) when predict_node_types=True.")
                        return None
                    h, node_type_logits = node_model_output
                    current_node_logits = node_type_logits[0, -1, :]
                    predicted_node_type_idx = torch.argmax(current_node_logits).item()
                    logger.debug(f"  Node {i}: Predicted type logits: {current_node_logits.cpu().numpy()}, Sampled index: {predicted_node_type_idx}")
                else:
                    h = node_model_output
                    # Fallback type assignment if not predicting
                    if i == 0: predicted_node_type_idx = 0 # CONST0 index
                    else: predicted_node_type_idx = 2 # AND index (example)

                list_predicted_node_types.append(predicted_node_type_idx)

                # Initialize/Set edge hidden state using node output 'h'
                if hasattr(edge_model, 'set_first_layer_hidden') and callable(edge_model.set_first_layer_hidden):
                     try: edge_model.set_first_layer_hidden(h)
                     except Exception as e: logger.error(f"Error setting edge hidden state: {e}"); return None
                elif hasattr(edge_model, 'hidden') and edge_model.hidden is None:
                     logger.warning(f"Edge hidden state None after node {i}, re-initializing.")
                     # Manual re-init if needed (should ideally be handled by reset/set_first_layer)
                     batch_size = 1; num_layers_edge=getattr(edge_model,'num_layers',1); hidden_size_edge=getattr(edge_model,'hidden_size',128)
                     edge_model.hidden = torch.zeros(num_layers_edge, batch_size, hidden_size_edge, device=device)

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
                    adj_indices_vec = edge_gen_function( # Should be rnn_edge_gen
                        edge_model, h, num_edges_to_generate, input_size, sample_fun, mode,
                        temperature, top_k, top_p, edge_sample_attempts # Pass attempts even if unused
                    )
                    current_indices_np = adj_indices_vec[0, 0, :num_edges_to_generate].cpu().numpy()
                except Exception as e:
                    logger.error(f"Error during EdgeModel generation for node {i}: {e}", exc_info=True)
                    return None

            list_adj_indices.append(current_indices_np)
            generated_nodes_count += 1
            # --- End Edge Prediction ---

            # --- Patience Stopping Check ---
            if generated_nodes_count >= min_nodes:
                only_no_edge_predicted = np.all(current_indices_np <= EDGE_TYPES_INTERNAL["NONE"]) if current_indices_np.size > 0 else True
                if only_no_edge_predicted: no_edge_streak += 1
                else: no_edge_streak = 0
                logger.debug(f"  Node {i}: Edge streak: {no_edge_streak}/{patience}")
                if no_edge_streak >= patience:
                    logger.info(f"Stopping early at node {i}: reached patience={patience}.")
                    break
            # --- End Patience Check ---

            # --- Prepare input for the NEXT node iteration ---
            next_adj_vec_input = torch.zeros([1, 1, input_size, NUM_EDGE_FEATURES], device=device)
            next_adj_vec_input[:, :, :, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
            num_indices = len(current_indices_np)
            len_slice = min(num_indices, input_size)
            for k in range(len_slice):
                class_idx = current_indices_np[k]
                if 0 <= k < input_size:
                    if 0 <= class_idx < NUM_EDGE_FEATURES:
                        next_adj_vec_input[0, 0, k, EDGE_TYPES_INTERNAL["NONE"]] = 0.0
                        next_adj_vec_input[0, 0, k, class_idx] = 1.0
                    else:
                        next_adj_vec_input[0, 0, k, :] = 0.0
                        next_adj_vec_input[0, 0, k, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
            adj_vec_input = next_adj_vec_input
            # --- End Input Prep ---

    # --- Post-processing After Loop ---
    if generated_nodes_count == 0: logger.warning("Generation resulted in 0 nodes."); return None
    elif generated_nodes_count < min_nodes: logger.warning(f"Generated nodes ({generated_nodes_count}) < min_nodes ({min_nodes}).")

    logger.info(f"Building NetworkX graph for {generated_nodes_count} generated nodes...")
    try:
        final_graph = build_graph_from_indices(
            list_adj_indices,
            list_predicted_node_types, # Pass predicted types
            input_size,
            generated_nodes_count
        )
        logger.debug(f"Graph building successful. Nodes: {final_graph.number_of_nodes()}, Edges: {final_graph.number_of_edges()}")
        return final_graph
    except Exception as e:
        logger.error(f"Error building NetworkX graph: {e}", exc_info=True)
        return None

# Example of how to use if run standalone (for testing)
if __name__ == '__main__':
    print("This file contains generation functions. Run get_aigs.py to execute.")

