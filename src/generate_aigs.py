# generate_aigs.py
import numpy as np
import torch
import logging
import os # Added for path joining
import pickle # Added for saving graphs
from typing import Optional, Tuple, Callable, Any, Dict, List
from .model import * # Import necessary model classes
import networkx as nx
import sys # Added for sys.exit

# --- Logger Setup ---
# Configure logger early
logger = logging.getLogger("generate_aigs")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Import the AIG configuration for type strings ---
# Ensure aig_config.py is in the same directory or Python path
try:
    import aig_config as aig_config
except ImportError:
    # Attempt relative import if run from src/
    try:
        from . import aig_config
        logger.info("Imported aig_config using relative path.")
    except ImportError:
        logger.error("Failed to import AIG configuration from 'aig_config.py' or '.aig_config'. Ensure it's accessible.")
        sys.exit(1)


# --- Constants ---
# Internal mapping used during generation sampling
EDGE_TYPES_INTERNAL = {"NONE": 0, "REGULAR": 1, "INVERTED": 2}
NUM_EDGE_FEATURES = 3 # Number of classes model predicts (None, Reg, Inv)

# Map internal integer indices back to config string keys for output graph
INT_TO_EDGE_TYPE_STR = {
    EDGE_TYPES_INTERNAL["REGULAR"]: aig_config.EDGE_TYPE_KEYS[0], # e.g., "EDGE_REG"
    EDGE_TYPES_INTERNAL["INVERTED"]: aig_config.EDGE_TYPE_KEYS[1], # e.g., "EDGE_INV"
}

# Node type strings from config for assignment
NODE_CONST0_STR = aig_config.NODE_TYPE_KEYS[0] # e.g., "NODE_CONST0"
NODE_PI_STR = aig_config.NODE_TYPE_KEYS[1]     # e.g., "NODE_PI"
NODE_AND_STR = aig_config.NODE_TYPE_KEYS[2]    # e.g., "NODE_AND"
NODE_PO_STR = aig_config.NODE_TYPE_KEYS[3]     # e.g., "NODE_PO"
NODE_UNKNOWN_STR = "UNKNOWN_NODE" # Fallback


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
    Returns the sampled class index (0, 1, or 2).
    """
    if logits.numel() == 0:
        logger.error("Cannot sample from empty logits tensor.")
        return EDGE_TYPES_INTERNAL["NONE"] # Return default/safe value

    if temperature <= 0:
        # Deterministic sampling: return the most likely class
        return torch.argmax(logits).item()

    # Apply temperature scaling
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # Apply top-p filtering
    if top_p > 0.0 and top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Keep elements until cumulative probability exceeds top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the mask to the right to keep the first element that exceeds p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # Scatter the mask back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        # Apply the mask
        probs = probs.masked_fill(indices_to_remove, 0.0)
        # Re-normalize if necessary
        prob_sum = torch.sum(probs)
        if prob_sum > 1e-6:
             probs = probs / prob_sum
        else:
             logger.warning("top_p filtered all probabilities. Falling back to uniform.")
             probs = torch.ones_like(logits) / logits.numel() # Fallback

    # Apply top-k filtering (can be combined with top-p, applied after)
    elif top_k > 0:
        top_k = min(top_k, probs.size(-1)) # Ensure k is valid
        # Keep only top-k probabilities, set others to 0
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(-1, top_k_indices, 1.0)
        probs = probs * mask
        # Re-normalize if necessary
        prob_sum = torch.sum(probs)
        if prob_sum > 1e-6:
             probs = probs / prob_sum
        else:
             logger.warning("top_k filtered all probabilities. Falling back to uniform.")
             probs = torch.ones_like(logits) / logits.numel() # Fallback

    # Sample from the (potentially filtered and re-normalized) distribution
    probs_np = probs.detach().cpu().numpy()
    # Ensure non-negative probabilities and normalize again for numpy safety
    probs_np = np.maximum(probs_np, 0)
    prob_sum_np = np.sum(probs_np)
    if prob_sum_np > 1e-6:
        probs_np = probs_np / prob_sum_np
    else:
        # If sum is still ~0, assign uniform probability
        logger.warning(f"Probabilities sum to {prob_sum_np} after filtering. Using uniform distribution.")
        probs_np = np.ones(len(probs_np)) / len(probs_np)

    try:
        # Sample using numpy's choice function
        sampled_index = np.random.choice(range(len(probs_np)), p=probs_np)
    except ValueError as e:
        logger.error(f"Value error during np.random.choice: {e}. Probs: {probs_np}. Falling back to argmax.")
        # Fallback to argmax if choice fails (e.g., due to rounding errors in probs)
        # Use original logits before filtering for fallback argmax
        sampled_index = torch.argmax(logits * temperature).item() # Use original tempered logits

    return sampled_index


# --- Edge Generation Helper Functions (Unchanged - they output indices 0, 1, 2) ---
def rnn_edge_gen(edge_rnn,
                 node_output_context,  # Node output sequence 'h'
                 num_edges, adj_vec_size, sample_fun, mode,
                 temperature=1.0, top_k=0, top_p=0.0, attempts=None):
    """ Generates edge indices (0, 1, or 2) using an RNN edge model. """
    try:
        device = node_output_context.device
    except AttributeError as e:
        logger.error(f"Could not get device from node_output_context (type: {type(node_output_context)}). Error: {e}")
        try:
            device = next(edge_rnn.parameters()).device; logger.warning("Falling back to edge_rnn device.")
        except Exception:
            raise RuntimeError("Cannot determine device for rnn_edge_gen.") from e

    # Stores the sampled class index for each potential edge
    adj_indices_vec = torch.full((1, 1, adj_vec_size), EDGE_TYPES_INTERNAL["NONE"], dtype=torch.long, device=device)

    # Initial SOS token (one-hot for class 0 - NoEdge)
    # Use NUM_EDGE_FEATURES (which is 3) for the one-hot vector size
    x = torch.zeros([1, 1, NUM_EDGE_FEATURES], device=device)
    x[0, 0, EDGE_TYPES_INTERNAL["NONE"]] = 1.0 # Use float for model input

    # Initialize hidden state if needed (logic adapted from train.py)
    if hasattr(edge_rnn, 'hidden') and edge_rnn.hidden is None:
        batch_size = 1 # Generation is typically batch size 1
        num_layers = edge_rnn.num_layers
        hidden_size = edge_rnn.hidden_size
        if isinstance(edge_rnn, (EdgeLevelRNN, EdgeLevelAttentionRNN)):
            edge_rnn.hidden = torch.zeros(num_layers, batch_size, hidden_size, device=device)
            logger.debug("RNN edge_gen: Initialized GRU edge hidden state.")
        elif isinstance(edge_rnn, (EdgeLevelLSTM, EdgeLevelAttentionLSTM)) or hasattr(edge_rnn, 'lstm'):
            h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
            c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
            edge_rnn.hidden = (h_0, c_0)
            logger.debug("RNN edge_gen: Initialized LSTM edge hidden state.")

    # Set first layer hidden state if possible (only if not already initialized above)
    # This logic assumes the edge_rnn might need initialization from node context
    # if hasattr(edge_rnn, 'hidden') and edge_rnn.hidden is None and hasattr(edge_rnn, 'set_first_layer_hidden'):
    #      try:
    #          edge_rnn.set_first_layer_hidden(node_output_context)
    #          logger.debug("RNN edge_gen: Set first layer hidden from node context.")
    #      except Exception as e:
    #          logger.error(f"RNN edge_gen: Error setting first layer hidden: {e}. Hidden state remains None/uninitialized.")
             # Consider initializing to zeros here as a fallback if set_first_layer_hidden fails

    # Generation loop
    for i in range(num_edges):
        if not hasattr(edge_rnn, 'hidden') or edge_rnn.hidden is None:
            logger.error(f"RNN edge_gen: Hidden state is None before forward pass for edge {i}. Cannot proceed.")
            # Return partially filled vector or raise error
            return adj_indices_vec # Return what we have so far

        try:
            # Pass node_context=h if the edge model expects it (for attention/fusion)
            # Adjust based on your specific EdgeLevelAttentionRNN/LSTM implementation
            if isinstance(edge_rnn, (EdgeLevelAttentionRNN, EdgeLevelAttentionLSTM)):
                 raw_output = edge_rnn(x, node_context=node_output_context, return_logits=True)
            else:
                 raw_output = edge_rnn(x, return_logits=True)
        except TypeError:
            logger.debug("Edge RNN forward doesn't accept return_logits, assuming output is logits.")
            try:
                if isinstance(edge_rnn, (EdgeLevelAttentionRNN, EdgeLevelAttentionLSTM)):
                     raw_output = edge_rnn(x, node_context=node_output_context)
                else:
                     raw_output = edge_rnn(x)
            except Exception as e:
                logger.error(f"Error during edge_rnn forward pass: {e}")
                # Attempt to recover by resetting hidden state? Risky during generation.
                # Best to return current progress or raise.
                return adj_indices_vec # Return partial results

        # Ensure raw_output has the expected dimensions before indexing
        if raw_output is None or raw_output.dim() < 3:
             logger.error(f"Unexpected output shape from edge_rnn: {raw_output.shape if raw_output is not None else 'None'}")
             return adj_indices_vec

        logits = raw_output[0, 0, :] # Assuming [batch=1, seq=1, num_classes] output

        sampled_class_index = sample_fun(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        adj_indices_vec[0, 0, i] = sampled_class_index

        # Prepare next input (one-hot encoding of the sampled class)
        next_x_one_hot = torch.zeros_like(x)
        # Use NUM_EDGE_FEATURES (which is 3) for checking bounds
        if 0 <= sampled_class_index < NUM_EDGE_FEATURES:
            next_x_one_hot[0, 0, sampled_class_index] = 1.0
        else:
            logger.warning(f"RNN Gen: Invalid class index {sampled_class_index} sampled. Using NONE.")
            next_x_one_hot[0, 0, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
        x = next_x_one_hot

    return adj_indices_vec


def mlp_edge_gen(edge_mlp, h, num_edges, adj_vec_size, sample_fun, mode,
                 temperature=1.0, top_k=0, top_p=0.0, attempts=1):
    """ Generates edge indices (0, 1, or 2) using an MLP edge model. """
    device = h.device
    adj_indices_vec = torch.full((1, 1, adj_vec_size), EDGE_TYPES_INTERNAL["NONE"], dtype=torch.long, device=device)

    try:
        edge_logits = edge_mlp(h, return_logits=True)
    except TypeError:
        logger.debug("Edge MLP forward doesn't accept return_logits, assuming output is logits.")
        edge_logits = edge_mlp(h)

    # Expected shape: [batch=1, seq=1, output_size=m*features] or [batch=1, seq=1, m, features=3]
    # Handle both cases
    if edge_logits.dim() == 4 and edge_logits.shape[1] == 1: # [1, 1, m, C]
        edge_logits = edge_logits.squeeze(1) # -> [1, m, C]
    elif edge_logits.dim() == 3 and edge_logits.shape[1] == 1: # [1, 1, m*C]
        try:
            # Reshape based on NUM_EDGE_FEATURES (which is 3)
            edge_logits = edge_logits.view(1, adj_vec_size, NUM_EDGE_FEATURES)
        except RuntimeError as e:
            logger.error(f"MLP output shape mismatch: Cannot reshape {edge_logits.shape} to [1, {adj_vec_size}, {NUM_EDGE_FEATURES}]. Error: {e}")
            return adj_indices_vec
    elif edge_logits.dim() != 3 or edge_logits.shape[0] != 1 or edge_logits.shape[2] != NUM_EDGE_FEATURES: # Check if already [1, m, C=3]
        logger.error(f"Unexpected MLP output shape: {edge_logits.shape}. Expected ~[1, m, {NUM_EDGE_FEATURES}].")
        return adj_indices_vec

    num_logits_provided = edge_logits.shape[1] # This should be 'm'

    for attempt_num in range(attempts):
        sampled_indices_list = []
        edges_to_process = min(num_edges, num_logits_provided)

        for i in range(edges_to_process):
            logits = edge_logits[0, i, :] # Logits for edge i
            sampled_class_index = sample_fun(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            sampled_indices_list.append(sampled_class_index)

        if sampled_indices_list:
            # Check sampled indices against NUM_EDGE_FEATURES
            valid_indices = [idx for idx in sampled_indices_list if 0 <= idx < NUM_EDGE_FEATURES]
            if len(valid_indices) != len(sampled_indices_list):
                 logger.warning("MLP Gen: Invalid indices sampled, some skipped.")

            if valid_indices:
                current_attempt_indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
                # Ensure slicing doesn't go out of bounds
                len_to_assign = min(len(valid_indices), adj_indices_vec.shape[2])
                adj_indices_vec[0, 0, :len_to_assign] = current_attempt_indices[:len_to_assign]

        # Check if any edge > NONE was sampled in the relevant part
        if (adj_indices_vec[0, 0, :edges_to_process] > EDGE_TYPES_INTERNAL["NONE"]).any():
            break
        if attempts > 1:
             logger.debug(f"MLP Edge Gen: No edge > NONE sampled on attempt {attempt_num + 1}. Retrying if possible.")

    return adj_indices_vec


# --- MODIFIED: Graph Building ---
def build_graph_from_indices(list_adj_indices: List[np.ndarray], m: int, final_num_nodes: int) -> nx.DiGraph:
    """
    Builds a NetworkX DiGraph directly from the generated edge index lists.
    Edges are added with a STRING 'type' attribute.
    Nodes are added WITH GUESSED 'type' attributes based on heuristics.
    """
    G = nx.DiGraph()
    if final_num_nodes <= 0:
        return G

    # --- Step 1: Add nodes (initially without type) ---
    G.add_nodes_from(range(final_num_nodes))

    # --- Step 2: Add edges with string types ---
    for target_node_idx, class_indices in enumerate(list_adj_indices, start=1):
        num_connections_possible = min(target_node_idx, m)
        actual_indices_len = len(class_indices)
        len_to_process = min(actual_indices_len, num_connections_possible)

        for k in range(len_to_process):
            source_node_idx = (target_node_idx - 1) - k
            if source_node_idx < 0: continue

            edge_class_int = class_indices[k] # This is 0, 1, or 2
            edge_type_str = INT_TO_EDGE_TYPE_STR.get(edge_class_int)

            if edge_type_str: # Add edge only if not NONE
                if source_node_idx in G and target_node_idx in G:
                    G.add_edge(source_node_idx, target_node_idx, type=edge_type_str)
                else:
                     logger.warning(f"Skipping edge ({source_node_idx}-{target_node_idx}) due to missing node during build.")

    # --- Step 3: Guess Node Types based on degree heuristics ---
    logger.debug(f"Assigning guessed node types for {final_num_nodes} nodes...")
    assigned_node_count = 0
    for node_idx in range(final_num_nodes):
        if node_idx not in G: # Should not happen, but check
            logger.warning(f"Node {node_idx} not found in graph during type assignment.")
            continue

        try:
            in_deg = G.in_degree(node_idx)
            out_deg = G.out_degree(node_idx)
        except Exception as e:
            logger.warning(f"Could not get degree for node {node_idx} during type guessing: {e}. Assigning UNKNOWN.")
            G.nodes[node_idx]['type'] = NODE_UNKNOWN_STR
            continue

        guessed_type = NODE_UNKNOWN_STR # Default

        # --- Apply Heuristics with stricter PO check ---
        if node_idx == 0:
            guessed_type = NODE_CONST0_STR # Node 0 is always CONST0
        elif in_deg == 0:
            guessed_type = NODE_PI_STR # In-degree 0 (and not node 0) is PI
        elif out_deg == 0 and in_deg == 1: # <<< MODIFIED PO Check
            guessed_type = NODE_PO_STR # Out-degree 0 AND In-degree 1 is PO
        elif in_deg == 2:
            guessed_type = NODE_AND_STR # In-degree 2 is likely AND
        else:
            # Fallback for other cases (e.g., PO with wrong in-degree, other combos)
            guessed_type = NODE_UNKNOWN_STR # Assign UNKNOWN if it doesn't fit a clear pattern
            logger.debug(f"Node {node_idx} (in={in_deg}, out={out_deg}): Does not match heuristic for CONST0, PI, PO(in=1), or AND(in=2). Assigning {guessed_type}.")
        # --- End Heuristics ---

        G.nodes[node_idx]['type'] = guessed_type
        assigned_node_count += 1

    if assigned_node_count != final_num_nodes:
        logger.warning(f"Node type assignment mismatch: Assigned {assigned_node_count}, expected {final_num_nodes}")

    return G


# --- MODIFIED: Generate Function ---
def generate(
        node_model: torch.nn.Module,
        edge_model: torch.nn.Module,
        input_size: int,  # This is 'm'
        edge_gen_function: Callable,
        mode: str, # Keep mode if edge_gen_function needs it
        edge_feature_len: int, # Keep for validation/input prep
        # Generation parameters
        max_nodes: int,
        min_nodes: int,
        patience: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        edge_sample_attempts: int = 1
) -> Optional[nx.DiGraph]: # Return Type is NetworkX graph
    """
    Generates a DAG (like an AIG) using the provided models and parameters.
    Uses dynamic stopping based on patience mechanism.
    Returns: NetworkX DiGraph object with STRING edge types and GUESSED node types,
             or None on failure.
    """
    if node_model is None or edge_model is None or edge_gen_function is None:
        logger.error("Generation failed: Model components not provided.")
        return None

    if hasattr(node_model, 'eval'): node_model.eval()
    if hasattr(edge_model, 'eval'): edge_model.eval()

    try:
        device = next(node_model.parameters()).device
    except StopIteration:
        logger.warning("Could not determine device from node_model parameters. Using CPU.")
        device = torch.device('cpu')
        if hasattr(node_model, 'to'): node_model.to(device)

    try:
        if hasattr(edge_model, 'to'): edge_model.to(device)
    except Exception as e:
        logger.error(f"Failed to move edge_model to device {device}: {e}")

    sample_fun = sample_softmax # Assuming multi-class AIG generation

    # --- Input Parameter Validation ---
    # Validate against the internal 3-class system used by the model
    if edge_feature_len != NUM_EDGE_FEATURES:
        logger.warning(f"edge_feature_len mismatch: Expected {NUM_EDGE_FEATURES} internally, got {edge_feature_len}.")
        # Proceed, but input prep assumes 3 features
    min_nodes = max(1, min_nodes)
    if max_nodes < min_nodes:
        logger.warning(f"max_nodes ({max_nodes}) < min_nodes ({min_nodes}). Setting max_nodes = min_nodes.")
        max_nodes = min_nodes
    patience = max(1, patience)
    # --- End Validation ---

    # Initial SOS input for the node model (uses NUM_EDGE_FEATURES = 3)
    adj_vec_input = torch.zeros([1, 1, input_size, NUM_EDGE_FEATURES], device=device)
    adj_vec_input[:, :, :, EDGE_TYPES_INTERNAL["NONE"]] = 1.0

    list_adj_indices = [] # Store lists of sampled edge class indices (0, 1, or 2)
    if hasattr(node_model, 'reset_hidden') and callable(node_model.reset_hidden): node_model.reset_hidden()
    if hasattr(edge_model, 'reset_hidden') and callable(edge_model.reset_hidden): edge_model.reset_hidden()

    generated_nodes_count = 0
    no_edge_streak = 0

    # Initialize edge model hidden state if needed (copied from rnn_edge_gen init logic)
    if hasattr(edge_model, 'hidden') and edge_model.hidden is None:
        batch_size = 1
        if isinstance(edge_model, (EdgeLevelRNN, EdgeLevelAttentionRNN)):
            edge_model.hidden = torch.zeros(edge_model.num_layers, batch_size, edge_model.hidden_size, device=device)
        elif isinstance(edge_model, (EdgeLevelLSTM, EdgeLevelAttentionLSTM)) or hasattr(edge_model, 'lstm'):
            h_0 = torch.zeros(edge_model.num_layers, batch_size, edge_model.hidden_size, device=device)
            c_0 = torch.zeros(edge_model.num_layers, batch_size, edge_model.hidden_size, device=device)
            edge_model.hidden = (h_0, c_0)

    logger.info(
        f"Starting generation: max_nodes={max_nodes}, min_nodes={min_nodes}, patience={patience}, temp={temperature}, top_k={top_k}, top_p={top_p}")

    with torch.no_grad():
        for i in range(max_nodes):
            current_node_idx = i # 0-indexed node generation
            logger.debug(f"Generating node {current_node_idx}...")

            # --- Node Prediction ---
            # NOTE: We are NOT predicting node types here. They will be guessed later.

            try:
                h = node_model(adj_vec_input) # Get node model output sequence/context
                # Initialize/Set edge hidden state using node output 'h'
                if hasattr(edge_model, 'set_first_layer_hidden') and callable(edge_model.set_first_layer_hidden):
                     try:
                         # Ensure h is appropriate shape for set_first_layer_hidden
                         edge_model.set_first_layer_hidden(h)
                     except Exception as e:
                         logger.error(f"Error setting edge model hidden state for node {i}: {e}. Stopping generation.")
                         return None # Stop if hidden state setup fails
                elif hasattr(edge_model, 'hidden') and edge_model.hidden is None:
                     # Re-initialize if set_first_layer_hidden wasn't used/available and hidden is still None
                     logger.warning(f"Re-initializing edge hidden state for node {i}.")
                     batch_size = 1
                     if isinstance(edge_model, (EdgeLevelRNN, EdgeLevelAttentionRNN)):
                         edge_model.hidden = torch.zeros(edge_model.num_layers, batch_size, edge_model.hidden_size, device=device)
                     elif isinstance(edge_model, (EdgeLevelLSTM, EdgeLevelAttentionLSTM)) or hasattr(edge_model, 'lstm'):
                         h_0 = torch.zeros(edge_model.num_layers, batch_size, edge_model.hidden_size, device=device)
                         c_0 = torch.zeros(edge_model.num_layers, batch_size, edge_model.hidden_size, device=device)
                         edge_model.hidden = (h_0, c_0)

            except Exception as e:
                logger.error(f"Error during NodeModel forward pass for node {i}: {e}", exc_info=True)
                return None

            # --- Edge Prediction ---
            num_edges_to_generate = min(current_node_idx, input_size) # Edges into node i come from nodes 0 to i-1
            current_indices_np = np.array([], dtype=int) # Default for node 0

            if num_edges_to_generate > 0:
                logger.debug(f"  Generating {num_edges_to_generate} potential edges into node {i}...")
                try:
                    # Pass node output 'h' as context to the edge generator
                    adj_indices_vec = edge_gen_function(
                        edge_model,
                        h, # Pass the node output sequence
                        num_edges=num_edges_to_generate,
                        adj_vec_size=input_size,
                        sample_fun=sample_fun,
                        mode=mode, # Pass mode if needed by edge_gen_function
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        attempts=edge_sample_attempts
                    )
                    # Extract the sampled indices (class indices 0, 1, or 2)
                    current_indices_np = adj_indices_vec[0, 0, :num_edges_to_generate].cpu().numpy()
                except Exception as e:
                    logger.error(f"Error during EdgeModel generation for node {i}: {e}", exc_info=True)
                    return None # Stop generation if edge model fails

            list_adj_indices.append(current_indices_np)
            generated_nodes_count += 1

            # --- Patience Stopping Check ---
            if generated_nodes_count >= min_nodes:
                # Check if only NONE edges were predicted for this step
                only_no_edge_predicted = np.all(current_indices_np <= EDGE_TYPES_INTERNAL["NONE"]) if current_indices_np.size > 0 else True
                if only_no_edge_predicted:
                    no_edge_streak += 1
                    logger.debug(f"  Node {i}: No edge > NONE predicted. Streak: {no_edge_streak}/{patience}")
                else:
                    if no_edge_streak > 0: logger.debug(f"  Node {i}: Edge > NONE predicted. Resetting streak.")
                    no_edge_streak = 0 # Reset streak if any edge > NONE is predicted
                # Stop if streak reaches patience
                if no_edge_streak >= patience:
                    logger.info(f"Stopping early at node {i} (total {generated_nodes_count}): reached patience={patience}.")
                    break
            # --- End Patience Check ---

            # --- Prepare input for the NEXT node iteration ---
            # Input uses the 3-class one-hot representation
            next_adj_vec_input = torch.zeros([1, 1, input_size, NUM_EDGE_FEATURES], device=device)
            next_adj_vec_input[:, :, :, EDGE_TYPES_INTERNAL["NONE"]] = 1.0 # Default to NONE
            num_indices = len(current_indices_np)
            len_slice = min(num_indices, input_size)
            for k in range(len_slice):
                class_idx = current_indices_np[k] # This is 0, 1, or 2
                if 0 <= k < input_size:
                    # Use NUM_EDGE_FEATURES for bounds check
                    if 0 <= class_idx < NUM_EDGE_FEATURES:
                        next_adj_vec_input[0, 0, k, EDGE_TYPES_INTERNAL["NONE"]] = 0.0 # Clear NONE flag
                        next_adj_vec_input[0, 0, k, class_idx] = 1.0 # Set the sampled class flag
                    else: # If invalid index sampled, keep as NONE
                        next_adj_vec_input[0, 0, k, :] = 0.0
                        next_adj_vec_input[0, 0, k, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
            adj_vec_input = next_adj_vec_input
            # --- End Input Prep ---

    # --- Post-processing After Loop ---
    if generated_nodes_count == 0:
        logger.warning("Generation resulted in 0 nodes.")
        return None
    elif generated_nodes_count < min_nodes:
        logger.warning(f"Generation stopped early before min_nodes ({min_nodes}) was reached. Generated: {generated_nodes_count}")
        # Return the incomplete graph anyway, or None if strict min_nodes is required
        # return None # Uncomment this line if you require min_nodes to be met

    logger.info(f"Building NetworkX graph for {generated_nodes_count} generated nodes...")
    try:
        # Use the modified function to build the graph with STRING edge types
        # and GUESSED node types
        final_graph = build_graph_from_indices(list_adj_indices, input_size, generated_nodes_count)
        logger.debug(f"Graph building successful. Nodes: {final_graph.number_of_nodes()}, Edges: {final_graph.number_of_edges()}")
        return final_graph
    except Exception as e:
        logger.error(f"Error building NetworkX graph: {e}", exc_info=True)
        return None

# Example of how to use if run standalone (for testing)
if __name__ == '__main__':
    print("This file contains generation functions. Run get_aigs.py to execute.")

