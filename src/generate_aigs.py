# generate_aigs.py
import numpy as np
import torch
import logging
from typing import Optional, Tuple, Callable, Any, Dict, List

# --- Constants ---
# Define necessary constants here if not imported from elsewhere
EDGE_TYPES = {"NONE": 0, "REGULAR": 1, "INVERTED": 2}
NUM_EDGE_FEATURES = 3 # Ensure this matches your model and dataset

# --- Logger ---
logger = logging.getLogger("generate_aigs")
# Basic config if run standalone, but usually configured by the main script
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Sampling Functions ---
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
    """
    if logits.numel() == 0:
        logger.error("Cannot sample from empty logits tensor.")
        return EDGE_TYPES["NONE"] # Return default/safe value

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
        if torch.sum(probs) > 1e-6:
             probs = probs / torch.sum(probs)
        else:
             logger.warning("top_p filtered all probabilities. Falling back to uniform.")
             probs = torch.ones_like(logits) / logits.numel()

    elif top_k > 0:
        top_k = min(top_k, probs.size(-1))
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(-1, top_k_indices, 1.0)
        probs = probs * mask
        if torch.sum(probs) > 1e-6:
             probs = probs / torch.sum(probs)
        else:
             logger.warning("top_k filtered all probabilities. Falling back to uniform.")
             probs = torch.ones_like(logits) / logits.numel()

    probs_np = probs.detach().cpu().numpy()
    # Ensure non-negative probabilities and normalize
    probs_np = np.maximum(probs_np, 0)
    prob_sum = np.sum(probs_np)
    if prob_sum > 1e-6:
        probs_np = probs_np / prob_sum
    else:
        # If sum is still ~0, assign uniform probability
        logger.warning(f"Probabilities sum to {prob_sum} after filtering. Using uniform distribution.")
        probs_np = np.ones(len(probs_np)) / len(probs_np)

    try:
        sampled_index = np.random.choice(range(len(probs_np)), p=probs_np)
    except ValueError as e:
        logger.error(f"Value error during np.random.choice: {e}. Probs: {probs_np}. Falling back to argmax.")
        # Fallback to argmax if choice fails (e.g., due to rounding errors in probs)
        sampled_index = torch.argmax(logits * temperature).item() # Use original tempered logits

    return sampled_index


# --- Edge Generation Helper Functions ---
def rnn_edge_gen(edge_rnn, h, num_edges, adj_vec_size, sample_fun, mode,
                 temperature=1.0, top_k=0, top_p=0.0, attempts=None): # Added attempts for API consistency
    """ Generates edge indices using an RNN edge model. """
    device = h.device
    adj_indices_vec = torch.full((1, 1, adj_vec_size), EDGE_TYPES["NONE"], dtype=torch.long, device=device)

    # Check if set_first_layer_hidden exists and call it
    if hasattr(edge_rnn, 'set_first_layer_hidden') and callable(edge_rnn.set_first_layer_hidden):
        edge_rnn.set_first_layer_hidden(h)
    else:
        logger.warning("Edge RNN model does not have 'set_first_layer_hidden' method.")
        # Decide how to initialize hidden state if needed, or assume it's handled internally

    # Initial SOS token (assuming one-hot for class 0 - NoEdge)
    x = torch.zeros([1, 1, edge_rnn.edge_feature_len], device=device)
    x[0, 0, EDGE_TYPES["NONE"]] = 1

    for i in range(num_edges):
        # Ensure edge_rnn can return logits
        try:
            # Use return_logits=True if the model supports it
            raw_output = edge_rnn(x, return_logits=True)
        except TypeError:
            # Fallback if return_logits is not supported
            logger.debug("Edge RNN forward doesn't accept return_logits, assuming output is logits.")
            raw_output = edge_rnn(x)

        logits = raw_output[0, 0, :] # Assuming [batch=1, seq=1, features] output

        sampled_class_index = sample_fun(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        adj_indices_vec[0, 0, i] = sampled_class_index

        # Prepare next input
        next_x_one_hot = torch.zeros_like(x)
        if 0 <= sampled_class_index < edge_rnn.edge_feature_len:
            next_x_one_hot[0, 0, sampled_class_index] = 1
        else:
            logger.warning(f"RNN Gen: Invalid class index {sampled_class_index} sampled. Using NONE.")
            next_x_one_hot[0, 0, EDGE_TYPES["NONE"]] = 1
        x = next_x_one_hot

    return adj_indices_vec


def mlp_edge_gen(edge_mlp, h, num_edges, adj_vec_size, sample_fun, mode,
                 temperature=1.0, top_k=0, top_p=0.0, attempts=1):
    """ Generates edge indices using an MLP edge model. """
    device = h.device
    adj_indices_vec = torch.full((1, 1, adj_vec_size), EDGE_TYPES["NONE"], dtype=torch.long, device=device)

    # Ensure edge_mlp can return logits
    try:
        # Use return_logits=True if the model supports it
        edge_logits = edge_mlp(h, return_logits=True)
    except TypeError:
        # Fallback if return_logits is not supported
        logger.debug("Edge MLP forward doesn't accept return_logits, assuming output is logits.")
        edge_logits = edge_mlp(h)

    # Adjust slicing based on your MLP's actual output shape, aiming for [batch, m, num_classes]
    if edge_logits.dim() == 4 and edge_logits.shape[1] == 1: # Handle [1, 1, m, C]
        edge_logits = edge_logits.squeeze(1) # -> [1, m, C]
    elif edge_logits.dim() != 3:
        logger.error(f"Unexpected MLP output shape: {edge_logits.shape}. Expected ~[1, m, C].")
        return adj_indices_vec # Return empty/default

    num_logits_provided = edge_logits.shape[1] # This should be 'm'

    for attempt_num in range(attempts):
        sampled_indices_list = []
        # Generate only for available predecessors/logits, up to num_edges needed
        edges_to_process = min(num_edges, num_logits_provided)

        for i in range(edges_to_process):
            logits = edge_logits[0, i, :] # Logits for edge i (relative to current node)
            sampled_class_index = sample_fun(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            sampled_indices_list.append(sampled_class_index)

        # Store the sampled indices for this attempt
        if sampled_indices_list:
            # Ensure indices are within bounds before converting to tensor
            valid_indices = [idx for idx in sampled_indices_list if 0 <= idx < NUM_EDGE_FEATURES]
            if len(valid_indices) != len(sampled_indices_list):
                 logger.warning("MLP Gen: Invalid indices sampled, some skipped.")

            if valid_indices: # Only proceed if there are valid indices
                current_attempt_indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
                # Assign only up to the number of processed edges
                adj_indices_vec[0, 0, :len(valid_indices)] = current_attempt_indices # Adjust slice length

        # Check if any edge > 0 was sampled among the processed edges
        if (adj_indices_vec[0, 0, :edges_to_process] > EDGE_TYPES["NONE"]).any():
            break # Success, stop attempting
        if attempts > 1:
             logger.debug(f"MLP Edge Gen: No edge > NONE sampled on attempt {attempt_num + 1}. Retrying if possible.")

    return adj_indices_vec


# --- Matrix Building ---
def build_aig_matrices(list_adj_indices: List[np.ndarray], m: int, final_num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Builds connectivity and inversion matrices for an AIG from generated index lists. """
    n = final_num_nodes
    adj_conn = np.zeros((n, n), dtype=int)
    adj_inv = np.zeros((n, n), dtype=int)

    for i, class_indices in enumerate(list_adj_indices):
        current_node_idx = i # Node being generated (0-indexed)
        # class_indices[k] corresponds to potential source node current_node_idx - 1 - k
        num_connections_possible = min(current_node_idx, m)

        # We only generated/stored indices for possible connections
        actual_indices_len = len(class_indices)
        len_to_process = min(actual_indices_len, num_connections_possible)

        for k in range(len_to_process):
            source_node_idx = (current_node_idx - 1) - k
            if source_node_idx < 0: continue # Should not happen if k < current_node_idx

            edge_class = class_indices[k]

            if edge_class == EDGE_TYPES["REGULAR"]:
                adj_conn[source_node_idx, current_node_idx] = 1
                adj_inv[source_node_idx, current_node_idx] = 0
            elif edge_class == EDGE_TYPES["INVERTED"]:
                adj_conn[source_node_idx, current_node_idx] = 1
                adj_inv[source_node_idx, current_node_idx] = 1
            # Class EDGE_TYPES["NONE"] (0) means no edge, do nothing

    return adj_conn, adj_inv


# --- Main Generation Function ---
def generate(
    node_model: torch.nn.Module,
    edge_model: torch.nn.Module,
    input_size: int, # This is 'm'
    edge_gen_function: Callable,
    mode: str,
    edge_feature_len: int,
    # Generation parameters
    max_nodes: int,
    min_nodes: int,
    patience: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    edge_sample_attempts: int = 1
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generates a DAG (like an AIG) using the provided models and parameters.
    Uses dynamic stopping based on patience mechanism.
    Returns: Tuple (adj_conn, adj_inv) numpy arrays, or (None, None) on failure.
    """
    if node_model is None or edge_model is None or edge_gen_function is None:
        logger.error("Generation failed: Model components not provided.")
        return None, None

    # Set models to evaluation mode
    if hasattr(node_model, 'eval'): node_model.eval()
    if hasattr(edge_model, 'eval'): edge_model.eval()

    # Determine device from node_model parameters
    try:
        device = next(node_model.parameters()).device
        logger.info(f"Generation using device: {device}")
    except StopIteration:
        logger.warning("Could not determine device from node_model parameters. Using CPU.")
        device = torch.device('cpu')
        if hasattr(node_model, 'to'): node_model.to(device) # Try moving to CPU if empty

    # Ensure edge model is on the same device
    try:
        if hasattr(edge_model, 'to'): edge_model.to(device)
    except Exception as e:
         logger.error(f"Failed to move edge_model to device {device}: {e}")
         # Decide if fatal or try to continue
         # return None, None # Option to make it fatal

    sample_fun = sample_softmax

    # --- Input Parameter Validation ---
    if edge_feature_len != NUM_EDGE_FEATURES:
        logger.warning(f"edge_feature_len mismatch: Expected {NUM_EDGE_FEATURES}, got {edge_feature_len}.")
    min_nodes = max(1, min_nodes)
    if max_nodes < min_nodes:
        logger.warning(f"max_nodes ({max_nodes}) < min_nodes ({min_nodes}). Setting max_nodes = min_nodes.")
        max_nodes = min_nodes
    patience = max(1, patience)
    # --- End Validation ---

    # Initial SOS input for the node model
    # Shape: [batch=1, seq_len=1, input_size=m, edge_feature_len]
    adj_vec_input = torch.zeros([1, 1, input_size, edge_feature_len], device=device)
    # One-hot encode as "No Edge" (class 0) for all m potential predecessors
    adj_vec_input[:, :, :, EDGE_TYPES["NONE"]] = 1

    list_adj_indices = []
    # Reset hidden state if the method exists
    if hasattr(node_model, 'reset_hidden') and callable(node_model.reset_hidden):
         node_model.reset_hidden()
         logger.debug("Node model hidden state reset.")

    generated_nodes_count = 0
    no_edge_streak = 0

    logger.info(f"Starting generation: max_nodes={max_nodes}, min_nodes={min_nodes}, patience={patience}, temp={temperature}, top_k={top_k}, top_p={top_p}")

    with torch.no_grad():
        for i in range(max_nodes): # Loop up to the maximum allowed nodes
            current_node_idx = i
            logger.debug(f"Generating node {current_node_idx}...")

            # 1. Get hidden state from node model
            try:
                 # For generation, node_model usually takes only the previous step's output
                 h = node_model(adj_vec_input) # Pass input: [1, 1, m, C]
                 # Handle LSTM tuple output (h_n, c_n) -> take h_n for edge model input
                 # Assuming edge models are designed to take the hidden state sequence output, not (h_n, c_n)
                 if isinstance(h, tuple):
                     # If edge model needs h_n: h = h[0] # Shape [num_layers, batch, hidden]
                     # If edge model needs output sequence: keep h as is (output of LSTM layer)
                     # The provided edge models seem to expect the output sequence 'h'
                     # Let's assume h is the output sequence [batch, seq=1, hidden]
                     pass # Keep the output sequence for edge model input
                     # BUT, the set_first_layer_hidden takes h[0:1, :, :] assuming h is like GRU hidden state
                     # Let's adjust to pass h_n from LSTM if it exists
                     h_n, c_n = node_model.hidden # Assuming hidden stores (h_n, c_n)
                     h_for_edge = h_n # Pass the actual hidden state h_n
                 else:
                     # GRU output h is the sequence output, hidden is also updated internally
                     # set_first_layer_hidden expects the hidden state
                     h_for_edge = node_model.hidden # Pass the updated hidden state

                 # Ensure h_for_edge has the correct shape for edge models [layers, batch, hidden]
                 # Models expect h to initialize hidden state, need consistency
                 # Let's stick to passing the *last hidden state* (h_n or equivalent)
                 # Adjust `set_first_layer_hidden` if needed.

            except Exception as e:
                 logger.error(f"Error during NodeModel forward pass for node {i}: {e}", exc_info=True)
                 return None, None

            # 2. Generate edge *indices* for connections *into* the current node 'i'
            num_edges_to_generate = min(i, input_size) # How many predecessors exist and fit in 'm'
            current_indices = np.array([], dtype=int) # Default for node 0

            if num_edges_to_generate > 0:
                logger.debug(f"  Generating {num_edges_to_generate} potential edges into node {i}...")
                try:
                     # Call the appropriate edge generator (RNN or MLP)
                     adj_indices_vec = edge_gen_function(
                        edge_model,
                        h_for_edge, # Pass the relevant state from node model
                        num_edges=num_edges_to_generate,
                        adj_vec_size=input_size, # Max size ('m') for the output vector
                        sample_fun=sample_fun,
                        mode=mode,
                        # Pass sampling params
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        attempts=edge_sample_attempts # For MLP attempts
                     )
                     # Get the generated indices for the current node
                     # Shape is [1, 1, adj_vec_size], take the relevant slice
                     current_indices = adj_indices_vec[0, 0, :num_edges_to_generate].cpu().numpy()
                except Exception as e:
                     logger.error(f"Error during EdgeModel generation for node {i}: {e}", exc_info=True)
                     return None, None

            # Store the generated indices for this node
            list_adj_indices.append(current_indices)
            generated_nodes_count += 1

            # --- Patience Stopping Check ---
            num_nodes_generated_so_far = generated_nodes_count
            if num_nodes_generated_so_far >= min_nodes:
                # Check if *only* "NONE" edges were predicted among the generated indices
                only_no_edge_predicted = np.all(current_indices <= EDGE_TYPES["NONE"]) if current_indices.size > 0 else True

                if only_no_edge_predicted:
                    no_edge_streak += 1
                    logger.debug(f"  Node {i}: No edge > NONE predicted. Streak: {no_edge_streak}/{patience}")
                else:
                    if no_edge_streak > 0: logger.debug(f"  Node {i}: Edge > NONE predicted. Resetting streak.")
                    no_edge_streak = 0 # Reset streak if any edge was predicted

                if no_edge_streak >= patience:
                    logger.info(f"Stopping early at node {i} (total {num_nodes_generated_so_far}): "
                                f"reached patience={patience} of consecutive 'No Edge > NONE' steps.")
                    break # Exit the generation loop

            # 3. Prepare input for the *next* iteration (generating node i+1)
            # Create one-hot encoding based on `current_indices` for the *previous* step
            next_adj_vec_input = torch.zeros([1, 1, input_size, edge_feature_len], device=device)
            next_adj_vec_input[:, :, :, EDGE_TYPES["NONE"]] = 1 # Default to "No Edge"

            # Fill based on the edges generated *into node i* (stored in current_indices)
            # These describe connections from i-1, i-2, ..., i-m relative to node i.
            # For the input to node i+1, the k-th position represents the connection from node (i+1)-1-k = i-k.
            # The indices `current_indices[k]` describe edge type from `i-1-k -> i`.
            # We map this to the input vector for step i+1.
            num_indices = len(current_indices)
            len_slice = min(num_indices, input_size)

            for k in range(len_slice):
                 # class_idx is the type of edge from node i-1-k --> i
                 class_idx = current_indices[k]
                 # We need to place this information in the input vector for node i+1.
                 # The slot 'k' in the input vector corresponds to the edge from node i-k --> i+1.
                 # The GraphRNN structure relates input slot k to node i-k.
                 # Let's stick to the original interpretation: input slot k is edge from i-k.

                 # Position k in the input tensor corresponds to the edge from node i-k
                 # Check index bounds carefully
                 if 0 <= k < input_size:
                     # Check if class_idx is valid
                     if 0 <= class_idx < edge_feature_len:
                         next_adj_vec_input[0, 0, k, EDGE_TYPES["NONE"]] = 0 # Clear default NONE
                         next_adj_vec_input[0, 0, k, class_idx] = 1 # Set one-hot
                     else:
                         # This should ideally not happen if sampling is correct
                         logger.warning(f"Node {i}, Edge input prep: Invalid class index {class_idx} from generator. Using NONE.")
                         # Ensure it remains NONE
                         next_adj_vec_input[0, 0, k, :] = 0
                         next_adj_vec_input[0, 0, k, EDGE_TYPES["NONE"]] = 1

            # Update input for the next node generation step
            adj_vec_input = next_adj_vec_input

    # --- Post-processing After Loop ---
    if generated_nodes_count == 0:
         logger.warning("Generation resulted in 0 nodes.")
         return None, None
    elif generated_nodes_count < min_nodes and generated_nodes_count < max_nodes:
         logger.warning(f"Generation stopped early before min_nodes ({min_nodes}) was reached. Generated: {generated_nodes_count}")
         # Decide if this is acceptable or should return None
         # return None, None # Option to return None if min_nodes not met

    logger.info(f"Building AIG matrices for {generated_nodes_count} generated nodes...")
    try:
        adj_conn_final, adj_inv_final = build_aig_matrices(list_adj_indices, input_size, generated_nodes_count)
    except Exception as e:
        logger.error(f"Error building AIG matrices: {e}", exc_info=True)
        return None, None

    logger.info(f"Generation successful. Returning matrices for {generated_nodes_count} nodes.")
    return adj_conn_final, adj_inv_final

# Example of how to use if run standalone (for testing)
if __name__ == '__main__':
    print("This file contains generation functions. Run get_aigs.py to execute.")
    # Example: Test sample_softmax
    # test_logits = torch.tensor([1.0, 0.5, -0.5, 2.0])
    # print("Testing sample_softmax:")
    # for temp in [0.1, 1.0, 5.0]:
    #     samples = [sample_softmax(test_logits, temperature=temp) for _ in range(100)]
    #     print(f" Temp={temp}: Counts={dict(sorted(Counter(samples).items()))}")
    # print(f" Top-k=2 (T=1): Counts={dict(sorted(Counter([sample_softmax(test_logits, top_k=2) for _ in range(100)]).items()))}")
    # print(f" Top-p=0.8 (T=1): Counts={dict(sorted(Counter([sample_softmax(test_logits, top_p=0.8) for _ in range(100)]).items()))}")