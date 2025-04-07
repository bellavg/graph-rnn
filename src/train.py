import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np # Needed for debugging print if used

# Assume NUM_EDGE_FEATURES is available, e.g., from aig_dataset or config
# If not, determine dynamically from tensor shapes where possible.


def train_mlp_step(graph_rnn, edge_mlp, data,
                   criterion_edge, # Renamed from criterion
                   optim_graph_rnn, optim_edge_mlp,
                   scheduler_graph_rnn, scheduler_mlp, device, use_edge_features):
    """ Train GraphRNN with MLP edge model (Level Embedding Added). """
    graph_rnn.zero_grad()
    edge_mlp.zero_grad()

    s, lens_cpu = data['x'].float().to(device), data['len'].cpu() # Renamed lens
    batch_size, seq_len_padded, effective_m, num_features = s.shape

    # --- Get and prepare levels tensor ---
    levels_padded = data.get('levels')
    if levels_padded is not None:
        levels_padded = levels_padded.long().to(device)
    # --- End NEW ---

    # Add dummy feature dimension if needed
    if len(s.shape) == 3:
        s = s.unsqueeze(3)
        num_features = 1

    # Prepare inputs/targets with SOS/EOS frames
    one_frame = torch.ones([batch_size, 1, effective_m, num_features], device=device)
    zero_frame = torch.zeros([batch_size, 1, effective_m, num_features], device=device)
    x = torch.cat((one_frame, s), dim=1)
    y = torch.cat((s, zero_frame), dim=1)
    lens_with_sos = lens_cpu + 1

    # --- Prepare Levels tensor for GraphRNN input ---
    levels_for_rnn = None
    if levels_padded is not None and hasattr(graph_rnn, 'level_embedding') and graph_rnn.level_embedding is not None:
        sos_level = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        target_level_len = seq_len_padded
        levels_sliced = levels_padded[:, :target_level_len]
        levels_for_rnn = torch.cat((sos_level, levels_sliced), dim=1)

        if levels_for_rnn.shape[1] != x.shape[1]:
            print(f"Warning: Level tensor shape mismatch in train_mlp_step. Adjusting.")
            target_len = x.shape[1]
            levels_for_rnn = torch.cat((sos_level, levels_padded[:, :target_len - 1]), dim=1)
            if levels_for_rnn.shape[1] != target_len:
                print("ERROR: Cannot align level tensor. Disabling.")
                levels_for_rnn = None
    # --- End NEW ---

    # --- GraphRNN Forward Pass ---
    graph_rnn.reset_hidden()
    # Pass levels to graph_rnn. Assume it returns only hidden if node prediction is off.
    hidden = graph_rnn(x, lens_with_sos, levels=levels_for_rnn)
    # hidden shape: [batch, seq_len_padded+1, hidden_size_rnn]
    # --- End Modification ---

    # --- EdgeMLP Forward Pass ---
    y_pred = edge_mlp(hidden, return_logits=use_edge_features)
    # y_pred shape: [batch, seq_len_padded+1, effective_m, num_edge_classes]

    # --- Edge Loss Calculation (Revised based on train.py context) ---
    edge_loss = torch.tensor(0.0, device=device)
    try:
        # Pack target sequence y based on actual lengths
        y_packed = pack_padded_sequence(y, lens_with_sos.cpu(), batch_first=True, enforce_sorted=False)
        # Pad back to max length in this batch for comparison
        y_padded, _ = pad_packed_sequence(y_packed, batch_first=True, total_length=y.shape[1])
    except RuntimeError as e:
         print(f"Error during pack/pad_packed_sequence for y in train_mlp_step: {e}")
         # Return 0.0 loss to match original snippet's single return value style
         return 0.0

    if use_edge_features: # Multi-class edge prediction (AIG case)
        pred_batch, pred_seq, pred_m, pred_classes = y_pred.shape
        target_batch, target_seq, target_m, target_features = y_padded.shape

        if pred_batch == target_batch and pred_seq == target_seq and pred_m == target_m:
            num_classes = pred_classes
            y_pred_flat = y_pred.reshape(-1, num_classes) # [Batch * Seq * M, NumClasses]

            try:
                y_labels = torch.argmax(y_padded, dim=-1) # [Batch, Seq, M]
                y_labels_flat = y_labels.reshape(-1)      # [Batch * Seq * M]
            except RuntimeError as e:
                print(f"Error argmax/reshape y_padded in train_mlp_step: {e}")
                y_labels_flat = None

            if y_labels_flat is not None:
                mask = torch.arange(pred_seq, device=device)[None, :] < lens_with_sos.to(device)[:, None] # [Batch, Seq]
                mask = mask.unsqueeze(-1).expand(-1, -1, effective_m) # [Batch, Seq, M]
                mask_flat = mask.reshape(-1) # [Batch * Seq * M]

                if y_pred_flat.shape[0] == y_labels_flat.shape[0] == mask_flat.shape[0]:
                    valid_preds = y_pred_flat[mask_flat]
                    valid_labels = y_labels_flat[mask_flat]
                    if valid_labels.numel() > 0:
                         edge_loss = criterion_edge(valid_preds, valid_labels.long())
                else:
                     print(f"Warning: Flattened shape mismatch in MLP edge loss calculation. Skipping loss.")
        else:
             print(f"Warning: Dimension mismatch y_pred vs y_padded in MLP step. Skipping edge loss.")

    else: # Binary edge prediction
        print("Warning: Using binary edge loss path in MLP step.")
        mask = torch.arange(y_pred.shape[1], device=device)[None, :] < lens_with_sos.to(device)[:, None]
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(y_pred)
        if torch.any(mask):
             edge_loss = criterion_edge(y_pred[mask], y_padded[mask])
    # --- End Edge Loss Calculation ---

    # Total loss (only edge loss in this version)
    total_loss = edge_loss

    # Backpropagate
    if torch.isnan(total_loss) or torch.isinf(total_loss):
         print(f"Warning: Invalid loss detected (NaN or Inf): {total_loss.item()}. Skipping backpropagation.")
    elif total_loss.requires_grad and total_loss > 0:
        total_loss.backward()
        optim_graph_rnn.step()
        optim_edge_mlp.step()

    # Step schedulers
    scheduler_graph_rnn.step()
    scheduler_mlp.step()

    # Return loss value (matching original snippet style)
    return total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss


def train_rnn_step(graph_rnn, edge_rnn, data,
                   criterion, # Assuming this is criterion_edge
                   optim_graph_rnn,
                   optim_edge_rnn, # Changed from optim_edge_mlp
                   scheduler_graph_rnn,
                   scheduler_edge_rnn, # Changed from scheduler_mlp
                   device, use_edge_features):
    """ Train GraphRNN with RNN edge model. """
    graph_rnn.zero_grad()
    edge_rnn.zero_grad()

    seq, lens = data['x'].float().to(device), data['len'].cpu()
    batch_size, seq_len_padded, effective_m, num_features = seq.shape # Get shape info

    # --- NEW: Get and prepare levels tensor ---
    levels_padded = data.get('levels')  # Get levels tensor from data dictionary
    if levels_padded is not None:
        levels_padded = levels_padded.long().to(device) # Shape: [batch, max_n-1]
    # --- End NEW ---

    # If s does not have edge features, just add a dummy dimension 1
    # to the end (Note: This check might be redundant if edge_feature_len=3 always)
    if len(seq.shape) == 3:
        seq = seq.unsqueeze(3)
        num_features = 1 # Update num_features if dummy dim added

    # --- Prepare Node RNN Input ---
    # Add SOS token. Note: The logic here differs slightly from the other train step.
    # Using seq[:, :-1, :] might truncate the last actual step for some graphs.
    # Consider using the full sequence `seq` and adjusting lengths `lens + 1`.
    # Let's align with the logic from train_mlp_step for consistency:
    one_frame_node = torch.ones([batch_size, 1, effective_m, num_features], device=device)
    x_node_rnn_full = torch.cat((one_frame_node, seq), dim=1) # Shape: [batch, seq_len_padded+1, ...]
    lens_node_rnn = lens + 1 # Lengths including SOS token
    # --- End Input Prep Modification ---


    # --- NEW: Prepare Levels tensor for GraphRNN input ---
    levels_for_rnn = None
    if levels_padded is not None and hasattr(graph_rnn, 'level_embedding') and graph_rnn.level_embedding is not None:
        # levels_padded corresponds to nodes 1 to max_n-1
        # We need levels for input steps 0 (SOS) to max_n-1 (or seq_len_padded)
        sos_level = torch.zeros(batch_size, 1, dtype=torch.long, device=device) # Level 0 for SOS

        # Slice padded levels to match the length of the actual sequence part of x_node_rnn_full
        # x_node_rnn_full has length seq_len_padded + 1
        # We need levels for indices 1 to seq_len_padded
        target_level_len = seq_len_padded
        levels_sliced = levels_padded[:, :target_level_len] # Shape: [batch, seq_len_padded]

        # Concatenate SOS level with sliced levels
        levels_for_rnn = torch.cat((sos_level, levels_sliced), dim=1) # Shape: [batch, seq_len_padded+1]

        # Final shape check
        if levels_for_rnn.shape[1] != x_node_rnn_full.shape[1]:
            print(
                f"Warning: Level tensor seq length ({levels_for_rnn.shape[1]}) mismatch "
                f"with input x seq length ({x_node_rnn_full.shape[1]}) in train_rnn_step. Adjusting."
            )
            target_len = x_node_rnn_full.shape[1]
            # Attempt slicing adjustment
            levels_for_rnn = torch.cat((sos_level, levels_padded[:, :target_len - 1]), dim=1)
            if levels_for_rnn.shape[1] != target_len: # Check again
                print("ERROR: Cannot align level tensor length. Disabling level embedding for this batch.")
                levels_for_rnn = None # Disable if alignment fails
    # --- End NEW ---


    # --- Compute hidden graph-level representation ---
    graph_rnn.reset_hidden()
    # MODIFIED: Pass lens_node_rnn and levels_for_rnn
    # Assuming graph_rnn might return (hidden, node_type_logits) tuple if predicting node types
    output = graph_rnn(x_node_rnn_full, lens_node_rnn, levels=levels_for_rnn)
    if isinstance(output, tuple): # Handle potential tuple output
        hidden = output[0]
        # node_type_logits = output[1] # If node loss needed later
    else:
        hidden = output
    # hidden shape: [batch, seq_len_padded+1, hidden_size_rnn]
    # --- End Modification ---

    # --- Edge RNN Processing (Original logic follows, may need review) ---

    # Pack hidden states and target sequence
    try:
        # Pack hidden states corresponding to inputs for nodes 0 to n-1 (output steps 1 to n)
        hidden_for_edges_packed = pack_padded_sequence(hidden[:, 1:, :], lens, batch_first=True, enforce_sorted=False)
        hidden_packed_data = hidden_for_edges_packed.data # Shape [TotalNodes, HiddenSize]

        # Pack the target edge sequence 'seq' based on 'lens'
        seq_packed_obj = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
        y_edge_rnn = seq_packed_obj.data # Target edges: Shape [TotalNodes, effective_m, EdgeFeatures]
        seq_packed = y_edge_rnn # Use the packed data directly

    except RuntimeError as e:
         print(f"Error during pack_padded_sequence for hidden/seq in train_rnn_step: {e}")
         # Return 0 or raise error - returning 0 to match original snippet's implicit behavior
         return 0.0

    total_nodes = seq_packed.shape[0]
    if total_nodes == 0:
        print("Warning: Zero total nodes after packing in train_rnn_step. Skipping loss calculation.")
        return 0.0

    # We now need to compute the sequence lengths of `seq_packed`.
    # TODO: Do this more efficiently (Copied from original snippet)
    seq_packed_len = []
    m = effective_m # Use shape derived earlier
    for l in lens.tolist(): # Use tolist() for iteration
        for i in range(1, l + 1):
            seq_packed_len.append(min(i, m))

    # Convert to tensor and ensure it's on the correct device
    seq_packed_len_tensor = torch.tensor(seq_packed_len, dtype=torch.long, device=device)

    # Ensure packed lengths match TotalNodes (Added check)
    if len(seq_packed_len) != total_nodes:
         print(f"Warning: Mismatch between calculated edge_seq_lens ({len(seq_packed_len)}) and total_nodes ({total_nodes}) in train_rnn_step.")
         # Simple fix: Trim or pad seq_packed_len_tensor if possible, or return error
         if len(seq_packed_len) > total_nodes:
             seq_packed_len_tensor = seq_packed_len_tensor[:total_nodes]
         else:
             # This case is harder to fix, indicates packing issue upstream. Return 0 loss.
             print("Error: seq_packed_len is shorter than total_nodes. Cannot proceed.")
             return 0.0


    # Add SOS token to the edge-level RNN input
    # Use num_features derived from seq shape
    one_frame_edge = torch.ones([total_nodes, 1, num_features], device=device)
    # Input is SOS + target sequence shifted
    x_edge_rnn = torch.cat((one_frame_edge, seq_packed[:, :-1, :]), dim=1)
    # Target is the original packed sequence
    y_edge_rnn_target = seq_packed

    # Set hidden state for EdgeRNN
    edge_rnn.set_first_layer_hidden(hidden_packed_data)

    # Compute edge predictions
    # Pass seq_packed_len_tensor.cpu() as lengths must be on CPU for pack_padded_sequence
    y_edge_rnn_pred = edge_rnn(x_edge_rnn, seq_packed_len_tensor.cpu(), return_logits=use_edge_features)

    # --- Loss Calculation (Original logic follows, may need review) ---
    # Pad the target sequence based on calculated lengths
    try:
        y_edge_rnn_target_packed = pack_padded_sequence(y_edge_rnn_target, seq_packed_len_tensor.cpu(), batch_first=True, enforce_sorted=False)
        # Pad back to the max length observed in the predictions for comparison
        y_edge_rnn_target_padded, _ = pad_packed_sequence(y_edge_rnn_target_packed, batch_first=True, total_length=y_edge_rnn_pred.shape[1])
    except RuntimeError as e:
        print(f"Error packing/padding y_edge_rnn_target: {e}")
        return 0.0 # Return 0 loss on error


    # --- Masking and Loss (Adapting from train_mlp_step logic) ---
    edge_loss = torch.tensor(0.0, device=device)
    max_pred_len = y_edge_rnn_pred.shape[1] # Max length in the edge prediction sequence
    # Create mask based on actual lengths
    edge_mask = torch.arange(max_pred_len, device=device)[None, :] < seq_packed_len_tensor[:, None] # Shape: [TotalNodes, max_pred_len]

    if use_edge_features:
        # CrossEntropyLoss expects [N, C] predictions and [N] labels
        pred_nodes, pred_len, pred_classes = y_edge_rnn_pred.shape
        target_nodes, target_len, target_features = y_edge_rnn_target_padded.shape

        if pred_nodes != target_nodes or pred_len != target_len:
             print(f"Warning: Shape mismatch pred ({y_edge_rnn_pred.shape}) vs target ({y_edge_rnn_target_padded.shape}) in RNN step. Skipping edge loss.")
        else:
             num_classes = pred_classes
             # Flatten predictions: [TotalNodes * max_pred_len, NumClasses]
             y_pred_flat = y_edge_rnn_pred.reshape(-1, num_classes)

             # Convert targets (one-hot) to labels: [TotalNodes * max_pred_len]
             try:
                 y_labels_flat = torch.argmax(y_edge_rnn_target_padded, dim=-1).reshape(-1)
             except RuntimeError as e:
                 print(f"Error argmax/reshape y_edge_rnn_target_padded: {e}")
                 y_labels_flat = None

             if y_labels_flat is not None:
                 # Flatten mask: [TotalNodes * max_pred_len]
                 mask_flat = edge_mask.reshape(-1)

                 if y_pred_flat.shape[0] == y_labels_flat.shape[0] == mask_flat.shape[0]:
                     valid_preds = y_pred_flat[mask_flat]
                     valid_labels = y_labels_flat[mask_flat]
                     if valid_labels.numel() > 0:
                          # Ensure criterion is defined correctly outside this function
                          edge_loss = criterion(valid_preds, valid_labels.long())
                 else:
                     print(f"Warning: Flattened shape mismatch in RNN edge loss calculation. Skipping.")

    else: # Binary case (BCELoss - requires sigmoid)
        print("Warning: Using binary edge loss path for RNN step.")
        mask_expanded = edge_mask.unsqueeze(-1).expand_as(y_edge_rnn_pred)
        if torch.any(mask_expanded):
            # Ensure y_edge_rnn_pred has sigmoid if using BCELoss
            edge_loss = criterion(y_edge_rnn_pred[mask_expanded], y_edge_rnn_target_padded[mask_expanded])
    # --- End Masking and Loss ---

    # --- Backpropagation and Optimizing ---
    # Note: This snippet only calculates edge_loss. A complete version
    # should also calculate node_loss if applicable and sum them.
    total_loss = edge_loss # Assuming only edge loss for now based on snippet

    if torch.isnan(total_loss) or torch.isinf(total_loss):
         print(f"Warning: Invalid loss detected (NaN or Inf): {total_loss.item()}. Skipping backpropagation.")
    elif total_loss.requires_grad and total_loss > 0: # Check requires_grad and non-zero
        total_loss.backward()
        # Clip gradients?
        torch.nn.utils.clip_grad_norm_(graph_rnn.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(edge_rnn.parameters(), max_norm=1.0)
        optim_graph_rnn.step()
        optim_edge_rnn.step() # Use the correct optimizer for EdgeRNN

    # Step schedulers
    scheduler_graph_rnn.step()
    scheduler_edge_rnn.step() # Use the correct scheduler for EdgeRNN
    # --- End Backpropagation ---

    # Return loss dictionary (Matching the other train step)
    return {
        'total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        'edge': edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
        'node': 0.0 # Placeholder - node loss not calculated in this snippet
    }