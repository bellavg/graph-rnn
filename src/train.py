import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np # Needed for debugging print if used

# Assume NUM_EDGE_FEATURES is available, e.g., from aig_dataset or config
# If not, determine dynamically from tensor shapes where possible.

def train_mlp_step(graph_rnn, edge_mlp, data,
                   criterion_edge, criterion_node, # Use specific criteria
                   optim_graph_rnn, optim_edge_mlp,
                   scheduler_graph_rnn, scheduler_mlp, device,
                   use_edge_features, predict_node_types, use_conditioning):
    graph_rnn.zero_grad()
    edge_mlp.zero_grad()

    # Get mandatory data & NEW: levels
    s, lens = data['x'].float().to(device), data['len'].cpu() # s: [batch, seq_len_padded, effective_m, features]
    levels_padded = data.get('levels') # Get levels tensor
    if levels_padded is not None:
        levels_padded = levels_padded.long().to(device) # Shape: [batch, max_n-1]
    # --- End NEW ---


    # --- Basic Shape Checks and Setup ---
    if s.numel() == 0 or lens.numel() == 0 or torch.all(lens == 0):
         print("Warning: Received empty batch or batch with all zero lengths. Skipping step.")
         # Need to return dummy loss or handle appropriately upstream
         return {'total': 0.0, 'edge': 0.0, 'node': 0.0} # Example dummy return

    # Add feature dim if missing (handle legacy data format if necessary)
    if len(s.shape) == 3:
        print("Warning: Input tensor 's' has 3 dims, expected 4. Unsqueezing last dim.")
        s = s.unsqueeze(3)
    elif len(s.shape) != 4:
        raise ValueError(f"Input tensor 's' has unexpected shape: {s.shape}. Expected 4 dimensions.")

    batch_size, seq_len_padded, effective_m, num_features = s.shape
    # Determine num_classes dynamically if possible, fallback to constant/config
    num_edge_classes = num_features if use_edge_features else 1 # Or pass NUM_EDGE_FEATURES

    # Get optional data
    node_type_labels = data.get('node_types')
    if predict_node_types and node_type_labels is not None:
        node_type_labels = node_type_labels.long().to(device)

    truth_table = data.get('y') # Should be None if not conditioning
    if use_conditioning and truth_table is not None:
        truth_table = truth_table.float().to(device)
        if len(truth_table.shape) > 2: # Flatten if needed
            truth_table = truth_table.view(truth_table.shape[0], -1) # Use view for efficiency
    else:
        truth_table = None # Ensure it's None

    # --- Prepare Inputs/Targets with SOS/EOS ---
    # SOS/EOS frames should match dimensions of s
    one_frame = torch.ones([batch_size, 1, effective_m, num_features], device=device)
    zero_frame = torch.zeros([batch_size, 1, effective_m, num_features], device=device)

    # x: Input sequence for GraphRNN (SOS + s)
    # y: Target sequence for EdgeMLP (s + EOS)
    x = torch.cat((one_frame, s), dim=1) # Shape: [batch, seq_len_padded+1, effective_m, features]
    y = torch.cat((s, zero_frame), dim=1) # Shape: [batch, seq_len_padded+1, effective_m, features]

    lens_with_sos = lens + 1

    # --- NEW: Prepare Levels tensor for GraphRNN input ---
    levels_for_rnn = None
    if levels_padded is not None:
        # levels_padded corresponds to nodes 1 to max_n-1
        # We need levels for input steps 0 (SOS) to max_n-1
        sos_level = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # Assign level 0 to SOS token
        # Slice padded levels to match max sequence length `seq_len_padded` = max_n-1
        levels_sliced = levels_padded[:, :seq_len_padded]  # Shape: [batch, max_n-1]
        levels_for_rnn = torch.cat((sos_level, levels_sliced), dim=1)  # Shape: [batch, max_n]
        # Ensure shape matches x's sequence dim
        if levels_for_rnn.shape[1] != x.shape[1]:
            print(
                f"Warning: Level tensor seq length ({levels_for_rnn.shape[1]}) mismatch with input x seq length ({x.shape[1]}). Adjusting.")
            # Adjust padding/slicing if necessary, though above logic should work if max_n is consistent
            target_len = x.shape[1]
            levels_for_rnn = torch.cat((sos_level, levels_padded[:, :target_len - 1]), dim=1)
            if levels_for_rnn.shape[1] != target_len:  # Check again after adjustment
                print("ERROR: Cannot align level tensor length. Disabling level embedding for this batch.")
                levels_for_rnn = None  # Disable if alignment fails
    # --- End NEW ---

    # --- GraphRNN Forward Pass ---
    graph_rnn.reset_hidden()
    output = graph_rnn(x, lens_with_sos, truth_table=truth_table, levels=levels_for_rnn)

    # Unpack output
    node_type_logits = None
    if predict_node_types and isinstance(output, tuple):
        hidden, node_type_logits = output
    else:
        hidden = output
    # hidden shape: [batch, seq_len_padded+1, hidden_size_rnn]

    # --- EdgeMLP Forward Pass ---
    # EdgeMLP predicts edge types based on hidden state
    # return_logits=True is needed for CrossEntropyLoss
    y_pred = edge_mlp(hidden, return_logits=True,
                      truth_table=truth_table if use_conditioning else None)
    # y_pred shape: [batch, seq_len_padded+1, effective_m, num_edge_classes]

    # --- Edge Loss Calculation ---
    edge_loss = torch.tensor(0.0, device=device) # Initialize loss

    # Pack target sequence 'y' based on actual lengths (lens_with_sos)
    try:
        # Pack based on actual lengths
        y_packed = pack_padded_sequence(y, lens_with_sos, batch_first=True, enforce_sorted=False)
        # Pad back to max length *in this batch* for comparison (needed for indexing/masking)
        # Note: y_padded's seq len dim might be shorter than y_pred's if last batch is short
        y_padded, _ = pad_packed_sequence(y_packed, batch_first=True, total_length=y.shape[1])
        # y_padded shape: [batch, seq_len_padded+1, effective_m, num_features]

    except RuntimeError as e:
         print(f"Error during pack/pad_packed_sequence for y: {e}")
         print(f"y shape: {y.shape}, lens_with_sos: {lens_with_sos}")
         # Handle error: maybe skip batch or return current losses
         return {'total': 0.0, 'edge': 0.0, 'node': 0.0}


    # Check shapes after padding
    # assert y_pred.shape == y_padded.shape, f"Shape mismatch: y_pred {y_pred.shape} vs y_padded {y_padded.shape}"

    if use_edge_features: # Multi-class edge prediction (AIG case)
        # CrossEntropyLoss expects [N, C] predictions and [N] labels
        pred_batch, pred_seq, pred_m, pred_classes = y_pred.shape
        target_batch, target_seq, target_m, target_features = y_padded.shape

        # Ensure dimensions match before proceeding
        if pred_batch != target_batch or pred_seq != target_seq or pred_m != target_m:
             print(f"Warning: Dimension mismatch between y_pred ({y_pred.shape}) and y_padded ({y_padded.shape}). Skipping edge loss.")
        else:
            num_classes = pred_classes # Num classes from prediction tensor
            # Reshape predictions: Flatten batch, seq, and predecessor dims
            # Shape: [Batch * Seq * M, NumClasses]
            y_pred_flat = y_pred.reshape(-1, num_classes)

            # Convert targets (one-hot) to labels (class indices)
            # Shape: [Batch, Seq, M] -> [Batch * Seq * M]
            try:
                y_labels = torch.argmax(y_padded, dim=-1) # Convert one-hot targets to class indices
                y_labels_flat = y_labels.reshape(-1)
            except RuntimeError as e:
                print(f"Error during torch.argmax or reshape for y_labels: {e}")
                print(f"y_padded shape: {y_padded.shape}")
                y_labels_flat = None # Mark as error

            if y_labels_flat is not None:
                # --- ADD THIS LINE ---
                # Move lengths to GPU for mask comparison
                lens_with_sos_gpu = lens_with_sos.to(device)
                # --- MODIFY THIS LINE (use lens_with_sos_gpu) ---
                mask = torch.arange(pred_seq, device=device)[None, :] < lens_with_sos_gpu[:, None]  # [Batch, Seq]
                mask = mask.unsqueeze(-1).expand(-1, -1, effective_m) # [Batch, Seq, M]
                mask_flat = mask.reshape(-1) # [Batch * Seq * M]

                # Ensure flattened shapes match before applying mask
                if y_pred_flat.shape[0] == y_labels_flat.shape[0] == mask_flat.shape[0]:
                    # Apply criterion only to valid (non-padded) elements indicated by mask
                    valid_preds = y_pred_flat[mask_flat]
                    valid_labels = y_labels_flat[mask_flat]

                    if valid_labels.numel() > 0: # Check if there are any valid elements
                         edge_loss = criterion_edge(valid_preds, valid_labels.long())
                else:
                    print(f"Warning: Flattened shape mismatch."
                          f" Preds: {y_pred_flat.shape[0]}, Labels: {y_labels_flat.shape[0]}, Mask: {mask_flat.shape[0]}. Skipping edge loss.")

    else: # Binary edge prediction (Original BCELoss path - less likely for AIG)
        # Requires sigmoid activation in MLP if using BCELoss, or use BCEWithLogitsLoss
        print("Warning: Using binary edge loss path. Ensure model/loss are configured correctly.")
        mask = torch.arange(y_pred.shape[1], device=device)[None, :] < lens_with_sos[:, None] # [Batch, Seq]
        # Expand mask to match y_pred/y_padded shape [Batch, Seq, M, Features=1]
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(y_pred)
        # Apply loss only where mask is True
        if torch.any(mask):
             edge_loss = criterion_edge(y_pred[mask], y_padded[mask])


    # --- Node Loss Calculation (Conditional) ---
    node_loss = torch.tensor(0.0, device=device) # Initialize loss
    if predict_node_types and criterion_node is not None and node_type_logits is not None and node_type_labels is not None:
        # node_type_logits shape: [batch, seq_len+1, num_node_types]
        # node_type_labels shape: [batch, orig_seq_len_padded] (padded to max_node_count-1)

        batch_size_n, seq_len_plus_1_n, num_node_classes = node_type_logits.shape
        # Labels correspond to nodes 1 to n, logits correspond to steps 0 to n (incl SOS)

        # Create mask based on lens (actual lengths, excluding SOS position for labels)
        # Logit mask considers SOS, label mask does not
        logit_mask = torch.arange(seq_len_plus_1_n, device=device)[None, :] < lens_with_sos[:, None] # [batch, seq+1]

        # Align labels: Padded labels from dataset might already be correct shape
        # Target labels correspond to logits[1:]
        # Ensure labels are padded to same seq_len+1 dim, e.g., with ignore_index
        aligned_labels = torch.full((batch_size_n, seq_len_plus_1_n), criterion_node.ignore_index, dtype=torch.long, device=device)
        for i in range(batch_size_n):
             actual_len = lens[i].item() # Original sequence length
             # Ensure we don't index out of bounds of provided labels
             len_to_copy = min(actual_len, node_type_labels.shape[1])
             if len_to_copy > 0:
                  # Place labels at indices 1 to actual_len+1 in aligned_labels
                  aligned_labels[i, 1 : actual_len + 1] = node_type_labels[i, :len_to_copy]

        # Flatten logits and labels WHERE MASK IS TRUE
        valid_logits = node_type_logits[logit_mask] # [NumValidLogitSteps, C]
        valid_labels = aligned_labels[logit_mask]   # [NumValidLogitSteps]

        # Compute loss, ignoring SOS position implicitly via ignore_index in aligned_labels[0]
        if valid_labels.numel() > 0:
            # Ensure labels are long type
            node_loss = criterion_node(valid_logits, valid_labels.long())


    # --- Combine Losses and Backpropagate ---
    total_loss = edge_loss + node_loss

    # Prevent backprop if loss is invalid (e.g., NaN)
    if torch.isnan(total_loss) or torch.isinf(total_loss):
         print(f"Warning: Invalid loss detected (NaN or Inf): {total_loss.item()}. Skipping backpropagation for this step.")
         # Optionally clear gradients here if optimizers were used before check
         # optim_graph_rnn.zero_grad()
         # optim_edge_mlp.zero_grad()
    elif total_loss.requires_grad: # Check if loss requires grad (might be 0.0 if batch was skipped)
        total_loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(graph_rnn.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(edge_mlp.parameters(), max_norm=1.0)
        optim_graph_rnn.step()
        optim_edge_mlp.step()

    # Step schedulers AFTER optimizer steps
    scheduler_graph_rnn.step()
    scheduler_mlp.step()

    # Return loss dictionary
    return {
        'total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        'edge': edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
        'node': node_loss.item() if isinstance(node_loss, torch.Tensor) else node_loss
    }


def train_rnn_step(graph_rnn, edge_rnn, data,
                   criterion_edge, criterion_node,
                   optim_graph_rnn, optim_edge_rnn,
                   scheduler_graph_rnn, scheduler_edge_rnn,
                   device, use_edge_features,
                   predict_node_types, use_conditioning):
    graph_rnn.zero_grad()
    edge_rnn.zero_grad()

    # Get mandatory data & NEW: levels
    seq, lens = data['x'].float().to(device), data['len'].cpu()
    levels_padded = data.get('levels') # Get levels tensor
    if levels_padded is not None:
        levels_padded = levels_padded.long().to(device) # Shape: [batch, max_n-1]
    # --- End NEW ---


    # --- Basic Shape Checks and Setup ---
    if seq.numel() == 0 or lens.numel() == 0 or torch.all(lens == 0):
         print("Warning: Received empty batch or batch with all zero lengths. Skipping step.")
         return {'total': 0.0, 'edge': 0.0, 'node': 0.0}

    # Add feature dim if missing
    if len(seq.shape) == 3:
        print("Warning: Input tensor 'seq' has 3 dims, expected 4. Unsqueezing last dim.")
        seq = seq.unsqueeze(3)
    elif len(seq.shape) != 4:
         raise ValueError(f"Input tensor 'seq' has unexpected shape: {seq.shape}. Expected 4 dimensions.")

    batch_size, seq_len_padded, effective_m, num_features = seq.shape
    num_edge_classes = num_features if use_edge_features else 1

    # Get optional data
    node_type_labels = data.get('node_types')
    if predict_node_types and node_type_labels is not None:
        node_type_labels = node_type_labels.long().to(device)

    truth_table = data.get('y')
    if use_conditioning and truth_table is not None:
        truth_table = truth_table.float().to(device)
        if len(truth_table.shape) > 2:
            truth_table = truth_table.view(truth_table.shape[0], -1)
    else:
        truth_table = None

    # --- Node RNN Input Prep ---
    # Input for node RNN uses sequence up to T-1 relative to 'lens'
    # SOS token needs same effective_m and features dimensions
    one_frame_node = torch.ones([batch_size, 1, effective_m, num_features], device=device)
    # We need seq[:, :max(lens)-1, :, :] potentially? No, lens handles it.
    # Use seq directly, lengths `lens` will handle the effective length.
    x_node_rnn = torch.cat((one_frame_node, seq), dim=1)[:, :-1, :, :] # SOS + seq[0...n-1] -> feed lens+1? Check GraphRNN forward
    # Let's try passing SOS + seq (full), and lens+1 to GraphRNN
    x_node_rnn_full = torch.cat((one_frame_node, seq), dim=1) # Shape: [batch, seq_len_padded+1, effective_m, features]
    lens_node_rnn = lens + 1 # Lengths including SOS

    # --- NEW: Prepare Levels tensor for GraphRNN input ---
    # (Identical logic to train_mlp_step)
    levels_for_rnn = None
    if levels_padded is not None:
        sos_level = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        levels_sliced = levels_padded[:, :seq_len_padded]  # Shape: [batch, max_n-1]
        levels_for_rnn = torch.cat((sos_level, levels_sliced), dim=1)  # Shape: [batch, max_n]
        if levels_for_rnn.shape[1] != x_node_rnn_full.shape[1]:
            print(
                f"Warning: Level tensor seq length ({levels_for_rnn.shape[1]}) mismatch with input x seq length ({x_node_rnn_full.shape[1]}). Adjusting.")
            target_len = x_node_rnn_full.shape[1]
            levels_for_rnn = torch.cat((sos_level, levels_padded[:, :target_len - 1]), dim=1)
            if levels_for_rnn.shape[1] != target_len:
                print("ERROR: Cannot align level tensor length. Disabling level embedding for this batch.")
                levels_for_rnn = None

    # --- GraphRNN Forward Pass ---
    graph_rnn.reset_hidden()
    output = graph_rnn(x_node_rnn_full, lens_node_rnn,
                       truth_table=truth_table, levels=levels_for_rnn)

    # Unpack output
    hidden = None
    node_type_logits = None
    if predict_node_types and isinstance(output, tuple):
        hidden, node_type_logits = output
    else:
        hidden = output
    # hidden shape: [batch, seq_len_padded+1, hidden_size_rnn]
    # We need the hidden states corresponding to the *actual* sequence lengths (nodes 0 to n-1)
    # This hidden state corresponds to inputs SOS, node 0, ..., node n-1

    # --- Edge RNN Processing ---

    # Pack the hidden states corresponding to actual nodes (indices 1 to n in hidden)
    # and the target sequence 'seq' (nodes 1 to n) based on original 'lens'
    try:
        # Pack hidden states (need H for nodes 0 to n-1, which are hidden[:, 1:, :])
        # Let's pack the hidden state corresponding to input node 0...n-1, which is hidden output steps 1..n
        hidden_for_edges_packed = pack_padded_sequence(hidden[:, 1:, :], lens, batch_first=True, enforce_sorted=False)
        hidden_packed_data = hidden_for_edges_packed.data # Shape [TotalNodes, HiddenSize]

        # Pack the target edge sequence 'seq' based on 'lens'
        seq_packed_obj = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
        # Target edge data (y_edge_rnn)
        # Shape [TotalNodes, effective_m, EdgeFeatures]
        y_edge_rnn = seq_packed_obj.data
    except RuntimeError as e:
         print(f"Error during pack_padded_sequence for hidden/seq: {e}")
         print(f"hidden shape: {hidden.shape}, seq shape: {seq.shape}, lens: {lens}")
         return {'total': 0.0, 'edge': 0.0, 'node': 0.0}


    # Prepare EdgeRNN input sequence (x_edge_rnn) = SOS + y_edge_rnn shifted
    # SOS needs shape [TotalNodes, 1, EdgeFeatures]
    total_nodes = y_edge_rnn.shape[0]
    if total_nodes == 0: # Handle case where packing resulted in zero nodes
        print("Warning: Zero total nodes after packing. Skipping edge RNN step.")
        edge_loss = torch.tensor(0.0, device=device)
    else:
        sos_edge = torch.ones(total_nodes, 1, num_features, device=device)
        # Input sequence for EdgeRNN
        # Shape [TotalNodes, effective_m, EdgeFeatures]
        x_edge_rnn = torch.cat([sos_edge, y_edge_rnn[:, :-1, :]], dim=1)

        # --- Calculate Edge Sequence Lengths ---
        # Length for node `node_idx` (0-based index in packed seq) is min(node_idx + 1, effective_m)
        # We need the original index within each batch item for this.
        # This mapping is complex. Let's simplify:
        # The length of the predecessor sequence for node `i` (1-based index in original graph)
        # is `min(i, effective_m)`. We need this length for each packed node.
        # Let's try getting lengths directly related to `lens`.
        edge_seq_lens = []
        for length_in_batch in lens.tolist(): # Iterate through original lengths
             for i in range(1, length_in_batch + 1): # For nodes 1 to n for this graph
                  # Length of edge vector for node i is number of predecessors to predict
                  edge_seq_lens.append(min(i, effective_m)) # Use effective_m here
        edge_seq_lens_tensor = torch.tensor(edge_seq_lens, dtype=torch.long, device=device)

        # Ensure packed lengths match TotalNodes
        if len(edge_seq_lens) != total_nodes:
             print(f"Warning: Mismatch between calculated edge_seq_lens ({len(edge_seq_lens)}) and total_nodes ({total_nodes}).")
             # Fallback or error handling needed
             # As fallback, create lens based on target shape if possible
             edge_seq_lens_tensor = torch.clamp(torch.tensor(edge_seq_lens[:total_nodes], device=device), max=effective_m)


        # --- Conditional Truth Table for Edge RNN ---
        edge_truth_table = None
        if use_conditioning and truth_table is not None:
            # Repeat graph-level TT for each node in the packed sequence
            repeated_tt = []
            # Use indices from packed hidden state (should match seq packing)
            original_indices = hidden_for_edges_packed.sorted_indices
            if original_indices is None: original_indices = torch.arange(batch_size) # Assume original order

            current_pos = 0
            batch_sizes = hidden_for_edges_packed.batch_sizes
            for i in range(len(batch_sizes)):
                batch_size_at_step = batch_sizes[i]
                original_batch_indices_at_step = original_indices[current_pos : current_pos + batch_size_at_step]
                repeated_tt.append(truth_table[original_batch_indices_at_step])
                current_pos += batch_size_at_step
            if repeated_tt:
                 edge_truth_table = torch.cat(repeated_tt, dim=0) # Shape [TotalNodes, tt_size]
            # else:
            #      edge_truth_table = torch.empty((0, tt_size), device=device)

            if edge_truth_table.shape[0] != total_nodes:
                print(f"Warning: Truth table repetition mismatch: expected {total_nodes}, got {edge_truth_table.shape[0]}.")
                edge_truth_table = None # Disable if mismatch occurs


        # --- Set Hidden State & Run Edge RNN Forward Pass ---
        edge_rnn.set_first_layer_hidden(hidden_packed_data)

        # Pass edge_truth_table conditionally, return logits
        y_edge_rnn_pred = edge_rnn(
            x_edge_rnn,
            x_lens=edge_seq_lens_tensor.cpu(), # pack_padded needs cpu lengths
            return_logits=True,
            truth_table=edge_truth_table if use_conditioning else None
        ) # Output shape: [TotalNodes, max(edge_seq_lens), EdgeFeatures]

        # --- Edge Loss Calculation ---
        edge_loss = torch.tensor(0.0, device=device)

        # Pad the target edge sequence (y_edge_rnn) based on edge_seq_lens
        # Pad to the same max length as y_edge_rnn_pred for comparison
        max_pred_len = y_edge_rnn_pred.shape[1]
        try:
             y_edge_rnn_packed_obj = pack_padded_sequence(y_edge_rnn, edge_seq_lens_tensor, batch_first=True, enforce_sorted=False)
             # Pad target to match max length of prediction sequences
             y_edge_rnn_padded, _ = pad_packed_sequence(y_edge_rnn_packed_obj, batch_first=True, total_length=max_pred_len)
             # y_edge_rnn_padded shape: [TotalNodes, max_pred_len, EdgeFeatures]
        except RuntimeError as e:
             print(f"Error packing/padding y_edge_rnn: {e}")
             y_edge_rnn_padded = None # Mark as error

        if y_edge_rnn_padded is not None:
            # Create mask based on edge_seq_lens and max_pred_len
            edge_mask = torch.arange(max_pred_len, device=device)[None, :] < edge_seq_lens_tensor[:, None] # [TotalNodes, max_pred_len]

            if use_edge_features: # Multi-class
                 pred_nodes, pred_len, pred_classes = y_edge_rnn_pred.shape
                 target_nodes, target_len, target_features = y_edge_rnn_padded.shape

                 if pred_nodes != target_nodes or pred_len != target_len:
                      print(f"Warning: Shape mismatch pred ({y_edge_rnn_pred.shape}) vs target ({y_edge_rnn_padded.shape}). Skipping edge loss.")
                 else:
                      num_classes = pred_classes
                      # Flatten predictions: [TotalNodes * max_pred_len, NumClasses]
                      y_pred_flat = y_edge_rnn_pred.reshape(-1, num_classes)
                      # Convert targets to labels: [TotalNodes * max_pred_len]
                      try:
                           y_labels_flat = torch.argmax(y_edge_rnn_padded, dim=-1).reshape(-1)
                      except RuntimeError as e:
                            print(f"Error argmax/reshape y_edge_rnn_padded: {e}")
                            y_labels_flat = None

                      if y_labels_flat is not None:
                           # Flatten mask: [TotalNodes * max_pred_len]
                           mask_flat = edge_mask.reshape(-1)

                           if y_pred_flat.shape[0] == y_labels_flat.shape[0] == mask_flat.shape[0]:
                                valid_preds = y_pred_flat[mask_flat]
                                valid_labels = y_labels_flat[mask_flat]
                                if valid_labels.numel() > 0:
                                     edge_loss = criterion_edge(valid_preds, valid_labels.long())
                           else:
                                print("Warning: Flattened shape mismatch in RNN edge loss. Skipping.")
            else: # Binary
                 print("Warning: Using binary edge loss path for RNN edge model.")
                 # Expand mask to match 3D shape [Nodes, Len, Features=1]
                 mask_expanded = edge_mask.unsqueeze(-1).expand_as(y_edge_rnn_pred)
                 if torch.any(mask_expanded):
                      edge_loss = criterion_edge(y_edge_rnn_pred[mask_expanded], y_edge_rnn_padded[mask_expanded])

    # --- Node Loss Calculation (Conditional) ---
    # Node loss calculation is the same as in MLP step
    node_loss = torch.tensor(0.0, device=device)
    if predict_node_types and criterion_node is not None and node_type_logits is not None and node_type_labels is not None:
        # node_type_logits shape: [batch, seq_len_node_rnn, num_node_types] (seq_len matches lens+1)
        batch_size_n, seq_len_plus_1_n, num_node_classes = node_type_logits.shape

        # Mask based on lens_node_rnn (lens+1)
        logit_mask = torch.arange(seq_len_plus_1_n, device=device)[None, :] < lens_node_rnn[:, None] # [batch, seq+1]

        # Align labels - same logic as MLP step
        aligned_labels = torch.full((batch_size_n, seq_len_plus_1_n), criterion_node.ignore_index, dtype=torch.long, device=device)
        for i in range(batch_size_n):
             actual_len = lens[i].item() # Original sequence length
             len_to_copy = min(actual_len, node_type_labels.shape[1])
             if len_to_copy > 0:
                  aligned_labels[i, 1 : actual_len + 1] = node_type_labels[i, :len_to_copy]

        valid_logits = node_type_logits[logit_mask]
        valid_labels = aligned_labels[logit_mask]

        if valid_labels.numel() > 0:
            node_loss = criterion_node(valid_logits, valid_labels.long())


    # --- Combine Losses and Backpropagate ---
    total_loss = edge_loss + node_loss

    if torch.isnan(total_loss) or torch.isinf(total_loss):
         print(f"Warning: Invalid loss detected (NaN or Inf): {total_loss.item()}. Skipping backpropagation.")
    elif total_loss.requires_grad:
        total_loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(graph_rnn.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(edge_rnn.parameters(), max_norm=1.0)
        optim_graph_rnn.step()
        optim_edge_rnn.step() # Use edge RNN optimizer

    # Step schedulers
    scheduler_graph_rnn.step()
    scheduler_edge_rnn.step() # Use edge RNN scheduler

    return {
        'total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        'edge': edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
        'node': node_loss.item() if isinstance(node_loss, torch.Tensor) else node_loss
    }