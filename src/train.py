
import numpy as np # Needed for debugging print if used
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence



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


# src/train.py (Partial - Only showing updated train_rnn_step function)

import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
import numpy as np


def train_rnn_step(graph_rnn, edge_rnn, data,
                   criterion_edge,
                   optim_graph_rnn,
                   optim_edge_rnn,
                   scheduler_graph_rnn,
                   scheduler_edge_rnn,
                   device, use_edge_features):
    """ Train GraphRNN with Attention RNN edge model. """
    graph_rnn.zero_grad()
    edge_rnn.zero_grad()

    # --- Data Preparation ---
    seq, lens_cpu = data['x'].float().to(device), data['len'].cpu()  # Target edge sequences
    batch_size, seq_len_padded, effective_m, num_features = seq.shape

    levels_padded = data.get('levels')
    if levels_padded is not None:
        levels_padded = levels_padded.long().to(device)

    if len(seq.shape) == 3:  # Add feature dim if missing
        seq = seq.unsqueeze(3)
        num_features = 1

    # Node RNN inputs (SOS + Target Sequence `seq`)
    one_frame_node = torch.ones([batch_size, 1, effective_m, num_features], device=device)
    x_node_rnn_input = torch.cat((one_frame_node, seq), dim=1)  # Shape: [B, MaxSeq+1, M, F]
    lens_node_rnn = lens_cpu + 1  # Lengths including SOS

    # Prepare Levels tensor for Node RNN
    levels_for_node_rnn = None
    if levels_padded is not None and hasattr(graph_rnn, 'level_embedding') and graph_rnn.level_embedding is not None:
        sos_level = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        target_level_len = seq_len_padded
        levels_sliced = levels_padded[:, :target_level_len]
        levels_for_node_rnn = torch.cat((sos_level, levels_sliced), dim=1)

        if levels_for_node_rnn.shape[1] != x_node_rnn_input.shape[1]:
            target_len = x_node_rnn_input.shape[1]
            levels_for_node_rnn = torch.cat((sos_level, levels_padded[:, :target_len - 1]), dim=1)
            if levels_for_node_rnn.shape[1] != target_len:
                print("ERROR: Cannot align level tensor length. Disabling level embedding.")
                levels_for_node_rnn = None

    # --- 1. Run GraphLevelRNN to get all node hidden states ---
    graph_rnn.reset_hidden()
    node_hiddens_padded = graph_rnn(x_node_rnn_input, lens_node_rnn,
                                    levels=levels_for_node_rnn,
                                    return_all_hiddens=True)

    # --- 2. Prepare inputs for EdgeLevelRNN ---
    try:
        seq_packed_obj = pack_padded_sequence(seq, lens_cpu, batch_first=True, enforce_sorted=False)
        y_edge_target_packed = seq_packed_obj
    except RuntimeError as e:
        print(f"Error packing target sequence 'seq': {e}")
        return {'total': 0.0, 'edge': 0.0}

    total_steps_edges = sum(lens_cpu).item()  # Total number of actual edge steps across the batch
    if total_steps_edges == 0:
        print("Warning: Zero total edge steps. Skipping loss calculation.")
        return {'total': 0.0, 'edge': 0.0}

    seq_padded_for_shift, _ = pad_packed_sequence(y_edge_target_packed, batch_first=True)
    x_edge_input_padded = torch.cat((
        torch.ones(batch_size, 1, effective_m, num_features, device=device),
        seq_padded_for_shift[:, :-1, :, :]
    ), dim=1)
    try:
        x_edge_input_packed = pack_padded_sequence(x_edge_input_padded, lens_cpu, batch_first=True,
                                                   enforce_sorted=False)
    except RuntimeError as e:
        print(f"Error packing input sequence 'x_edge_input_padded': {e}")
        return {'total': 0.0, 'edge': 0.0}

    # --- 3. Prepare `prev_node_hiddens` AND `attn_mask` for Attention ---
    # FIXED: Correctly collect node histories across packed sequence
    prev_node_hiddens_list = []
    valid_history_lengths = []

    # Need to track original batch indices for each entry in the packed sequence
    if hasattr(x_edge_input_packed, 'unsorted_indices') and x_edge_input_packed.unsorted_indices is not None:
        unsorted_indices = x_edge_input_packed.unsorted_indices
    else:
        unsorted_indices = torch.arange(batch_size, device=device)

    # Initialize accumulated indices for each batch item
    batch_item_positions = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Process each batch size in the packed sequence
    offset = 0
    for batch_size_at_step in x_edge_input_packed.batch_sizes:
        batch_size_at_step = batch_size_at_step.item()

        # For each active batch item at this step
        for i in range(batch_size_at_step):
            # Get the original batch index for this item
            if hasattr(x_edge_input_packed, 'sorted_indices') and x_edge_input_packed.sorted_indices is not None:
                orig_batch_idx = x_edge_input_packed.sorted_indices[i].item()
            else:
                orig_batch_idx = i

            # Get the current position (time step) for this batch item
            current_pos = batch_item_positions[orig_batch_idx].item()

            # For attention, we need node hidden states up to and including the current node
            # +1 to include SOS node state
            num_prev_nodes = current_pos + 1 + 1  # position + 1 (for current node) + 1 (for SOS)

            # Get the node hidden states for all previous nodes for this batch item
            if num_prev_nodes <= node_hiddens_padded.shape[1]:
                history = node_hiddens_padded[orig_batch_idx, :num_prev_nodes, :]
                prev_node_hiddens_list.append(history)
                valid_history_lengths.append(num_prev_nodes)
            else:
                # This shouldn't happen with proper indexing
                print(
                    f"Warning: History length {num_prev_nodes} exceeds available node hiddens {node_hiddens_padded.shape[1]}")
                # Use all available history
                history = node_hiddens_padded[orig_batch_idx, :, :]
                prev_node_hiddens_list.append(history)
                valid_history_lengths.append(history.shape[0])

            # Update position for this batch item
            batch_item_positions[orig_batch_idx] += 1

        # Move offset for the next batch size
        offset += batch_size_at_step

    # Verify we have the correct number of histories
    if len(prev_node_hiddens_list) != total_steps_edges:
        print(
            f"Warning: History list length ({len(prev_node_hiddens_list)}) vs total edge steps ({total_steps_edges}).")
        total_steps_edges = len(prev_node_hiddens_list)  # Adjust to match what we have

    # Prepare padded tensors for attention
    max_prev_nodes = max(valid_history_lengths) if valid_history_lengths else 0
    node_hidden_size = node_hiddens_padded.shape[-1]

    padded_prev_node_hiddens = torch.zeros(
        total_steps_edges, max_prev_nodes, node_hidden_size, device=device)

    # Populate the padded tensor with actual histories
    for i, hist in enumerate(prev_node_hiddens_list):
        if i < total_steps_edges:  # Safety check
            actual_len = min(hist.shape[0], max_prev_nodes)
            padded_prev_node_hiddens[i, :actual_len, :] = hist[:actual_len, :]

    # Create attention mask: True = masked positions (to be ignored)
    attn_mask = torch.ones(total_steps_edges, max_prev_nodes, dtype=torch.bool, device=device)
    for i, length in enumerate(valid_history_lengths):
        if i < total_steps_edges:  # Safety check
            attn_mask[i, :min(length, max_prev_nodes)] = False

    # --- 4. Run EdgeLevelRNN with Attention ---
    try:
        # Set initial hidden state for EdgeRNN from the GraphRNN output
        hidden_for_edges_padded = node_hiddens_padded[:, 1:, :]  # Skip SOS
        hidden_for_edges_packed = pack_padded_sequence(hidden_for_edges_padded, lens_cpu, batch_first=True,
                                                       enforce_sorted=False)
        edge_rnn_init_hidden_packed = hidden_for_edges_packed.data
    except Exception as e:
        print(f"Error preparing/packing initial hidden states for EdgeRNN: {e}")
        return {'total': 0.0, 'edge': 0.0}

    # Set first layer hidden state for EdgeRNN
    edge_rnn.set_first_layer_hidden(edge_rnn_init_hidden_packed)

    # Forward pass through EdgeRNN
    y_edge_pred_packed_data = edge_rnn(
        x=x_edge_input_packed.data.view(-1, 1, effective_m, num_features),  # Reshape to match expected input
        prev_node_hiddens=padded_prev_node_hiddens,
        attn_mask=attn_mask,
        x_lens=None,
        return_logits=use_edge_features
    ).view(-1, effective_m, num_features)

    # Repack the output into a PackedSequence
    y_edge_pred_packed = torch.nn.utils.rnn.PackedSequence(
        data=y_edge_pred_packed_data,
        batch_sizes=x_edge_input_packed.batch_sizes,
        sorted_indices=x_edge_input_packed.sorted_indices,
        unsorted_indices=x_edge_input_packed.unsorted_indices
    )

    # --- 5. Calculate Loss ---
    edge_loss = torch.tensor(0.0, device=device)
    pred_padded, _ = pad_packed_sequence(y_edge_pred_packed, batch_first=True)
    target_padded, lens_orig = pad_packed_sequence(y_edge_target_packed, batch_first=True)

    if pred_padded.shape != target_padded.shape:
        print(
            f"Warning: Shape mismatch after unpacking pred ({pred_padded.shape}) vs target ({target_padded.shape}). Skipping loss.")
    else:
        max_len_in_batch = pred_padded.shape[1]
        loss_mask = torch.arange(max_len_in_batch, device=device)[None, :] < lens_orig.to(device)[:, None]

        if use_edge_features:  # CrossEntropyLoss
            num_classes = pred_padded.shape[-1]
            pred_flat = pred_padded.reshape(-1, num_classes)
            try:
                labels_flat = torch.argmax(target_padded, dim=-1).reshape(-1)
            except RuntimeError as e:
                print(f"Error argmax/reshape target_padded: {e}")
                labels_flat = None

            if labels_flat is not None:
                mask_flat = loss_mask.reshape(-1)
                if pred_flat.shape[0] == labels_flat.shape[0] == mask_flat.shape[0]:
                    valid_preds = pred_flat[mask_flat]
                    valid_labels = labels_flat[mask_flat]
                    if valid_labels.numel() > 0:
                        edge_loss = criterion_edge(valid_preds, valid_labels.long())
                    else:
                        print("Warning: No valid elements after masking for edge loss.")
                else:
                    print("Warning: Flattened shape mismatch in RNN attention loss. Skipping.")

        else:  # BCELoss
            print("Warning: Using binary edge loss path for RNN step.")
            mask_expanded = loss_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_padded)
            if torch.any(mask_expanded):
                edge_loss = criterion_edge(pred_padded[mask_expanded], target_padded[mask_expanded])

    # --- Backpropagation ---
    total_loss = edge_loss

    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"Warning: Invalid loss detected (NaN or Inf): {total_loss.item()}. Skipping backpropagation.")
    elif total_loss.requires_grad and total_loss > 0:
        total_loss.backward()
        optim_graph_rnn.step()
        optim_edge_rnn.step()

    scheduler_graph_rnn.step()
    scheduler_edge_rnn.step()

    return {
        'total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        'edge': edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
        'node': 0.0  # Placeholder
    }