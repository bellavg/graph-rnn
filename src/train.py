import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np # Needed for debugging print if used
import torch.nn.functional as F # Added for loss calculation

# Assume NUM_EDGE_FEATURES is available, e.g., from aig_dataset or config
# If not, determine dynamically from tensor shapes where possible.

# --- train_mlp_step remains unchanged for now ---
def train_mlp_step(graph_rnn, edge_mlp, data,
                   criterion_edge,
                   optim_graph_rnn, optim_edge_mlp,
                   scheduler_graph_rnn, scheduler_mlp, device, use_edge_features):
    """ Train GraphRNN with MLP edge model (Level Embedding Added). """
    # --- THIS FUNCTION IS NOT MODIFIED FOR NODE LOSS YET ---
    # --- Modifications would be needed if using MLP with node prediction ---
    graph_rnn.zero_grad()
    edge_mlp.zero_grad()

    s, lens_cpu = data['x'].float().to(device), data['len'].cpu()
    batch_size, seq_len_padded, effective_m, num_features = s.shape

    # --- Get and prepare levels tensor ---
    levels_padded = data.get('levels')
    if levels_padded is not None:
        levels_padded = levels_padded.long().to(device)
    # --- End ---

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
    # --- End ---

    # --- GraphRNN Forward Pass ---
    # NOTE: If graph_rnn.predict_node_types is True, this call needs modification
    # to handle two outputs (hidden, node_type_logits)
    graph_rnn.reset_hidden()
    hidden = graph_rnn(x, lens_with_sos, levels=levels_for_rnn)
    # --- End ---

    # --- Node Loss Calculation (Placeholder - Requires modifications if enabled) ---
    node_loss = torch.tensor(0.0, device=device)
    # if graph_rnn.predict_node_types:
    #     # 1. Get node_type_logits from graph_rnn output
    #     # 2. Get y_node_type from data
    #     # 3. Calculate masked node_loss using criterion_node
    #     pass # Placeholder
    # --- End Node Loss ---


    # --- EdgeMLP Forward Pass ---
    y_pred = edge_mlp(hidden, return_logits=use_edge_features)

    # --- Edge Loss Calculation ---
    edge_loss = torch.tensor(0.0, device=device)
    try:
        y_packed = pack_padded_sequence(y, lens_with_sos.cpu(), batch_first=True, enforce_sorted=False)
        y_padded, _ = pad_packed_sequence(y_packed, batch_first=True, total_length=y.shape[1])
    except RuntimeError as e:
         print(f"Error during pack/pad_packed_sequence for y in train_mlp_step: {e}")
         return {'total': 0.0, 'edge': 0.0, 'node': 0.0}

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

    # Total loss (needs update if node_loss is added)
    total_loss = edge_loss # + node_loss_weight * node_loss

    # Backpropagate
    if torch.isnan(total_loss) or torch.isinf(total_loss):
         print(f"Warning: Invalid loss detected (NaN or Inf): {total_loss.item()}. Skipping backpropagation.")
    elif total_loss.requires_grad and total_loss > 0:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(graph_rnn.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(edge_mlp.parameters(), max_norm=1.0)
        optim_graph_rnn.step()
        optim_edge_mlp.step()

    # Step schedulers
    scheduler_graph_rnn.step()
    scheduler_mlp.step()

    edge_loss_val = edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss
    node_loss_val = node_loss.item() if isinstance(node_loss, torch.Tensor) else node_loss # Placeholder
    total_loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

    return {
        'total': total_loss_val,
        'edge': edge_loss_val,
        'node': node_loss_val # Needs update
    }


# --- MODIFIED train_rnn_step ---
def train_rnn_step(graph_rnn, edge_rnn, data,
                   criterion_edge, # Renamed from criterion
                   criterion_node, # ADDED: Loss criterion for node types
                   optim_graph_rnn,
                   optim_edge_rnn,
                   scheduler_graph_rnn,
                   scheduler_edge_rnn,
                   device, use_edge_features,
                   node_loss_weight=1.0): # ADDED: Weight for node loss
    """ Train GraphRNN with RNN edge model, including node type prediction loss. """
    graph_rnn.zero_grad()
    edge_rnn.zero_grad()

    # --- Get Data ---
    seq, lens = data['x'].float().to(device), data['len'].cpu() # Edge sequence targets
    levels_padded = data.get('levels') # Node levels
    y_node_type = data.get('y_node_type') # ADDED: Target node types
    # --- End Get Data ---

    # Validate node type targets are present if needed
    predict_node_types_flag = getattr(graph_rnn, 'predict_node_types', False)
    if predict_node_types_flag and y_node_type is None:
        raise ValueError("Node type prediction is enabled in model, but 'y_node_type' not found in data batch.")
    if predict_node_types_flag:
        y_node_type = y_node_type.long().to(device) # Shape: [batch, max_n-1]

    batch_size, seq_len_padded, effective_m, num_features = seq.shape

    if levels_padded is not None:
        levels_padded = levels_padded.long().to(device)

    if len(seq.shape) == 3:
        seq = seq.unsqueeze(3)
        num_features = 1

    # --- Prepare Node RNN Input ---
    one_frame_node = torch.ones([batch_size, 1, effective_m, num_features], device=device) # SOS token
    x_node_rnn_full = torch.cat((one_frame_node, seq), dim=1) # Input: SOS + edge sequence
    lens_node_rnn = lens + 1 # Lengths including SOS token
    # --- End Input Prep ---

    # --- Prepare Levels tensor for GraphRNN input ---
    levels_for_rnn = None
    if levels_padded is not None and hasattr(graph_rnn, 'level_embedding') and graph_rnn.level_embedding is not None:
        sos_level = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        target_level_len = seq_len_padded
        levels_sliced = levels_padded[:, :target_level_len]
        levels_for_rnn = torch.cat((sos_level, levels_sliced), dim=1)

        if levels_for_rnn.shape[1] != x_node_rnn_full.shape[1]:
            print(f"Warning: Level tensor shape mismatch. Adjusting.")
            target_len = x_node_rnn_full.shape[1]
            levels_for_rnn = torch.cat((sos_level, levels_padded[:, :target_len - 1]), dim=1)
            if levels_for_rnn.shape[1] != target_len:
                print("ERROR: Cannot align level tensor length. Disabling.")
                levels_for_rnn = None
    # --- End ---

    # --- Node RNN Forward Pass ---
    graph_rnn.reset_hidden()
    # Get output(s) from node model
    node_model_output = graph_rnn(x_node_rnn_full, lens_node_rnn, levels=levels_for_rnn)

    # Unpack output based on whether node prediction is enabled
    if predict_node_types_flag:
        if not isinstance(node_model_output, tuple) or len(node_model_output) != 2:
             raise RuntimeError("GraphLevelRNN with predict_node_types=True should return (final_output, node_type_logits)")
        hidden, node_type_logits = node_model_output
        # hidden shape: [batch, seq_len+1, node_output_size]
        # node_type_logits shape: [batch, seq_len+1, num_node_types]
    else:
        hidden = node_model_output
        node_type_logits = None # No logits produced
    # --- End Node RNN Forward ---

    # --- Node Loss Calculation (if enabled) ---
    node_loss = torch.tensor(0.0, device=device)
    if predict_node_types_flag and node_type_logits is not None:
        # Targets `y_node_type` correspond to nodes 1 to n (output steps 1 to n)
        # Logits `node_type_logits` correspond to output steps 0 to n
        # We need logits for steps 1 to n to match targets
        logits_for_loss = node_type_logits[:, 1:, :] # Shape: [batch, seq_len, num_node_types]
        targets_for_loss = y_node_type # Shape: [batch, seq_len]

        # Create mask based on original sequence lengths `lens` (nodes 1 to n)
        max_len_nodes = logits_for_loss.shape[1] # This is seq_len_padded
        node_mask = torch.arange(max_len_nodes, device=device)[None, :] < lens.to(device)[:, None] # Shape: [batch, seq_len]

        if logits_for_loss.shape[0:2] == targets_for_loss.shape == node_mask.shape:
            # Flatten logits, targets, and mask
            num_classes_node = logits_for_loss.shape[2]
            logits_flat = logits_for_loss.reshape(-1, num_classes_node) # [batch*seq_len, num_node_types]
            targets_flat = targets_for_loss.reshape(-1) # [batch*seq_len]
            mask_flat = node_mask.reshape(-1)           # [batch*seq_len]

            # Apply mask
            valid_logits = logits_flat[mask_flat]
            valid_targets = targets_flat[mask_flat]

            if valid_targets.numel() > 0:
                node_loss = criterion_node(valid_logits, valid_targets) # Assumes criterion_node handles logits
            else:
                print("Warning: No valid targets found for node loss calculation after masking.")
        else:
            print(f"Warning: Shape mismatch during node loss masking. Skipping node loss.")
            print(f"  Logits shape: {logits_for_loss.shape}")
            print(f"  Targets shape: {targets_for_loss.shape}")
            print(f"  Mask shape: {node_mask.shape}")
    # --- End Node Loss Calculation ---

    # --- Edge RNN Processing ---
    edge_loss = torch.tensor(0.0, device=device)
    # Pack hidden states (output steps 1 to n) and target sequence (nodes 1 to n)
    try:
        # hidden shape: [batch, seq_len+1, node_output_size] -> use hidden[:, 1:, :]
        hidden_for_edges_packed = pack_padded_sequence(hidden[:, 1:, :], lens, batch_first=True, enforce_sorted=False)
        hidden_packed_data = hidden_for_edges_packed.data # Shape [TotalNodes, HiddenSize]

        seq_packed_obj = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
        y_edge_rnn_target_packed_data = seq_packed_obj.data # Target edges: Shape [TotalNodes, effective_m, EdgeFeatures]

    except RuntimeError as e:
         print(f"Error during pack_padded_sequence for hidden/seq in train_rnn_step: {e}")
         # Return zero losses to avoid crashing training loop
         return {'total': 0.0, 'edge': 0.0, 'node': node_loss.item()} # Return calculated node_loss

    total_nodes = y_edge_rnn_target_packed_data.shape[0]
    if total_nodes == 0:
        print("Warning: Zero total nodes after packing in train_rnn_step. Skipping edge loss calculation.")
    else:
        # Calculate lengths for edge sequences
        seq_packed_len = []
        m = effective_m
        for l_val in lens.tolist():
            for i in range(1, l_val + 1):
                seq_packed_len.append(min(i, m))
        seq_packed_len_tensor = torch.tensor(seq_packed_len, dtype=torch.long, device=device)

        if len(seq_packed_len) != total_nodes:
             print(f"Warning: Mismatch edge_seq_lens ({len(seq_packed_len)}) vs total_nodes ({total_nodes}).")
             # Attempt to fix or return safely
             if len(seq_packed_len) > total_nodes:
                 seq_packed_len_tensor = seq_packed_len_tensor[:total_nodes]
             else:
                 print("Error: seq_packed_len shorter than total_nodes. Cannot proceed with edge loss.")
                 # Return calculated node loss, zero edge/total loss
                 return {'total': node_loss.item() * node_loss_weight, 'edge': 0.0, 'node': node_loss.item()}

        # Prepare Edge RNN inputs/targets
        one_frame_edge = torch.ones([total_nodes, 1, num_features], device=device) # SOS
        # Input: SOS + shifted target edge data
        x_edge_rnn = torch.cat((one_frame_edge, y_edge_rnn_target_packed_data[:, :-1, :]), dim=1)
        # Target: Original packed edge data
        y_edge_rnn_target = y_edge_rnn_target_packed_data

        # Set hidden state for EdgeRNN using node context
        try:
            edge_rnn.set_first_layer_hidden(hidden_packed_data)
        except Exception as e:
             print(f"Error setting edge RNN hidden state: {e}. Skipping edge loss.")
             # Return calculated node loss, zero edge/total loss
             return {'total': node_loss.item() * node_loss_weight, 'edge': 0.0, 'node': node_loss.item()}


        # Compute edge predictions
        y_edge_rnn_pred = edge_rnn(x_edge_rnn, seq_packed_len_tensor.cpu(), return_logits=use_edge_features)

        # Pad the target edge sequence for loss calculation
        try:
            y_edge_rnn_target_packed_again = pack_padded_sequence(y_edge_rnn_target, seq_packed_len_tensor.cpu(), batch_first=True, enforce_sorted=False)
            y_edge_rnn_target_padded, _ = pad_packed_sequence(y_edge_rnn_target_packed_again, batch_first=True, total_length=y_edge_rnn_pred.shape[1])
        except RuntimeError as e:
            print(f"Error packing/padding y_edge_rnn_target: {e}")
            # Return calculated node loss, zero edge/total loss
            return {'total': node_loss.item() * node_loss_weight, 'edge': 0.0, 'node': node_loss.item()}

        # Calculate masked edge loss
        max_pred_len = y_edge_rnn_pred.shape[1]
        edge_mask = torch.arange(max_pred_len, device=device)[None, :] < seq_packed_len_tensor[:, None]

        if use_edge_features: # Multi-class edge loss
            pred_nodes, pred_len, pred_classes = y_edge_rnn_pred.shape
            target_nodes, target_len, target_features = y_edge_rnn_target_padded.shape

            if pred_nodes == target_nodes and pred_len == target_len:
                 num_classes_edge = pred_classes
                 y_pred_flat = y_edge_rnn_pred.reshape(-1, num_classes_edge)
                 try:
                     y_labels_flat = torch.argmax(y_edge_rnn_target_padded, dim=-1).reshape(-1)
                 except RuntimeError as e:
                     print(f"Error argmax/reshape y_edge_rnn_target_padded: {e}")
                     y_labels_flat = None

                 if y_labels_flat is not None:
                     mask_flat = edge_mask.reshape(-1)
                     if y_pred_flat.shape[0] == y_labels_flat.shape[0] == mask_flat.shape[0]:
                         valid_preds = y_pred_flat[mask_flat]
                         valid_labels = y_labels_flat[mask_flat]
                         if valid_labels.numel() > 0:
                              edge_loss = criterion_edge(valid_preds, valid_labels.long())
                     else:
                         print(f"Warning: Flattened shape mismatch in RNN edge loss calculation.")
            else:
                 print(f"Warning: Shape mismatch pred vs target in RNN step edge loss.")

        else: # Binary edge loss
            print("Warning: Using binary edge loss path for RNN step.")
            mask_expanded = edge_mask.unsqueeze(-1).expand_as(y_edge_rnn_pred)
            if torch.any(mask_expanded):
                edge_loss = criterion_edge(y_edge_rnn_pred[mask_expanded], y_edge_rnn_target_padded[mask_expanded])
    # --- End Edge RNN Processing ---

    # --- Combine Losses ---
    # Apply weight to node loss before combining
    total_loss = edge_loss + node_loss_weight * node_loss
    # --- End Combine Losses ---

    # --- Backpropagation and Optimizing ---
    if torch.isnan(total_loss) or torch.isinf(total_loss):
         print(f"Warning: Invalid total loss detected (NaN or Inf): {total_loss.item()}. Skipping backpropagation.")
    elif total_loss.requires_grad and total_loss > 0:
        total_loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(graph_rnn.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(edge_rnn.parameters(), max_norm=1.0)
        # Step optimizers
        optim_graph_rnn.step()
        optim_edge_rnn.step()

    # Step schedulers
    scheduler_graph_rnn.step()
    scheduler_edge_rnn.step()
    # --- End Backpropagation ---

    # Return loss dictionary
    return {
        'total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        'edge': edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
        'node': node_loss.item() if isinstance(node_loss, torch.Tensor) else node_loss # Return node loss
    }
