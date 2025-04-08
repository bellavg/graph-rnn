
import traceback

# Assume NUM_EDGE_FEATURES is available, e.g., from aig_dataset or config
# If not, determine dynamically from tensor shapes where possible.
def train_mlp_step(graph_rnn, edge_mlp, data,
                   criterion_edge,
                   optim_graph_rnn, optim_edge_mlp,
                   scheduler_graph_rnn, scheduler_mlp, device, use_edge_features, **kwargs):
    """ Train GraphRNN with MLP edge model (Level Embedding Added). """
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
    graph_rnn.reset_hidden()
    # Pass levels to graph_rnn
    hidden = graph_rnn(x, lens_with_sos, levels=levels_for_rnn)
    # --- End ---

    # --- EdgeMLP Forward Pass ---
    y_pred = edge_mlp(hidden, return_logits=use_edge_features)

    # --- Edge Loss Calculation ---
    edge_loss = torch.tensor(0.0, device=device)
    try:
        # Pack target sequence y based on actual lengths
        y_packed = pack_padded_sequence(y, lens_with_sos.cpu(), batch_first=True, enforce_sorted=False)
        # Pad back to max length in this batch for comparison
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

    # Total loss (only edge loss in this version)
    total_loss = edge_loss

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
    total_loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

    return {
        'total': total_loss_val,
        'edge': edge_loss_val,
        'node': 0.0  # MLP step doesn't calculate node loss separately here
    }




# src/train.py (showing relevant modifications in train_rnn_step)

import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence



def train_rnn_step(
    # Base model components
    graph_rnn,
    edge_rnn,
    data,
    criterion_edge,
    optim_graph_rnn,
    optim_edge_rnn,
    scheduler_graph_rnn,
    scheduler_edge_rnn,
    device,
    use_edge_features,
    # Node predictor components (optional)
    execute_node_pred_step_flag: bool, # Controls if node pred happens this step
    node_type_predictor=None,
    criterion_node=None,
    optim_node_type=None,
    scheduler_node_type=None
    ):
    """
    Train GraphRNN with RNN edge model and optional independent node predictor.
    """

    # --- Zero Grads ---
    graph_rnn.zero_grad(set_to_none=True) # Use set_to_none=True for potential efficiency
    edge_rnn.zero_grad(set_to_none=True)
    if execute_node_pred_step_flag and optim_node_type:
        optim_node_type.zero_grad(set_to_none=True)

    # --- Prepare Data ---
    seq, lens = data['x'].float().to(device), data['len'].cpu()
    batch_size, seq_len_padded, effective_m, num_features = seq.shape

    levels_padded = data.get('levels')
    if levels_padded is not None:
        levels_padded = levels_padded.long().to(device)

    node_types_true_padded = None
    can_execute_node_pred = execute_node_pred_step_flag # Rename for clarity
    if execute_node_pred_step_flag: # Only load if we might train
        if 'node_types' not in data:
            print("Warning (train_rnn_step): 'node_types' missing. Disabling node pred.")
            can_execute_node_pred = False
        else:
            node_types_true_padded = data['node_types'].long().to(device)

    if len(seq.shape) == 3:
        seq = seq.unsqueeze(3)
        num_features = 1

    # --- Prepare Node RNN Input ---
    one_frame_node = torch.ones(
        [batch_size, 1, effective_m, num_features], device=device
    )
    x_node_rnn_full = torch.cat((one_frame_node, seq), dim=1)
    lens_node_rnn = lens + 1

    levels_for_rnn = None
    if levels_padded is not None and hasattr(graph_rnn, 'level_embedding') and \
       graph_rnn.level_embedding is not None:
        try:
            sos_level = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            target_level_len = seq_len_padded
            current_level_len = levels_padded.shape[1]
            if current_level_len < target_level_len:
                levels_padded = torch.nn.functional.pad(
                    levels_padded, (0, target_level_len - current_level_len), value=0
                )
            levels_sliced = levels_padded[:, :target_level_len]
            levels_for_rnn = torch.cat((sos_level, levels_sliced), dim=1)
            if levels_for_rnn.shape[1] != x_node_rnn_full.shape[1]:
                print(f"Warn: Level shape mismatch. Disabling levels.")
                levels_for_rnn = None
        except Exception as e_lvl:
            print(f"Error processing levels: {e_lvl}")
            levels_for_rnn = None

    # --- Compute hidden graph-level representation ---
    graph_rnn.reset_hidden()
    try:
        output = graph_rnn(x_node_rnn_full, lens_node_rnn, levels=levels_for_rnn)
        hidden_state_sequence = output[0] if isinstance(output, tuple) else output
    except Exception as e_graph_fwd:
        print(f"Error in graph_rnn forward: {e_graph_fwd}")
        return {'total': 0.0, 'edge': 0.0, 'node': 0.0}


    # --- Pack hidden states and target sequence for Edge RNN ---
    hidden_for_edges = hidden_state_sequence[:, 1:, :] # States for nodes 1..n
    try:
        hidden_for_edges_packed = pack_padded_sequence(
            hidden_for_edges, lens, batch_first=True, enforce_sorted=False
        )
        hidden_packed_data = hidden_for_edges_packed.data # Input for node pred

        seq_packed_obj = pack_padded_sequence(
            seq, lens, batch_first=True, enforce_sorted=False
        )
        y_edge_rnn_target_packed_data = seq_packed_obj.data # Target edges
    except RuntimeError as e_pack:
        print(f"Error packing hidden/seq in train_rnn_step: {e_pack}")
        return {'total': 0.0, 'edge': 0.0, 'node': 0.0}

    total_nodes_in_batch = y_edge_rnn_target_packed_data.shape[0]
    if total_nodes_in_batch == 0:
        return {'total': 0.0, 'edge': 0.0, 'node': 0.0}

    # Calculate edge sequence lengths
    seq_packed_len = [min(i, effective_m)
                      for l_val in lens.tolist()
                      for i in range(1, l_val + 1)]
    seq_packed_len_tensor = torch.tensor(seq_packed_len, dtype=torch.long, device=device)
    if len(seq_packed_len) != total_nodes_in_batch:
        print(f"Warn: Mismatch edge_seq_lens vs total_nodes.")
        # Attempt to fix if possible, otherwise risk error later
        seq_packed_len_tensor = seq_packed_len_tensor[:total_nodes_in_batch]


    # --- Edge RNN Forward Pass ---
    try:
        one_frame_edge = torch.ones(
            [total_nodes_in_batch, 1, num_features], device=device
        )
        # Input uses packed targets shifted
        x_edge_rnn = torch.cat(
            (one_frame_edge, y_edge_rnn_target_packed_data[:, :-1, :]), dim=1
        )
        edge_rnn.set_first_layer_hidden(hidden_packed_data)
        y_edge_rnn_pred_logits = edge_rnn(
            x_edge_rnn, seq_packed_len_tensor.cpu(), return_logits=True
        )
    except Exception as e_edge_fwd:
        print(f"Error in edge_rnn forward: {e_edge_fwd}")
        return {'total': 0.0, 'edge': 0.0, 'node': 0.0}


    # --- Initialize Losses ---
    edge_loss_val = 0.0
    node_loss_val = 0.0

    # ==============================================
    # --- Independent Node Type Prediction Step ---
    # ==============================================
    if can_execute_node_pred:
        try:
            # Predict on the packed hidden states
            node_type_logits = node_type_predictor(hidden_packed_data)

            # Prepare ground truth labels (packed)
            true_types_packed_obj = pack_padded_sequence(
                node_types_true_padded, lens, batch_first=True, enforce_sorted=False
            )
            true_types_packed = true_types_packed_obj.data

            if node_type_logits.shape[0] == true_types_packed.shape[0]:
                # Calculate loss (criterion ignores index -1 by default)
                current_node_loss = criterion_node(node_type_logits, true_types_packed)

                # Backpropagate Node Loss SEPARATELY if valid
                if not torch.isnan(current_node_loss) and \
                   not torch.isinf(current_node_loss) and \
                   current_node_loss.requires_grad:

                    node_loss_val = current_node_loss.item() # Store loss value
                    # Zero grad, backward, clip, step optimizer, step scheduler
                    optim_node_type.zero_grad()
                    current_node_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        node_type_predictor.parameters(), max_norm=1.0
                    )
                    optim_node_type.step()
                    if scheduler_node_type:
                        scheduler_node_type.step()
                else:
                    print(f"Warn: Invalid node_loss. Skipping node backprop.")
            else:
                print(f"Warn: Shape mismatch node logits vs labels. Skipping node loss.")

        except Exception as e_node_pred:
            print(f"Error during node prediction step: {e_node_pred}")
            traceback.print_exc()
    # ==============================================
    # --- End Node Type Prediction Step ---
    # ==============================================


    # ==============================================
    # --- Edge Prediction Step ---
    # ==============================================
    try:
        # Pad the target sequence for loss calculation
        y_edge_rnn_target_packed_for_loss = pack_padded_sequence(
            y_edge_rnn_target_packed_data,
            seq_packed_len_tensor.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        y_edge_rnn_target_padded, _ = pad_packed_sequence(
            y_edge_rnn_target_packed_for_loss,
            batch_first=True,
            total_length=y_edge_rnn_pred_logits.shape[1] # Match prediction length
        )

        # Masking and Edge Loss Calculation
        max_pred_len = y_edge_rnn_pred_logits.shape[1]
        edge_mask = torch.arange(max_pred_len, device=device)[None, :] \
                    < seq_packed_len_tensor[:, None]

        current_edge_loss = torch.tensor(0.0, device=device) # Init loss for step
        if use_edge_features: # CrossEntropyLoss for multi-class edges
            num_classes = y_edge_rnn_pred_logits.shape[-1]
            y_pred_flat = y_edge_rnn_pred_logits.reshape(-1, num_classes)
            y_labels_edge = torch.argmax(y_edge_rnn_target_padded, dim=-1)
            y_labels_flat = y_labels_edge.reshape(-1)
            mask_flat = edge_mask.reshape(-1)

            if y_pred_flat.shape[0] == y_labels_flat.shape[0] == mask_flat.shape[0]:
                valid_preds = y_pred_flat[mask_flat]
                valid_labels = y_labels_flat[mask_flat]
                if valid_labels.numel() > 0:
                    current_edge_loss = criterion_edge(valid_preds, valid_labels.long())
            else:
                print(f"Warn: Flattened shape mismatch edge loss.")
        else: # BCELoss
            print("Warn: Using BCELoss for edges.")
            mask_expanded = edge_mask.unsqueeze(-1).expand_as(y_edge_rnn_pred_logits)
            if torch.any(mask_expanded):
                # Apply sigmoid for BCELoss
                current_edge_loss = criterion_edge(
                    torch.sigmoid(y_edge_rnn_pred_logits[mask_expanded]),
                    y_edge_rnn_target_padded[mask_expanded]
                )

        edge_loss_val = current_edge_loss.item() # Store value

        # Backpropagate Edge Loss for Main Models if valid
        if not torch.isnan(current_edge_loss) and \
           not torch.isinf(current_edge_loss) and \
           current_edge_loss.requires_grad and \
           current_edge_loss.item() > 0: # Avoid backprop on zero loss

            optim_graph_rnn.zero_grad()
            optim_edge_rnn.zero_grad()
            current_edge_loss.backward() # Affects edge_rnn and graph_rnn
            torch.nn.utils.clip_grad_norm_(graph_rnn.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(edge_rnn.parameters(), max_norm=1.0)
            optim_graph_rnn.step()
            optim_edge_rnn.step()
            # Step main schedulers AFTER optimizer step
            scheduler_graph_rnn.step()
            scheduler_edge_rnn.step()
        else:
            # Log if skipping backprop for non-zero invalid loss
            if not (current_edge_loss.item() == 0 and not current_edge_loss.requires_grad):
                 print(f"Warn: Invalid edge_loss. Skipping main backprop.")
            # Step schedulers even if loss is zero/invalid (common practice)
            scheduler_graph_rnn.step()
            scheduler_edge_rnn.step()

    except Exception as e_edge_pred:
        print(f"Error during edge prediction step: {e_edge_pred}")
        traceback.print_exc()
    # ==============================================
    # --- End Edge Prediction Step ---
    # ==============================================

    # --- Total Loss (for logging) ---
    total_loss_val = edge_loss_val + node_loss_val

    # --- Return loss dictionary ---
    return {
        'total': total_loss_val,
        'edge': edge_loss_val,
        'node': node_loss_val
    }
