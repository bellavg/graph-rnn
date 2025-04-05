
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def train_mlp_step(graph_rnn, edge_mlp, data,
                   criterion_edge, criterion_node, # Use specific criteria
                   optim_graph_rnn, optim_edge_mlp,
                   scheduler_graph_rnn, scheduler_mlp, device,
                   use_edge_features, predict_node_types, use_conditioning): # Pass flags
    graph_rnn.zero_grad()
    edge_mlp.zero_grad()

    # Get mandatory data
    s, lens = data['x'].float().to(device), data['len'].cpu()
    # Get optional data
    node_type_labels = data.get('node_types')
    if node_type_labels is not None:
        node_type_labels = node_type_labels.long().to(device)

    truth_table = data.get('y')
    if use_conditioning and truth_table is not None:
        truth_table = truth_table.float().to(device)
        if len(truth_table.shape) > 2: # Flatten if needed
            truth_table = truth_table.reshape(truth_table.shape[0], -1)
    else:
        truth_table = None # Ensure it's None if not conditioning

    # Add feature dim if missing (original logic)
    if len(s.shape) == 3:
        s = s.unsqueeze(3)

    # Prepare inputs/targets with SOS/EOS tokens (original logic)
    one_frame = torch.ones([s.shape[0], 1, s.shape[2], s.shape[3]], device=device)
    zero_frame = torch.zeros([s.shape[0], 1, s.shape[2], s.shape[3]], device=device)
    x = torch.cat((one_frame, s), dim=1) # Input includes SOS
    y = torch.cat((s, zero_frame), dim=1) # Target includes EOS

    lens_with_sos = lens + 1

    # GraphRNN forward pass
    graph_rnn.reset_hidden()
    # Pass truth_table only if conditioning is enabled
    output = graph_rnn(x, lens_with_sos, truth_table=truth_table if use_conditioning else None)

    # Unpack output based on whether node types are predicted
    hidden = None
    node_type_logits = None
    if predict_node_types:
        hidden, node_type_logits = output
    else:
        hidden = output

    # EdgeMLP forward pass
    # Pass truth_table only if conditioning is enabled
    y_pred = edge_mlp(hidden, return_logits=True, # ALWAYS return logits for CrossEntropy
                      truth_table=truth_table if use_conditioning else None)

    # --- Edge Loss Calculation ---
    # Target 'y' needs packing/padding to align dimensions
    y_packed = pack_padded_sequence(y, lens_with_sos, batch_first=True, enforce_sorted=False)
    y_padded, _ = pad_packed_sequence(y_packed, batch_first=True)

    edge_loss = 0.0
    if use_edge_features:
        # CrossEntropyLoss expects [N, C] predictions and [N] labels
        # y_pred shape: [batch, seq_len+1, m, num_edge_classes]
        # y_padded shape: [batch, seq_len+1, m, num_edge_classes] (one-hot)

        num_classes = y_pred.shape[-1]
        y_pred_reshaped = y_pred.reshape(-1, num_classes) # [Batch*Seq*M, C]

        y_indices = torch.argmax(y_padded, dim=-1) # [batch, seq_len+1, m]
        y_labels_reshaped = y_indices.reshape(-1) # [Batch*Seq*M]

        # Create mask to ignore loss on padding tokens
        # We need mask based on lens_with_sos for y_pred/y_padded shape
        mask = torch.arange(y_pred.shape[1], device=device)[None, :] < lens_with_sos[:, None] # [batch, seq_len+1]
        mask = mask.unsqueeze(-1).expand(-1, -1, y_pred.shape[2]) # [batch, seq_len+1, m]
        mask_flat = mask.reshape(-1) # [Batch*Seq*M]

        # Apply criterion only to valid (non-padded) elements
        edge_loss = criterion_edge(y_pred_reshaped[mask_flat], y_labels_reshaped[mask_flat].long())

    else: # Original BCELoss path (requires sigmoid activation in MLP)
        # Ensure MLP applies sigmoid if return_logits=False AND you use BCELoss
        # If using BCEWithLogitsLoss, MLP should return logits.
        # Assuming BCELoss and MLP returns probabilities (sigmoid applied):
        # Need masking for BCELoss too
        mask = torch.arange(y_pred.shape[1], device=device)[None, :] < lens_with_sos[:, None]
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(y_pred)
        edge_loss = criterion_edge(y_pred[mask], y_padded[mask])


    # --- Node Loss Calculation (Conditional) ---
    node_loss = 0.0
    if predict_node_types and criterion_node is not None and node_type_logits is not None and node_type_labels is not None:
        # node_type_logits shape: [batch, seq_len+1, num_node_types]
        # node_type_labels shape: [batch, orig_seq_len] (NO SOS)

        batch_size, seq_len_plus_1, num_classes = node_type_logits.shape
        orig_seq_len = node_type_labels.shape[1]

        # Mask valid positions (ignore padding AND SOS token position for labels)
        # Mask based on original lengths 'lens' for labels, 'lens_with_sos' for logits
        logit_mask = torch.arange(seq_len_plus_1, device=device)[None, :] < lens_with_sos[:, None] # [batch, seq_len+1]

        # Align labels: Create tensor with padding value (-100), fill with actual labels shifted by 1 (for SOS)
        aligned_labels = torch.full((batch_size, seq_len_plus_1), criterion_node.ignore_index, dtype=torch.long, device=device)
        for i in range(batch_size):
            actual_len = min(lens[i].item(), orig_seq_len) # Use original length
            if actual_len > 0:
                aligned_labels[i, 1:actual_len + 1] = node_type_labels[i, :actual_len]

        # Flatten logits and labels WHERE MASK IS TRUE
        valid_logits = node_type_logits[logit_mask] # [NumValidSteps, C]
        valid_labels = aligned_labels[logit_mask]   # [NumValidSteps]

        # Exclude loss calculation for SOS tokens explicitly if needed, although ignore_index handles padding
        # valid_labels will have -100 at padded positions AND potentially at SOS if mask included it & labels started there
        # The ignore_index=-100 in CrossEntropyLoss handles this correctly.

        if valid_labels.numel() > 0: # Ensure there are valid labels to compute loss on
            node_loss = criterion_node(valid_logits, valid_labels)
        else:
            node_loss = torch.tensor(0.0, device=device) # Or handle as appropriate

    # Combine losses
    total_loss = edge_loss + node_loss

    # Backpropagation and optimizer steps (original logic)
    total_loss.backward()
    optim_graph_rnn.step()
    optim_edge_mlp.step()
    scheduler_graph_rnn.step()
    scheduler_mlp.step()

    # Return dictionary of losses
    return {
        'total': total_loss.item(),
        'edge': edge_loss.item(),
        'node': node_loss.item() if isinstance(node_loss, torch.Tensor) else node_loss # Handle tensor or float
    }


def train_rnn_step(graph_rnn, edge_rnn, data,
                   criterion_edge, criterion_node,  # Pass specific criteria
                   optim_graph_rnn, optim_edge_rnn, # Note: optim_edge_rnn used here
                   scheduler_graph_rnn, scheduler_edge_rnn, # Note: scheduler_edge_rnn used here
                   device, use_edge_features,
                   predict_node_types, use_conditioning): # Pass flags
    """
    Train GraphRNN with RNN edge model, including optional node type prediction
    and truth table conditioning.
    """
    graph_rnn.zero_grad()
    edge_rnn.zero_grad()

    # Get mandatory data
    seq, lens = data['x'].float().to(device), data['len'].cpu()
    # Get optional data
    node_type_labels = data.get('node_types')
    if predict_node_types and node_type_labels is not None:
        node_type_labels = node_type_labels.long().to(device)
    else:
        node_type_labels = None # Ensure it's None if not predicting or not present

    truth_table = data.get('y')
    if use_conditioning and truth_table is not None:
        truth_table = truth_table.float().to(device)
        if len(truth_table.shape) > 2: # Flatten if needed
            truth_table = truth_table.reshape(truth_table.shape[0], -1)
    else:
        truth_table = None # Ensure it's None if not conditioning

    # Add feature dim if missing (original logic)
    if len(seq.shape) == 3:
        seq = seq.unsqueeze(3)

    # Add SOS token to the node-level RNN input (original logic)
    # Input for node RNN uses sequence up to T-1
    one_frame_node = torch.ones([seq.shape[0], 1, seq.shape[2], seq.shape[3]], device=device)
    x_node_rnn = torch.cat((one_frame_node, seq[:, :-1, :]), dim=1) # Use seq[:, :-1, :] to match length with lens

    # Compute hidden graph-level representation
    graph_rnn.reset_hidden()
    # Pass truth_table conditionally
    output = graph_rnn(x_node_rnn, lens, # Use original lens here
                       truth_table=truth_table if use_conditioning else None)

    # Unpack output based on whether node types are predicted
    hidden = None
    node_type_logits = None
    if predict_node_types and isinstance(output, tuple):
        hidden, node_type_logits = output # hidden shape [batch, seq_len, hidden_size]
                                           # node_type_logits shape [batch, seq_len, num_node_types]
    else:
        hidden = output # shape [batch, seq_len, hidden_size]

    # --- Edge RNN Processing ---

    # Pack the full sequence (including last element) for edge RNN targets
    # seq shape [batch, orig_seq_len, m, features]
    seq_packed_obj = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
    seq_packed = seq_packed_obj.data # This holds the actual data, shape [total_nodes, m, features]
    batch_sizes = seq_packed_obj.batch_sizes # Needed for unpacking later if required

    # Prepare sequence lengths for the packed edge data
    # Each node `i` in sequence `j` gets a length of min(i+1, m)
    # (This matches the original code's logic, although slightly complex to derive seq_packed_len)
    seq_packed_len = []
    m = graph_rnn.input_size # Assuming input_size is 'm'
    for batch_idx, l in enumerate(lens):
        for node_idx in range(l.item()):
             # The length of the edge vector for node `node_idx` (0-indexed) is min(node_idx + 1, m)
             # Note: Original code had `min(i, m)` where i was 1-based. If node_idx is 0-based, need +1.
             # Let's stick to the structure from file: using 1 to l+1 range for i
             seq_packed_len.append(min(node_idx + 1, m))
    # seq_packed_len.sort() # Sorting might not be necessary if pack/pad handles it

    # Add SOS token for edge RNN input (original logic)
    # Input is SOS + packed_sequence up to T-1 for edges
    # seq_packed shape: [TotalNodes, m, Features]
    # We need input shape for RNN: [TotalNodes, MaxEdgeSeqLen, Features] ?? - No, RNN takes [N, L, H_in]
    # The original code's logic adds SOS to seq_packed directly? Seems tricky.
    # Let's reconsider x_edge_rnn and y_edge_rnn based on EdgeLevelRNN input expectations.
    # EdgeLevelRNN expects input [N, L, H_in] where N is batch, L is seq_len, H_in is features.
    # Here, our "batch" for edge RNN is TotalNodes. SeqLen is 'm'. Features is 'edge_feature_len'.
    # seq_packed needs reshaping or careful handling.

    # --- REVISED Edge RNN Input/Target Prep ---
    # Target is the packed sequence itself
    y_edge_rnn = seq_packed # Shape [TotalNodes, m, EdgeFeatures]

    # Input needs SOS. Create SOS tokens. Edge feature dim should match y_edge_rnn.
    # Shape [TotalNodes, 1, EdgeFeatures]
    sos_edge = torch.ones(y_edge_rnn.shape[0], 1, y_edge_rnn.shape[2], device=device)

    # Input is SOS followed by target sequence shifted (edges 0 to m-1)
    # Shape [TotalNodes, m, EdgeFeatures]
    x_edge_rnn = torch.cat([sos_edge, y_edge_rnn[:, :-1, :]], dim=1)

    # Lengths for the edge sequences (should correspond to x_edge_rnn)
    # Each edge sequence has length min(node_idx+1, m) + 1 (for SOS)
    # Let's use seq_packed_len derived earlier, maybe it's correct for the targets y_edge_rnn
    # The lengths passed to edge_rnn should match x_edge_rnn.
    # Let's assume seq_packed_len is correct for the *target* edge sequences (y_edge_rnn).
    # The input x_edge_rnn also has these lengths.

    # --- Conditional Truth Table for Edge RNN ---
    edge_truth_table = None
    if use_conditioning and truth_table is not None:
        # Need to repeat the graph-level truth table for each node in the packed sequence
        repeated_tt = []
        original_indices = seq_packed_obj.sorted_indices # Get original batch index if sorted
        if original_indices is None: original_indices = torch.arange(len(lens)) # Assume original order if not sorted

        current_pos = 0
        for i in range(len(batch_sizes)):
             batch_size_at_step = batch_sizes[i]
             original_batch_indices_at_step = original_indices[current_pos : current_pos + batch_size_at_step]
             repeated_tt.append(truth_table[original_batch_indices_at_step])
             current_pos += batch_size_at_step
        edge_truth_table = torch.cat(repeated_tt, dim=0) # Shape [TotalNodes, tt_size]


    # Set hidden state for Edge RNN (original logic)
    hidden_packed = pack_padded_sequence(hidden, lens, batch_first=True, enforce_sorted=False).data # Shape [TotalNodes, HiddenSize]
    edge_rnn.set_first_layer_hidden(hidden_packed)

    # --- Edge RNN Forward Pass ---
    # Pass edge_truth_table conditionally
    y_edge_rnn_pred = edge_rnn(
        x_edge_rnn,
        x_lens=seq_packed_len, # Pass the calculated lengths
        return_logits=True, # ALWAYS return logits for CrossEntropy
        truth_table=edge_truth_table if use_conditioning else None
    ) # Output shape: [TotalNodes, m, EdgeFeatures]

    # --- Edge Loss Calculation ---
    # Target y_edge_rnn needs padding/masking based on seq_packed_len
    # Prediction y_edge_rnn_pred also needs masking

    # Pad the target sequence based on seq_packed_len
    # Requires converting seq_packed_len to tensor
    y_edge_rnn_padded = pack_padded_sequence(y_edge_rnn, torch.tensor(seq_packed_len, device=device), batch_first=True, enforce_sorted=False)
    y_edge_rnn_padded, _ = pad_packed_sequence(y_edge_rnn_padded, batch_first=True, total_length=m) # Pad to max length m

    # Create mask based on seq_packed_len
    edge_mask = torch.arange(m, device=device)[None, :] < torch.tensor(seq_packed_len, device=device)[:, None] # [TotalNodes, m]

    edge_loss = 0.0
    if use_edge_features:
        # CrossEntropyLoss: expects [N, C], target [N]
        # y_edge_rnn_pred shape: [TotalNodes, m, NumEdgeClasses]
        # y_edge_rnn_padded shape: [TotalNodes, m, NumEdgeClasses] (one-hot)

        num_classes = y_edge_rnn_pred.shape[-1]
        # Reshape predictions: [TotalNodes * m, NumEdgeClasses]
        y_pred_flat = y_edge_rnn_pred.reshape(-1, num_classes)

        # Convert targets to labels: [TotalNodes, m] -> [TotalNodes * m]
        y_labels_flat = torch.argmax(y_edge_rnn_padded, dim=-1).reshape(-1)

        # Flatten mask: [TotalNodes, m] -> [TotalNodes * m]
        mask_flat = edge_mask.reshape(-1)

        # Apply criterion only to valid elements defined by the mask
        edge_loss = criterion_edge(y_pred_flat[mask_flat], y_labels_flat[mask_flat].long())

    else: # Original BCELoss path
        # Ensure edge_rnn applies sigmoid if return_logits=False for BCELoss
        # Assuming BCELoss and edge_rnn returns probabilities (sigmoid applied):
        edge_mask_expanded = edge_mask.unsqueeze(-1).expand_as(y_edge_rnn_pred)
        edge_loss = criterion_edge(y_edge_rnn_pred[edge_mask_expanded], y_edge_rnn_padded[edge_mask_expanded])


    # --- Node Loss Calculation (Conditional) ---
    node_loss = 0.0
    if predict_node_types and criterion_node is not None and node_type_logits is not None and node_type_labels is not None:
        # node_type_logits shape: [batch, seq_len, num_node_types] (seq_len matches original lens)
        # node_type_labels shape: [batch, orig_seq_len]

        batch_size, seq_len_logits, num_classes = node_type_logits.shape
        orig_seq_len = node_type_labels.shape[1]

        # Mask valid positions based on original lengths 'lens'
        node_mask = torch.arange(seq_len_logits, device=device)[None, :] < lens[:, None] # [batch, seq_len_logits]

        # Get only the valid node type predictions
        valid_logits = node_type_logits[node_mask] # [NumValidNodes, C]

        # Align labels: Use padding value (-100) for positions not in original labels
        pad_value = -100 # criterion_node.ignore_index if available
        aligned_labels = torch.full((batch_size, seq_len_logits), pad_value, dtype=torch.long, device=device)

        # Fill with actual labels (SOS token position in logits corresponds to no label)
        for i in range(batch_size):
            actual_len = min(lens[i].item(), orig_seq_len)
            # Fill labels starting from index 1 in aligned_labels to skip SOS position
            aligned_labels[i, :actual_len] = node_type_labels[i, :actual_len] # Assuming logits seq_len matches labels seq_len here

        # Get only the valid labels based on the mask
        valid_labels = aligned_labels[node_mask]   # [NumValidNodes]

        if valid_labels.numel() > 0:
            node_loss = criterion_node(valid_logits, valid_labels)
        else:
             node_loss = torch.tensor(0.0, device=device)


    # Combine losses
    total_loss = edge_loss + node_loss

    # Backpropagation and optimizer steps
    total_loss.backward()
    optim_graph_rnn.step()
    optim_edge_rnn.step() # Use edge RNN optimizer
    scheduler_graph_rnn.step()
    scheduler_edge_rnn.step() # Use edge RNN scheduler

    # Return dictionary of losses
    return {
        'total': total_loss.item(),
        'edge': edge_loss.item(),
        'node': node_loss.item() if isinstance(node_loss, torch.Tensor) else node_loss
    }

