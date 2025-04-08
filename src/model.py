import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math
from aig_dataset import NUM_NODE_TYPES
import torch.nn.functional as F # <--- Added for NodeTypePredictor


class NodeTypePredictor(nn.Module):
    """
    Predicts node types based on the hidden state from a node-level model.
    """
    def __init__(self, hidden_size: int, num_node_types: int = NUM_NODE_TYPES):
        """
        Args:
            node_hidden_size: The dimension of the hidden state output by the node-level model.
            num_node_types: The number of distinct node types to predict.
        """
        super().__init__()
        # Simple linear layer for prediction
        self.predictor = nn.Linear(hidden_size, num_node_types)
        self.num_node_types = num_node_types
        print(f"Initialized NodeTypePredictor: node_hidden_size={hidden_size}, num_node_types={num_node_types}")

    def forward(self, node_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Predicts node type logits.

        Args:
            node_hidden_state: Tensor output from the node-level model.
                              Shape: [batch_size, seq_len, node_hidden_size]

        Returns:
            Tensor: Logits for each node type.
                    Shape: [batch_size, seq_len, num_node_types]
        """
        logits = self.predictor(node_hidden_state)
        return logits

    def predict_types(self, node_hidden_state: torch.Tensor) -> torch.Tensor:
        """ Predicts the most likely node type index. """
        logits = self.forward(node_hidden_state)
        probabilities = F.softmax(logits, dim=-1)
        predicted_indices = torch.argmax(probabilities, dim=-1)
        return predicted_indices


# --- Renamed GraphLevelAttentionRNN ---
class GraphLevelAttentionRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 output_size=None, edge_feature_len=3,
                 # Kept for flexibility
                 use_conditioning=False, tt_size=None, # Kept for flexibility
                 max_level=None,
                 attention_heads=4, # Default 4 heads
                 attention_dropout=0.1): # Default 0.1 dropout
        super().__init__()
        # --- (Keep the __init__ implementation from the previous response) ---
        self.input_size = input_size # m or max_nodes-1
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.edge_feature_len = edge_feature_len

        self.use_conditioning = use_conditioning
        self.tt_size = tt_size
        self.max_level = max_level

        lin_in_features = input_size * edge_feature_len
        if self.use_conditioning and self.tt_size is not None:
            lin_in_features += self.tt_size
        self.linear_in = nn.Linear(lin_in_features, embedding_size)
        self.relu = nn.ReLU()

        self.level_embedding = None
        if self.max_level is not None and self.max_level >= 0:
             self.level_embedding = nn.Embedding(self.max_level + 1, embedding_size)
             print(f"INFO: GraphLevelAttentionRNN using level embedding up to level {self.max_level}")

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(embedding_size)
        self.attention_dropout_layer = nn.Dropout(attention_dropout)

        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        self.linear_out1 = None
        self.linear_out2 = None
        if output_size:
            self.linear_out1 = nn.Linear(hidden_size, embedding_size)
            self.linear_out2 = nn.Linear(embedding_size, output_size)

        self.hidden = None


    def reset_hidden(self):
        self.hidden = None

    def forward(self, x, x_lens=None, truth_table=None, levels=None):
        # x shape: [batch, seq_len, input_size, edge_feature_len]
        # x_lens shape: [batch] (tensor of actual lengths)
        # levels shape: [batch, seq_len]

        batch_size, seq_len, _, _ = x.shape
        device = x.device # Get device from input tensor

        # 1. Flatten edge features & apply initial embedding
        x_flat = torch.flatten(x, 2, 3)

        # 2. Concatenate truth table if conditioning
        if self.use_conditioning and truth_table is not None:
            tt_expanded = truth_table.unsqueeze(1).expand(-1, seq_len, -1)
            x_flat = torch.cat((x_flat, tt_expanded), dim=2)

        embedded_input = self.relu(self.linear_in(x_flat)) # [batch, seq_len, embedding_dim]

        # 3. Add Level Embedding
        if self.level_embedding is not None and levels is not None:
            if levels.shape[0] == batch_size and levels.shape[1] == seq_len:
                try:
                    clamped_levels = torch.clamp(levels, 0, self.max_level)
                    lvl_emb = self.level_embedding(clamped_levels)
                    embedded_input = embedded_input + lvl_emb
                except Exception as e: print(f"Warning: Error adding level embedding: {e}")

        # --- 4. Create Padding Mask ---
        padding_mask = None
        if x_lens is not None:
            if not isinstance(x_lens, torch.Tensor):
                x_lens_tensor = torch.tensor(x_lens, dtype=torch.long, device=device)
            else:
                x_lens_tensor = x_lens.to(device)
            max_len_pad = seq_len # Use actual sequence length for mask dim
            indices_pad = torch.arange(max_len_pad, device=device).expand(batch_size, max_len_pad)
            padding_mask = indices_pad >= x_lens_tensor.unsqueeze(1) # Shape: [batch, seq_len]

        # --- 5. Create Causal (Look-Ahead) Mask ---
        # Standard square causal mask for sequence length `seq_len`
        # Shape needs to be (seq_len, seq_len) for nn.MultiheadAttention
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        # Convert boolean mask to float mask expected by MHA (-inf for masked positions)
        # Note: MHA automatically handles adding this mask for float types.
        # If using older PyTorch versions, might need:
        # causal_mask = causal_mask.float().masked_fill(causal_mask, float('-inf')).masked_fill(~causal_mask, 0.0)

        # --- 6. Apply Multi-Head Self-Attention ---
        attn_output, attn_weights = self.self_attention(
            query=embedded_input,
            key=embedded_input,
            value=embedded_input,
            key_padding_mask=padding_mask, # Mask based on sequence lengths
            attn_mask=causal_mask          # Mask based on sequence position (causality)
        )
        # Apply dropout, residual connection, and layer normalization
        attended_input = embedded_input + self.attention_dropout_layer(attn_output)
        attended_input_norm = self.attention_norm(attended_input)
        # --- End Attention ---

        # 7. Pack sequence (if lengths provided)
        gru_input = attended_input_norm
        target_padded_length = gru_input.shape[1] # Store length before packing

        if x_lens is not None:
            x_lens_cpu = x_lens if isinstance(x_lens, torch.Tensor) else torch.tensor(x_lens)
            x_lens_cpu = x_lens_cpu.cpu()
            try:
                gru_input = pack_padded_sequence(gru_input, x_lens_cpu, batch_first=True, enforce_sorted=False)
            except RuntimeError as e: print(f"Error packing sequence: {e}")

        # 8. Process through GRU
        gru_output_packed, self.hidden = self.gru(gru_input, self.hidden)

        # 9. Unpack sequence
        gru_output = gru_output_packed
        if x_lens is not None:
             try:
                gru_output, _ = pad_packed_sequence(gru_output_packed, batch_first=True, total_length=target_padded_length)
             except RuntimeError as e: print(f"Error unpacking sequence: {e}")

        # 11. Final output projection (optional)
        final_output = gru_output
        if self.linear_out1:
            final_output = self.relu(self.linear_out1(final_output))
            final_output = self.linear_out2(final_output)

        return final_output


class EdgeLevelAttentionRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers,
                 edge_feature_len=3, # Default AIG edge features
                 attention_heads=4,  # Attention params for this model
                 attention_dropout=0.1,
                 use_conditioning=False, # Optional conditioning
                 tt_size=None,
                 tt_embedding_size=64):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.edge_feature_len = edge_feature_len
        self.embedding_size = embedding_size # Store for attention dim
        self.use_conditioning = use_conditioning and (tt_size is not None)

        # --- Input Embedding for Edge Features ---
        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()

        # --- Conditioning Setup ---
        self.tt_embedding = None
        attn_input_dim = embedding_size # Start with edge embedding dim
        if self.use_conditioning:
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size),
                nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size),
                nn.ReLU()
            )
            # If conditioning, attention input might include TT embedding per step
            # For simplicity here, we'll apply attention *before* concatenating TT,
            # but concatenating first is also possible. Let's keep it simpler for now.
            # gru_input_size will handle TT concatenation later.
            print(f"INFO: EdgeLevelAttentionRNN optionally using TT conditioning.")

        # --- Multi-Head Self-Attention Layer ---
        # Applies attention over the sequence length dimension (the edge sequence)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=attn_input_dim, # Attention on edge embeddings
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(attn_input_dim)
        self.attention_dropout_layer = nn.Dropout(attention_dropout)

        # --- GRU Layer ---
        # Input size depends on whether we concat TT *after* attention
        gru_input_size = attn_input_dim # Start with attention output dim
        if self.use_conditioning:
            gru_input_size += tt_embedding_size # Add TT embedding dim

        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        # --- Output Layers ---
        self.linear_out1 = nn.Linear(hidden_size, embedding_size) # Project back down
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None

    def set_first_layer_hidden(self, h):
        # (This method remains identical to the one in EdgeLevelRNN)
        if h.shape[-1] != self.hidden_size:
             raise ValueError(f"Hidden state dimension mismatch in set_first_layer_hidden. "
                              f"GraphRNN output ({h.shape[-1]}) != EdgeAttentionRNN hidden ({self.hidden_size})")
        if len(h.shape) == 3 and h.shape[0] > 1: h_first_layer = h[0:1, :, :]
        elif len(h.shape) == 2: h_first_layer = h.unsqueeze(0)
        elif len(h.shape) == 3 and h.shape[0] == 1: h_first_layer = h
        else: raise ValueError(f"Unexpected shape for hidden state h: {h.shape}")
        zeros = torch.zeros([self.num_layers-1, h_first_layer.shape[1], h_first_layer.shape[2]], device=h.device)
        self.hidden = torch.cat([h_first_layer, zeros], dim=0)

        # --- In class EdgeLevelAttentionRNN in model.py ---

    def forward(self, x, x_lens=None, return_logits=False, truth_table=None):
        # x shape: [batch (packed), seq_len_edge, edge_feature_len]
        # x_lens shape: [batch (packed)] (tensor of actual edge sequence lengths)
        # truth_table shape: [batch (packed), tt_size] (if used)
        assert self.hidden is not None, "Hidden state not set for EdgeLevelAttentionRNN!"
        device = x.device  # Get device from input tensor

        # 1. Embed edge features
        embedded_x = self.relu(self.linear_in(x))  # [batch, seq_len_edge, embedding_size]
        batch_size, seq_len_edge, _ = embedded_x.shape  # Get dimensions after embedding

        # --- 2. Create Padding Mask ---
        padding_mask = None
        if x_lens is not None:
            # Ensure x_lens is a tensor on the correct device
            if not isinstance(x_lens, torch.Tensor):
                # NOTE: If x_lens comes from pack_padded_sequence's batch_sizes,
                # it needs careful handling. Assuming x_lens is passed correctly
                # corresponding to the batch dimension of x.
                x_lens_tensor = torch.tensor(x_lens, dtype=torch.long, device=device)
            else:
                x_lens_tensor = x_lens.to(device)

            # Create mask: True for positions >= length
            max_len = seq_len_edge  # Use seq dim size from embedded_x
            indices = torch.arange(max_len, device=device).expand(batch_size, max_len)  # [batch, seq_len_edge]
            padding_mask = indices >= x_lens_tensor.unsqueeze(
                1)  # [batch, seq_len_edge] -> True where index >= length

        # --- 3. Apply Self-Attention over edge sequence ---
        attn_output, _ = self.self_attention(
            query=embedded_x,
            key=embedded_x,
            value=embedded_x,
            key_padding_mask=padding_mask  # <<< Pass the mask here
        )
        attended_edges = embedded_x + self.attention_dropout_layer(attn_output)
        attended_edges_norm = self.attention_norm(attended_edges)
        # --- End Attention ---

        # 4. Condition on Truth Table (if enabled) - AFTER attention
        gru_input = attended_edges_norm
        if self.use_conditioning and truth_table is not None and self.tt_embedding is not None:
            truth_table = truth_table.to(gru_input.device)
            tt_emb = self.tt_embedding(truth_table)
            tt_emb_expanded = tt_emb.unsqueeze(1).expand(-1, seq_len_edge, -1)
            gru_input = torch.cat((gru_input, tt_emb_expanded), dim=2)

        # 5. Pack sequence (if lengths provided)
        # Note: Packing *after* attention here. If you pack *before* attention,
        # the masking logic would need adjustment.
        target_padded_length = gru_input.shape[1]
        if x_lens is not None:
            x_lens_cpu = x_lens if isinstance(x_lens, torch.Tensor) else torch.tensor(x_lens)
            x_lens_cpu = x_lens_cpu.cpu()
            try:
                gru_input = pack_padded_sequence(gru_input, x_lens_cpu, batch_first=True, enforce_sorted=False)
            except RuntimeError as e:
                print(f"Error packing sequence in EdgeLevelAttentionRNN: {e}")

        # 6. Process through GRU
        gru_output_packed, self.hidden = self.gru(gru_input, self.hidden)

        # 7. Unpack sequence
        gru_output = gru_output_packed
        if x_lens is not None:
            try:
                gru_output, _ = pad_packed_sequence(gru_output_packed, batch_first=True,
                                                    total_length=target_padded_length)
            except RuntimeError as e:
                print(f"Error unpacking sequence in EdgeLevelAttentionRNN: {e}")

        # 8. Output layers
        out = self.relu(self.linear_out1(gru_output))
        logits = self.linear_out2(out)

        if not return_logits:
            return self.sigmoid(logits)
        else:
            return logits


# --- Original GraphLevelRNN (No Attention) ---
class GraphLevelRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 output_size=None, edge_feature_len=1,
                 # Kept for potential future use
                 use_conditioning=False, tt_size=None, # Kept for potential future use
                 max_level=None): # For level embedding
        super().__init__()
        # --- (Keep original __init__ implementation from your code or the previous non-attention version) ---
        # --- Make sure it includes level embedding logic if you want it in the non-attention version too ---
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.edge_feature_len = edge_feature_len
        # Store these flags
        self.use_conditioning = use_conditioning
        self.tt_size = tt_size
        self.max_level = max_level

        lin_in_features = input_size * edge_feature_len
        if self.use_conditioning and self.tt_size is not None:
            lin_in_features += self.tt_size

        self.linear_in = nn.Linear(lin_in_features, embedding_size)
        self.relu = nn.ReLU()

        self.level_embedding = None
        if self.max_level is not None and self.max_level >= 0:
             self.level_embedding = nn.Embedding(self.max_level + 1, embedding_size)
             print(f"INFO: GraphLevelRNN (No Attention) using level embedding up to level {self.max_level}")

        # Standard GRU (No attention layer here)
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        self.linear_out1 = None
        self.linear_out2 = None
        if output_size:
            self.linear_out1 = nn.Linear(hidden_size, embedding_size)
            self.linear_out2 = nn.Linear(embedding_size, output_size)

        self.hidden = None

    def reset_hidden(self):
        self.hidden = None

    def forward(self, x, x_lens=None, truth_table=None, levels=None):
        # --- (Keep original forward implementation without the attention block) ---
        batch_size, seq_len, _, _ = x.shape
        x_flat = torch.flatten(x, 2, 3)

        if self.use_conditioning and truth_table is not None:
            tt_expanded = truth_table.unsqueeze(1).expand(-1, seq_len, -1)
            x_flat = torch.cat((x_flat, tt_expanded), dim=2)

        embedded_input = self.relu(self.linear_in(x_flat))

        if self.level_embedding is not None and levels is not None:
            # Add level embedding (same logic as attention version)
            if levels.shape[0] == batch_size and levels.shape[1] == seq_len:
                try:
                    clamped_levels = torch.clamp(levels, 0, self.max_level)
                    lvl_emb = self.level_embedding(clamped_levels)
                    embedded_input = embedded_input + lvl_emb
                except Exception as e:
                    print(f"Warning: Error adding level embedding in non-attention RNN: {e}")

        gru_input = embedded_input # Direct input to GRU
        target_padded_length = gru_input.shape[1]

        if x_lens is not None:
            x_lens_cpu = x_lens if isinstance(x_lens, torch.Tensor) else torch.tensor(x_lens)
            x_lens_cpu = x_lens_cpu.cpu()
            try:
                gru_input = pack_padded_sequence(gru_input, x_lens_cpu, batch_first=True, enforce_sorted=False)
            except RuntimeError as e: print(f"Error packing sequence in GraphLevelRNN: {e}")


        gru_output_packed, self.hidden = self.gru(gru_input, self.hidden)

        gru_output = gru_output_packed
        if x_lens is not None:
             try:
                gru_output, _ = pad_packed_sequence(gru_output_packed, batch_first=True, total_length=target_padded_length)
             except RuntimeError as e: print(f"Error unpacking sequence in GraphLevelRNN: {e}")



        final_output = gru_output
        if self.linear_out1:
            final_output = self.relu(self.linear_out1(final_output))
            final_output = self.linear_out2(final_output)


        return final_output



# --- EdgeLevelRNN and EdgeLevelMLP remain unchanged ---
# (Include their definitions from your src/model.py or the previous response here)
class EdgeLevelMLP(nn.Module):
    # --- MODIFIED __init__ ---
    def __init__(self, input_size, hidden_size, output_size, edge_feature_len=3,
                 use_conditioning=False, tt_size=None): # NEW args
        super().__init__()
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = use_conditioning
        self.tt_size = tt_size

        lin1_in_features = input_size # Input is hidden state from GraphLevelRNN
        # --- NEW: Conditioning Input ---
        if self.use_conditioning and self.tt_size is not None:
             lin1_in_features += self.tt_size
        # --- END NEW ---

        self.linear1 = nn.Linear(lin1_in_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size * edge_feature_len)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() # Keep for potential BCELoss case

    # --- MODIFIED forward ---
    def forward(self, x, return_logits=False, truth_table=None): # Add truth_table
        # x shape: [batch, seq_len, input_size (hidden_size from GraphRNN)]
        # truth_table shape: [batch, tt_size]

        # --- NEW: Concatenate truth table if conditioning ---
        if self.use_conditioning and truth_table is not None:
             tt_expanded = truth_table.unsqueeze(1).expand(-1, x.shape[1], -1)
             x_conditioned = torch.cat((x, tt_expanded), dim=2)
        else:
             x_conditioned = x
        # --- END NEW ---

        h = self.relu(self.linear1(x_conditioned)) # [batch, seq_len, hidden_size]
        out = self.linear2(h) # [batch, seq_len, output_size * edge_feature_len] (Logits)

        if not return_logits:
             # Apply sigmoid only if not returning logits (for BCELoss compatibility)
             # CrossEntropyLoss expects raw logits
             out = self.sigmoid(out)

        # Reshape output to separate edge features
        # [batch, seq_len, output_size, edge_feature_len]
        out_reshaped = torch.reshape(out, [out.shape[0], out.shape[1], -1, self.edge_feature_len])

        return out_reshaped

class EdgeLevelRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers,
                 edge_feature_len=3, # Default AIG edge features
                 use_conditioning=False, # Keep for flexibility
                 tt_size=None, # Truth table size for conditioning
                 tt_embedding_size=64): # Size of TT embedding if used
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = use_conditioning and (tt_size is not None)

        # --- Conditioning Setup ---
        self.tt_embedding = None
        gru_input_size = embedding_size
        if self.use_conditioning:
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size),
                nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size),
                nn.ReLU()
            )
            gru_input_size += tt_embedding_size
            print(f"INFO: EdgeLevelRNN using TT conditioning. GRU input size: {gru_input_size}")
        # --- End Conditioning Setup ---

        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        self.linear_out1 = nn.Linear(hidden_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid() # Keep for BCELoss compatibility if needed
        self.hidden = None

    def set_first_layer_hidden(self, h):
        if h.shape[-1] != self.hidden_size:
             raise ValueError(f"Hidden state dimension mismatch in set_first_layer_hidden. "
                              f"GraphRNN output ({h.shape[-1]}) != EdgeRNN hidden ({self.hidden_size})")
        if len(h.shape) == 3 and h.shape[0] > 1: h_first_layer = h[0:1, :, :]
        elif len(h.shape) == 2: h_first_layer = h.unsqueeze(0)
        elif len(h.shape) == 3 and h.shape[0] == 1: h_first_layer = h
        else: raise ValueError(f"Unexpected shape for hidden state h: {h.shape}")
        zeros = torch.zeros([self.num_layers-1, h_first_layer.shape[1], h_first_layer.shape[2]], device=h.device)
        self.hidden = torch.cat([h_first_layer, zeros], dim=0)

    def forward(self, x, x_lens=None, return_logits=False, truth_table=None):
        assert self.hidden is not None, "Hidden state not set for EdgeLevelRNN!"
        embedded_x = self.relu(self.linear_in(x))
        gru_input = embedded_x
        if self.use_conditioning and truth_table is not None and self.tt_embedding is not None:
             truth_table = truth_table.to(embedded_x.device)
             tt_emb = self.tt_embedding(truth_table)
             seq_len_edge = embedded_x.shape[1]
             tt_emb_expanded = tt_emb.unsqueeze(1).expand(-1, seq_len_edge, -1)
             gru_input = torch.cat((embedded_x, tt_emb_expanded), dim=2)
        target_padded_length = gru_input.shape[1]
        if x_lens is not None:
            x_lens_cpu = x_lens if isinstance(x_lens, torch.Tensor) else torch.tensor(x_lens)
            x_lens_cpu = x_lens_cpu.cpu()
            try: gru_input = pack_padded_sequence(gru_input, x_lens_cpu, batch_first=True, enforce_sorted=False)
            except RuntimeError as e: print(f"Error packing sequence in EdgeLevelRNN: {e}")
        gru_output_packed, self.hidden = self.gru(gru_input, self.hidden)
        gru_output = gru_output_packed
        if x_lens is not None:
            try: gru_output, _ = pad_packed_sequence(gru_output_packed, batch_first=True, total_length=target_padded_length)
            except RuntimeError as e: print(f"Error unpacking sequence in EdgeLevelRNN: {e}")
        out = self.relu(self.linear_out1(gru_output))
        logits = self.linear_out2(out)
        if not return_logits: return self.sigmoid(logits)
        else: return logits


# --- NEW LSTM-based Models ---

class GraphLevelLSTM(nn.Module):
    """ LSTM version of GraphLevelRNN """
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 output_size=None, edge_feature_len=3,
                 use_conditioning=False, tt_size=None,
                 max_level=None):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = use_conditioning
        self.tt_size = tt_size
        self.max_level = max_level

        lin_in_features = input_size * edge_feature_len
        if self.use_conditioning and self.tt_size is not None:
            lin_in_features += self.tt_size
        self.linear_in = nn.Linear(lin_in_features, embedding_size)
        self.relu = nn.ReLU()

        self.level_embedding = None
        if self.max_level is not None and self.max_level >= 0:
            self.level_embedding = nn.Embedding(self.max_level + 1, embedding_size)
            print(f"INFO: GraphLevelLSTM using level embedding up to level {self.max_level}")

        # --- USE LSTM ---
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # --- END LSTM ---
        self.linear_out1 = None
        self.linear_out2 = None
        if output_size:
            self.linear_out1 = nn.Linear(hidden_size, embedding_size)
            self.linear_out2 = nn.Linear(embedding_size, output_size)

        self.hidden = None # LSTM hidden is a tuple (h_n, c_n)

    def reset_hidden(self):
        self.hidden = None

    def forward(self, x, x_lens=None, truth_table=None, levels=None):
        batch_size, seq_len, _, _ = x.shape
        x_flat = torch.flatten(x, 2, 3)

        if self.use_conditioning and truth_table is not None:
            tt_expanded = truth_table.unsqueeze(1).expand(-1, seq_len, -1)
            x_flat = torch.cat((x_flat, tt_expanded), dim=2)

        embedded_input = self.relu(self.linear_in(x_flat))

        if self.level_embedding is not None and levels is not None:
            if levels.shape[0] == batch_size and levels.shape[1] == seq_len:
                try:
                    clamped_levels = torch.clamp(levels, 0, self.max_level)
                    lvl_emb = self.level_embedding(clamped_levels)
                    embedded_input = embedded_input + lvl_emb
                except Exception as e:
                    print(f"Warning: Error adding level embedding in GraphLevelLSTM: {e}")

        lstm_input = embedded_input
        target_padded_length = lstm_input.shape[1]

        if x_lens is not None:
            x_lens_cpu = x_lens if isinstance(x_lens, torch.Tensor) else torch.tensor(x_lens)
            x_lens_cpu = x_lens_cpu.cpu()
            try:
                lstm_input = pack_padded_sequence(lstm_input, x_lens_cpu, batch_first=True, enforce_sorted=False)
            except RuntimeError as e: print(f"Error packing sequence in GraphLevelLSTM: {e}")

        # --- Use LSTM ---
        # Pass self.hidden (which is None initially or the tuple from previous step)
        lstm_output_packed, self.hidden = self.lstm(lstm_input, self.hidden)
        # self.hidden is now the tuple (h_n, c_n)
        # --- End LSTM ---

        lstm_output = lstm_output_packed
        if x_lens is not None:
             try:
                # Unpack only the output sequence, hidden state is managed internally
                lstm_output, _ = pad_packed_sequence(lstm_output_packed, batch_first=True, total_length=target_padded_length)
             except RuntimeError as e: print(f"Error unpacking sequence in GraphLevelLSTM: {e}")



        final_output = lstm_output
        if self.linear_out1:
            final_output = self.relu(self.linear_out1(final_output))
            # LSTM output needs to feed into EdgeLevel model, so final projection might depend
            # on whether EdgeLevel model is RNN/LSTM or MLP
            if self.linear_out2:
                 final_output = self.linear_out2(final_output)


            # Return only the output sequence (used as input 'h' to edge models)
            # LSTM's hidden state (h_n, c_n) is stored in self.hidden
        return final_output


class GraphLevelAttentionLSTM(nn.Module):
    """ LSTM version of GraphLevelAttentionRNN """
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 output_size=None, edge_feature_len=3,
                 use_conditioning=False, tt_size=None,
                 max_level=None,
                 attention_heads=4,
                 attention_dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.edge_feature_len = edge_feature_len

        self.use_conditioning = use_conditioning
        self.tt_size = tt_size
        self.max_level = max_level

        lin_in_features = input_size * edge_feature_len
        if self.use_conditioning and self.tt_size is not None:
            lin_in_features += self.tt_size
        self.linear_in = nn.Linear(lin_in_features, embedding_size)
        self.relu = nn.ReLU()

        self.level_embedding = None
        if self.max_level is not None and self.max_level >= 0:
            self.level_embedding = nn.Embedding(self.max_level + 1, embedding_size)
            print(f"INFO: GraphLevelAttentionLSTM using level embedding up to level {self.max_level}")

        # --- Attention Layers (same as RNN version) ---
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(embedding_size)
        self.attention_dropout_layer = nn.Dropout(attention_dropout)
        # --- End Attention ---

        # --- USE LSTM ---
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # --- END LSTM ---

        self.linear_out1 = None
        self.linear_out2 = None
        if output_size:
            self.linear_out1 = nn.Linear(hidden_size, embedding_size)
            self.linear_out2 = nn.Linear(embedding_size, output_size)

        self.hidden = None # LSTM hidden is a tuple (h_n, c_n)

    def reset_hidden(self):
        self.hidden = None

    def forward(self, x, x_lens=None, truth_table=None, levels=None):
        batch_size, seq_len, _, _ = x.shape
        device = x.device

        x_flat = torch.flatten(x, 2, 3)

        if self.use_conditioning and truth_table is not None:
            tt_expanded = truth_table.unsqueeze(1).expand(-1, seq_len, -1)
            x_flat = torch.cat((x_flat, tt_expanded), dim=2)

        embedded_input = self.relu(self.linear_in(x_flat))

        if self.level_embedding is not None and levels is not None:
            if levels.shape[0] == batch_size and levels.shape[1] == seq_len:
                try:
                    clamped_levels = torch.clamp(levels, 0, self.max_level)
                    lvl_emb = self.level_embedding(clamped_levels)
                    embedded_input = embedded_input + lvl_emb
                except Exception as e: print(f"Warning: Error adding level embedding: {e}")

        padding_mask = None
        if x_lens is not None:
            if not isinstance(x_lens, torch.Tensor):
                x_lens_tensor = torch.tensor(x_lens, dtype=torch.long, device=device)
            else:
                x_lens_tensor = x_lens.to(device)
            max_len_pad = seq_len
            indices_pad = torch.arange(max_len_pad, device=device).expand(batch_size, max_len_pad)
            padding_mask = indices_pad >= x_lens_tensor.unsqueeze(1)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

        attn_output, attn_weights = self.self_attention(
            query=embedded_input, key=embedded_input, value=embedded_input,
            key_padding_mask=padding_mask, attn_mask=causal_mask
        )
        attended_input = embedded_input + self.attention_dropout_layer(attn_output)
        attended_input_norm = self.attention_norm(attended_input)

        lstm_input = attended_input_norm
        target_padded_length = lstm_input.shape[1]

        if x_lens is not None:
            x_lens_cpu = x_lens if isinstance(x_lens, torch.Tensor) else torch.tensor(x_lens)
            x_lens_cpu = x_lens_cpu.cpu()
            try:
                lstm_input = pack_padded_sequence(lstm_input, x_lens_cpu, batch_first=True, enforce_sorted=False)
            except RuntimeError as e: print(f"Error packing sequence: {e}")

        # --- Use LSTM ---
        lstm_output_packed, self.hidden = self.lstm(lstm_input, self.hidden)
        # --- End LSTM ---

        lstm_output = lstm_output_packed
        if x_lens is not None:
             try:
                lstm_output, _ = pad_packed_sequence(lstm_output_packed, batch_first=True, total_length=target_padded_length)
             except RuntimeError as e: print(f"Error unpacking sequence: {e}")


        final_output = lstm_output
        if self.linear_out1:
            final_output = self.relu(self.linear_out1(final_output))
            if self.linear_out2:
                final_output = self.linear_out2(final_output)

        return final_output


class EdgeLevelLSTM(nn.Module):
    """ LSTM version of EdgeLevelRNN """
    def __init__(self, embedding_size, hidden_size, num_layers,
                 edge_feature_len=3,
                 use_conditioning=False,
                 tt_size=None,
                 tt_embedding_size=64):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = use_conditioning and (tt_size is not None)

        self.tt_embedding = None
        lstm_input_size = embedding_size # Renamed from gru_input_size
        if self.use_conditioning:
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size), nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size), nn.ReLU()
            )
            lstm_input_size += tt_embedding_size
            print(f"INFO: EdgeLevelLSTM using TT conditioning. LSTM input size: {lstm_input_size}")

        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()

        # --- USE LSTM ---
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # --- END LSTM ---

        self.linear_out1 = nn.Linear(hidden_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None # LSTM hidden is a tuple (h_n, c_n)

    def set_first_layer_hidden(self, h):
        # h comes from the node-level model (GraphLevelLSTM or GraphLevelAttentionLSTM)
        # It's the output sequence, typically shape [batch, seq_len, node_hidden_size]
        # We need the *final* hidden state h_n from the node model to initialize the edge model.
        # However, the current GRU implementation passes the *entire output sequence* h
        # and expects the *edge* model's hidden state size to match the *node* model's hidden size.
        # Let's keep that assumption for now, but it might be conceptually cleaner to pass
        # the actual final hidden state h_n from the node model if possible.

        # Adapting the GRU logic for LSTM:
        # We'll use the first layer of h (assuming h is shaped like GRU output)
        # as the initial h_0 for the LSTM, and initialize c_0 to zeros.
        if h.shape[-1] != self.hidden_size:
            raise ValueError(f"Hidden state dimension mismatch in set_first_layer_hidden. "
                             f"Node model output dim ({h.shape[-1]}) != EdgeLevelLSTM hidden ({self.hidden_size})")

        # Extract the first layer's state (like in GRU version)
        if len(h.shape) == 3 and h.shape[0] > 1: h_first_layer = h[0:1, :, :]
        elif len(h.shape) == 2: h_first_layer = h.unsqueeze(0)
        elif len(h.shape) == 3 and h.shape[0] == 1: h_first_layer = h
        else: raise ValueError(f"Unexpected shape for hidden state h: {h.shape}")

        # Create initial hidden state h_0
        h_0 = torch.cat([h_first_layer,
                         torch.zeros([self.num_layers - 1, h_first_layer.shape[1], h_first_layer.shape[2]], device=h.device)],
                        dim=0)
        # Create initial cell state c_0 (zeros)
        c_0 = torch.zeros_like(h_0)

        self.hidden = (h_0, c_0) # Store as tuple

    def forward(self, x, x_lens=None, return_logits=False, truth_table=None):
        assert self.hidden is not None, "Hidden state not set for EdgeLevelLSTM!"
        embedded_x = self.relu(self.linear_in(x))
        lstm_input = embedded_x

        if self.use_conditioning and truth_table is not None and self.tt_embedding is not None:
             truth_table = truth_table.to(embedded_x.device)
             tt_emb = self.tt_embedding(truth_table)
             seq_len_edge = embedded_x.shape[1]
             tt_emb_expanded = tt_emb.unsqueeze(1).expand(-1, seq_len_edge, -1)
             lstm_input = torch.cat((embedded_x, tt_emb_expanded), dim=2)

        target_padded_length = lstm_input.shape[1]
        if x_lens is not None:
            x_lens_cpu = x_lens if isinstance(x_lens, torch.Tensor) else torch.tensor(x_lens)
            x_lens_cpu = x_lens_cpu.cpu()
            try:
                lstm_input = pack_padded_sequence(lstm_input, x_lens_cpu, batch_first=True, enforce_sorted=False)
            except RuntimeError as e: print(f"Error packing sequence in EdgeLevelLSTM: {e}")

        # --- Use LSTM ---
        lstm_output_packed, self.hidden = self.lstm(lstm_input, self.hidden)
        # --- End LSTM ---

        lstm_output = lstm_output_packed
        if x_lens is not None:
            try:
                lstm_output, _ = pad_packed_sequence(lstm_output_packed, batch_first=True, total_length=target_padded_length)
            except RuntimeError as e: print(f"Error unpacking sequence in EdgeLevelLSTM: {e}")

        out = self.relu(self.linear_out1(lstm_output))
        logits = self.linear_out2(out)

        if not return_logits: return self.sigmoid(logits)
        else: return logits


class EdgeLevelAttentionLSTM(nn.Module):
    """ LSTM version of EdgeLevelAttentionRNN """
    def __init__(self, embedding_size, hidden_size, num_layers,
                 edge_feature_len=3,
                 attention_heads=4,
                 attention_dropout=0.1,
                 use_conditioning=False,
                 tt_size=None,
                 tt_embedding_size=64):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.edge_feature_len = edge_feature_len
        self.embedding_size = embedding_size
        self.use_conditioning = use_conditioning and (tt_size is not None)

        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()

        self.tt_embedding = None
        attn_input_dim = embedding_size
        if self.use_conditioning:
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size), nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size), nn.ReLU()
            )
            print(f"INFO: EdgeLevelAttentionLSTM optionally using TT conditioning.")

        # --- Attention Layers ---
        self.self_attention = nn.MultiheadAttention(
            embed_dim=attn_input_dim, num_heads=attention_heads,
            dropout=attention_dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(attn_input_dim)
        self.attention_dropout_layer = nn.Dropout(attention_dropout)
        # --- End Attention ---

        # --- USE LSTM ---
        lstm_input_size = attn_input_dim # Input to LSTM is output of attention
        if self.use_conditioning:
            lstm_input_size += tt_embedding_size # Add TT embedding dim if conditioning

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        # --- END LSTM ---

        self.linear_out1 = nn.Linear(hidden_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None # LSTM hidden is tuple (h_n, c_n)

    def set_first_layer_hidden(self, h):
        # Same logic as EdgeLevelLSTM
        if h.shape[-1] != self.hidden_size:
            raise ValueError(f"Hidden state dimension mismatch. Node model output dim ({h.shape[-1]}) != EdgeLevelAttentionLSTM hidden ({self.hidden_size})")

        if len(h.shape) == 3 and h.shape[0] > 1: h_first_layer = h[0:1, :, :]
        elif len(h.shape) == 2: h_first_layer = h.unsqueeze(0)
        elif len(h.shape) == 3 and h.shape[0] == 1: h_first_layer = h
        else: raise ValueError(f"Unexpected shape for hidden state h: {h.shape}")

        h_0 = torch.cat([h_first_layer,
                         torch.zeros([self.num_layers - 1, h_first_layer.shape[1], h_first_layer.shape[2]], device=h.device)],
                        dim=0)
        c_0 = torch.zeros_like(h_0)
        self.hidden = (h_0, c_0)

    def forward(self, x, x_lens=None, return_logits=False, truth_table=None):
        assert self.hidden is not None, "Hidden state not set for EdgeLevelAttentionLSTM!"
        device = x.device

        embedded_x = self.relu(self.linear_in(x))
        batch_size, seq_len_edge, _ = embedded_x.shape

        padding_mask = None
        if x_lens is not None:
            if not isinstance(x_lens, torch.Tensor):
                x_lens_tensor = torch.tensor(x_lens, dtype=torch.long, device=device)
            else:
                x_lens_tensor = x_lens.to(device)
            max_len = seq_len_edge
            indices = torch.arange(max_len, device=device).expand(batch_size, max_len)
            padding_mask = indices >= x_lens_tensor.unsqueeze(1)

        attn_output, _ = self.self_attention(
            query=embedded_x, key=embedded_x, value=embedded_x,
            key_padding_mask=padding_mask
        )
        attended_edges = embedded_x + self.attention_dropout_layer(attn_output)
        attended_edges_norm = self.attention_norm(attended_edges)

        lstm_input = attended_edges_norm
        if self.use_conditioning and truth_table is not None and self.tt_embedding is not None:
            truth_table = truth_table.to(lstm_input.device)
            tt_emb = self.tt_embedding(truth_table)
            tt_emb_expanded = tt_emb.unsqueeze(1).expand(-1, seq_len_edge, -1)
            lstm_input = torch.cat((lstm_input, tt_emb_expanded), dim=2)

        target_padded_length = lstm_input.shape[1]
        if x_lens is not None:
            x_lens_cpu = x_lens if isinstance(x_lens, torch.Tensor) else torch.tensor(x_lens)
            x_lens_cpu = x_lens_cpu.cpu()
            try:
                lstm_input = pack_padded_sequence(lstm_input, x_lens_cpu, batch_first=True, enforce_sorted=False)
            except RuntimeError as e: print(f"Error packing sequence in EdgeLevelAttentionLSTM: {e}")

        # --- Use LSTM ---
        lstm_output_packed, self.hidden = self.lstm(lstm_input, self.hidden)
        # --- End LSTM ---

        lstm_output = lstm_output_packed
        if x_lens is not None:
            try:
                lstm_output, _ = pad_packed_sequence(lstm_output_packed, batch_first=True, total_length=target_padded_length)
            except RuntimeError as e: print(f"Error unpacking sequence in EdgeLevelAttentionLSTM: {e}")

        out = self.relu(self.linear_out1(lstm_output))
        logits = self.linear_out2(out)

        if not return_logits: return self.sigmoid(logits)
        else: return logits

