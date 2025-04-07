

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import Optional, Tuple, List, Union



class GraphLevelRNN(nn.Module):
    # --- MODIFIED __init__ ---
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 output_size=None, edge_feature_len=1,
                 predict_node_types=False, num_node_types=None,
                 use_conditioning=False, tt_size=None,
                 max_level=None): # ADDED max_level argument
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size # Store for level embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.edge_feature_len = edge_feature_len
        self.predict_node_types = predict_node_types
        self.use_conditioning = use_conditioning
        self.tt_size = tt_size
        self.max_level = max_level # ADDED

        # Adjust input linear layer if conditioning is used
        lin_in_features = input_size * edge_feature_len
        if self.use_conditioning and self.tt_size is not None:
            lin_in_features += self.tt_size

        self.linear_in = nn.Linear(lin_in_features, embedding_size)
        self.relu = nn.ReLU()

        # --- NEW: Level Positional Embedding ---
        self.level_embedding = None
        if self.max_level is not None and self.max_level >= 0:
             # Add 1 to max_level because levels are 0-indexed
             self.level_embedding = nn.Embedding(self.max_level + 1, embedding_size)
             print(f"INFO: GraphLevelRNN using level embedding up to level {self.max_level}")
        # --- END NEW ---

        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        # Optional output layer for original RNN edge model connection
        if output_size:
            self.linear_out1 = nn.Linear(hidden_size, embedding_size)
            self.linear_out2 = nn.Linear(embedding_size, output_size)
        else:
            self.linear_out1 = None
            self.linear_out2 = None

        # Optional node type prediction head
        self.node_type_predictor = None
        if self.predict_node_types:
            if num_node_types is None:
                raise ValueError("num_node_types must be specified if predict_node_types is True")
            self.node_type_predictor = nn.Linear(hidden_size, num_node_types)

        self.hidden = None # Stores hidden state across sequence steps

    def reset_hidden(self):
        self.hidden = None

    # --- MODIFIED forward ---
    def forward(self, x, x_lens=None, truth_table=None, levels=None, return_all_hiddens=False): # ADDED levels argument, return_all_hiddens
        # x shape: [batch, seq_len, input_size, edge_feature_len]
        # levels shape: [batch, seq_len] (should align with x's seq_len after SOS)

        batch_size, seq_len, _, _ = x.shape

        # Flatten edge features
        x = torch.flatten(x, 2, 3)  # [batch, seq_len, input_size * edge_feature_len]

        # Concatenate truth table if conditioning
        if self.use_conditioning and truth_table is not None:
            tt_expanded = truth_table.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat((x, tt_expanded), dim=2)

        # Initial embedding
        x = self.relu(self.linear_in(x))  # [batch, seq_len, embedding_dim]

        # --- Add Level Embedding ---
        if self.level_embedding is not None and levels is not None:
            if levels.shape[0] == batch_size and levels.shape[1] == seq_len:
                try:
                    clamped_levels = torch.clamp(levels, 0, self.max_level)
                    lvl_emb = self.level_embedding(clamped_levels) # Shape: [batch, seq_len, embedding_dim]
                    x = x + lvl_emb # Add level embedding
                except IndexError as e:
                    print(f"Warning: Index error during level embedding lookup: {e}")
                except Exception as e:
                    print(f"Warning: Error adding level embedding: {e}")
            else:
                 print(f"Warning: Shape mismatch x vs levels. Skipping level embedding.")
        # --- END Level Embedding ---

        target_padded_length = x.shape[1] # Store original padded length

        packed_input = x
        if x_lens is not None:
            packed_input = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)

        # Process through GRU
        # gru_output shape (packed): [Sum(lens), hidden_size]
        # gru_output shape (unpacked): [batch, seq_len, hidden_size]
        # self.hidden shape: [num_layers, batch, hidden_size]
        gru_output, self.hidden = self.gru(packed_input, self.hidden)

        # Unpack sequence
        if x_lens is not None:
            # gru_output shape: [batch, max_len_in_batch, hidden_size]
            # Pad to the original target_padded_length
            gru_output, _ = pad_packed_sequence(gru_output, batch_first=True, total_length=target_padded_length)
        else:
            # If no lengths provided, assume full sequences (e.g., during generation)
            # gru_output shape: [batch, seq_len, hidden_size]
            pass

        # --- Node type prediction ---
        node_type_logits = None
        if self.predict_node_types and self.node_type_predictor is not None:
            node_type_logits = self.node_type_predictor(gru_output)  # [batch, seq_len, num_node_types]
        # --- END Node type prediction ---

        # Output for EdgeRNN (either final hidden state or projected output)
        output_for_edge_rnn = gru_output
        if self.linear_out1:
            output_for_edge_rnn = self.relu(self.linear_out1(gru_output))
            output_for_edge_rnn = self.linear_out2(output_for_edge_rnn)

        # Prepare return value
        if return_all_hiddens:
            # Return all hidden states for attention in EdgeLevelRNN
            if self.predict_node_types:
                 return gru_output, node_type_logits # Return GRU output states directly
            else:
                 return gru_output
        else:
             # Return the final output (potentially projected) needed by EdgeRNN
             if self.predict_node_types:
                  return output_for_edge_rnn, node_type_logits
             else:
                  return output_for_edge_rnn



# ===========================================================
# EdgeLevelRNN with Attention
# ===========================================================
class EdgeLevelRNN(nn.Module):
    """
    Edge-Level RNN with attention over previous node hidden states.
    """
    def __init__(
            self,
            # Args for basic RNN functionality
            embedding_size: int,
            hidden_size: int,
            num_layers: int,
            edge_feature_len: int = 1,
            # Args for attention mechanism
            node_hidden_size: int = None, # Hidden size of the GraphLevelRNN outputting states
            attention_heads: int = 4,
            # Optional args (kept for compatibility, but attention usually replaces need for TT conditioning here)
            tt_size: Optional[int] = None,
            tt_embedding_size: int = 64
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size # Hidden size of this EdgeLevelRNN's GRU
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = tt_size is not None

        # --- Attention Setup ---
        if node_hidden_size is None:
             raise ValueError("node_hidden_size (GraphLevelRNN's hidden_size) must be provided for attention.")
        self.node_hidden_size = node_hidden_size
        self.attention_heads = attention_heads

        # Use PyTorch's MultiheadAttention layer
        # The GRU hidden state will be the query.
        # The previous GraphLevelRNN hidden states will be the key and value.
        # Ensure embed_dim is compatible. Here, query=edge_hidden, key/value=node_hidden
        # We'll project node_hidden to edge_hidden dim if they differ, or use edge_hidden for all
        attention_embed_dim = self.hidden_size # Using edge RNN hidden size for attention internal dim
        self.attention = nn.MultiheadAttention(embed_dim=attention_embed_dim,
                                               num_heads=attention_heads,
                                               kdim=self.node_hidden_size, # Key dim = node hidden
                                               vdim=self.node_hidden_size, # Value dim = node hidden
                                               batch_first=True) # Crucial: input format is [batch, seq, feature]

        # Linear layer to combine GRU output and attention context
        # Input size = GRU hidden + Attention output (which is attention_embed_dim)
        combined_feature_size = self.hidden_size + attention_embed_dim
        self.linear_combine = nn.Linear(combined_feature_size, embedding_size) # Project down to embedding size


        # --- Original Layers ---
        gru_input_size = embedding_size # Input to GRU is just embedded edge features now
        # Incorporate TT conditioning if needed (though maybe redundant with attention)
        if self.use_conditioning:
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size), nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size), nn.ReLU()
            )
            # If conditioning, GRU input needs to accommodate it too
            # This part becomes complex: maybe condition the *initial* hidden state instead?
            # For now, let's assume attention replaces TT conditioning need here.
            print("Warning: TT Conditioning and Node Attention used together in EdgeLevelRNN - interaction might be complex.")

        self.linear_in = nn.Linear(edge_feature_len, embedding_size) # Embed input edge type
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Output layers now take the combined features projection
        self.linear_out1 = nn.Linear(embedding_size, embedding_size) # Adjusted input size
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid() # Keep sigmoid for potential BCELoss

        self.hidden = None # Placeholder for GRU hidden state

    def set_first_layer_hidden(self, h: torch.Tensor):
        """
        Sets the hidden state of the first GRU layer using GraphLevelRNN's output.
        """
        # Ensure h matches the EdgeLevelRNN's hidden size if different from node_hidden_size
        # This might require a projection layer in GraphLevelRNN or here if sizes mismatch
        if h.shape[-1] != self.hidden_size:
             # Example: Add a projection if GraphLevelRNN output size != EdgeLevelRNN hidden size
             # This linear layer should be defined in __init__
             # h = self.initial_hidden_projection(h)
             print(f"Warning: GraphLevelRNN output size ({h.shape[-1]}) doesn't match EdgeLevelRNN hidden size ({self.hidden_size}). Hidden state init might be incorrect.")
             # Simple fix: Use only the first 'hidden_size' features if node_hidden > edge_hidden
             if h.shape[-1] > self.hidden_size:
                 h = h[..., :self.hidden_size]
             # Padding if node_hidden < edge_hidden? More complex. Assume sizes match for now.


        zeros = torch.zeros([self.num_layers - 1, h.shape[-2], self.hidden_size], device=h.device) # Use self.hidden_size
        if len(h.shape) == 2:
            h = h.unsqueeze(0) # Add layer dimension if needed
        # Ensure h is correctly shaped [1, batch, hidden_size] for the first layer
        if h.shape[0] != 1:
             h = h.permute(1,0,2)[:1,:,:] # Take only the last layer's hidden if multi-layer GRU in GraphLevelRNN

        self.hidden = torch.cat([h, zeros], dim=0) # [num_layers, batch, hidden_size]


    def forward(self, x, prev_node_hiddens, x_lens=None, return_logits=False, truth_table=None):
        """
        Forward pass for the EdgeLevelRNN with attention.

        Args:
            x (Tensor): Input sequence of edge features (e.g., SOS token followed by previous edge types).
                        Shape: [batch_or_total_nodes, edge_seq_len, edge_feature_len]
            prev_node_hiddens (Tensor): Sequence of hidden states from GraphLevelRNN for *previous* nodes.
                                        Shape: [batch_or_total_nodes, num_prev_nodes, node_hidden_size]
                                        Note: num_prev_nodes might vary or need padding/masking.
            x_lens (Tensor, optional): Lengths of the edge sequences in x. Shape: [batch_or_total_nodes]
            return_logits (bool): If True, return raw scores before final activation.
            truth_table (Tensor, optional): Truth table for conditioning (if enabled).

        Returns:
            Tensor: Predicted edge features (probabilities or logits).
                    Shape: [batch_or_total_nodes, edge_seq_len, edge_feature_len]
        """
        if self.hidden is None:
             raise AssertionError("Hidden state not set! Call set_first_layer_hidden first.")
        if prev_node_hiddens is None:
             raise ValueError("prev_node_hiddens must be provided for attention.")

        batch_size, edge_seq_len, _ = x.shape
        num_prev_nodes = prev_node_hiddens.shape[1]

        # Embed the input edge features (SOS, prev_edge, ...)
        embedded_x = self.relu(self.linear_in(x)) # Shape: [batch, edge_seq_len, embedding_size]

        # --- GRU Processing with Attention at each step ---
        # We need to run the GRU step-by-step to apply attention correctly.
        # The MultiheadAttention layer expects sequences, so we adapt.

        outputs = []
        current_hidden = self.hidden # Shape: [num_layers, batch, hidden_size]

        for t in range(edge_seq_len):
            # GRU input for this step: embedded edge feature
            gru_input = embedded_x[:, t:t+1, :] # Shape: [batch, 1, embedding_size]

            # Run GRU for one step
            # gru_output_t shape: [batch, 1, hidden_size]
            # current_hidden shape: [num_layers, batch, hidden_size]
            gru_output_t, current_hidden = self.gru(gru_input, current_hidden)

            # --- Attention Calculation ---
            # Query: Current GRU hidden state (last layer)
            # Use hidden state from the *last* layer of the GRU as the query
            query = current_hidden[-1:, :, :].permute(1, 0, 2) # Shape: [batch, 1, hidden_size]

            # Key/Value: Previous node hidden states from GraphLevelRNN
            key_value = prev_node_hiddens # Shape: [batch, num_prev_nodes, node_hidden_size]

            # Create attention mask if needed (e.g., for padding in prev_node_hiddens)
            # Assuming prev_node_hiddens is already appropriately masked or padded if necessary.
            # The length `num_prev_nodes` should correspond to the number of valid predecessors for edge t.
            attn_mask = None # Placeholder - depends on how prev_node_hiddens is structured

            # Apply multi-head attention
            # attn_output shape: [batch, 1, attention_embed_dim] (embed_dim = self.hidden_size here)
            # attn_weights shape: [batch, 1, num_prev_nodes]
            try:
                 attn_output, attn_weights = self.attention(query=query,
                                                            key=key_value,
                                                            value=key_value,
                                                            attn_mask=attn_mask,
                                                            need_weights=False) # Set True to debug weights
            except Exception as e:
                 print(f"Error during attention calculation: {e}")
                 # Handle error, maybe use zero context?
                 attn_output = torch.zeros_like(query)


            # --- Combine GRU output and Attention context ---
            # Concatenate along the feature dimension
            combined = torch.cat((gru_output_t, attn_output), dim=2) # Shape: [batch, 1, hidden_size + attention_embed_dim]

            # Project combined features
            projected_output = self.relu(self.linear_combine(combined)) # Shape: [batch, 1, embedding_size]


            # --- Final Output Layers for this step ---
            out1 = self.relu(self.linear_out1(projected_output)) # Shape: [batch, 1, embedding_size]
            out2 = self.linear_out2(out1)                      # Shape: [batch, 1, edge_feature_len]

            outputs.append(out2)

        # Concatenate results from all steps
        final_output = torch.cat(outputs, dim=1) # Shape: [batch, edge_seq_len, edge_feature_len]

        # Apply final activation if needed
        if not return_logits:
            final_output = self.sigmoid(final_output) # Use sigmoid for potential BCELoss

        return final_output




class EdgeLevelMLP(nn.Module):
    # --- MODIFIED __init__ ---
    def __init__(self, input_size, hidden_size, output_size, edge_feature_len=1,
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
