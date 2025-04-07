


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F # Needed for softmax
import math # Needed for sqrt in attention
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
            embedding_size: int,
            hidden_size: int,
            num_layers: int,
            edge_feature_len: int = 1,
            node_hidden_size: int = None,
            attention_heads: int = 4,
            tt_size: Optional[int] = None,
            tt_embedding_size: int = 64
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = tt_size is not None

        if node_hidden_size is None:
             raise ValueError("node_hidden_size (GraphLevelRNN's hidden_size) must be provided for attention.")
        self.node_hidden_size = node_hidden_size
        self.attention_heads = attention_heads

        attention_embed_dim = self.hidden_size
        self.attention = nn.MultiheadAttention(embed_dim=attention_embed_dim,
                                               num_heads=attention_heads,
                                               kdim=self.node_hidden_size,
                                               vdim=self.node_hidden_size,
                                               batch_first=True) # batch_first=True

        combined_feature_size = self.hidden_size + attention_embed_dim
        self.linear_combine = nn.Linear(combined_feature_size, embedding_size)

        gru_input_size = embedding_size
        if self.use_conditioning:
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size), nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size), nn.ReLU()
            )
            print("Warning: TT Conditioning and Node Attention used together in EdgeLevelRNN - interaction might be complex.")

        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear_out1 = nn.Linear(embedding_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid()

        self.hidden = None

    def set_first_layer_hidden(self, h: torch.Tensor):
        if h.shape[-1] != self.hidden_size:
             print(f"Warning: GraphLevelRNN output size ({h.shape[-1]}) doesn't match EdgeLevelRNN hidden size ({self.hidden_size}). Hidden state init might be incorrect.")
             if h.shape[-1] > self.hidden_size:
                 h = h[..., :self.hidden_size]

        zeros = torch.zeros([self.num_layers - 1, h.shape[-2], self.hidden_size], device=h.device)
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        if h.shape[0] != 1:
             h = h.permute(1,0,2)[:1,:,:]

        self.hidden = torch.cat([h, zeros], dim=0)


    # --- MODIFIED forward signature and attention call ---
    def forward(self, x, prev_node_hiddens, attn_mask: Optional[torch.Tensor] = None, x_lens=None, return_logits=False,
                truth_table=None):
        """
        Forward pass for the EdgeLevelRNN with attention.

        Args:
            x (Tensor): Input sequence of edge features.
                        Shape: [total_steps, edge_feature_len] for PackedSequence data
            prev_node_hiddens (Tensor): Sequence of hidden states from GraphLevelRNN for previous nodes.
                                       Shape: [total_steps, num_prev_nodes, node_hidden_size]
            attn_mask (Tensor, optional): Mask for attention mechanism. True values are masked out.
                                        Shape: [total_steps, num_prev_nodes]
            x_lens (Tensor, optional): Lengths of the edge sequences in x. Shape: [batch_size]
            return_logits (bool): If True, return raw scores before final activation.
            truth_table (Tensor, optional): Truth table for conditioning (if enabled).

        Returns:
            Tensor: Predicted edge features (probabilities or logits).
                    Shape: [total_steps, edge_feature_len]
        """
        if self.hidden is None:
            raise AssertionError("Hidden state not set! Call set_first_layer_hidden first.")
        if prev_node_hiddens is None:
            raise ValueError("prev_node_hiddens must be provided for attention.")

        # Handle packed sequence data - x shape will be [total_steps, edge_feature_len]
        # Reshape to add a sequence dimension of 1
        if len(x.shape) == 2:
            # For packed sequence data: [total_steps, edge_feature_len]
            total_steps, feat_len = x.shape
            x = x.view(total_steps, 1, feat_len)  # Add sequence length dim of 1
        elif len(x.shape) == 3:
            # Already in the expected shape
            total_steps, edge_seq_len, feat_len = x.shape
        else:
            # Handle 4D input (batch, seq_len, m, feat) - reshape to 3D
            batch_size, edge_seq_len, m, feat_len = x.shape
            x = x.view(-1, edge_seq_len, m * feat_len)  # Flatten m and feat dimensions
            total_steps = x.shape[0]

        embedded_x = self.relu(self.linear_in(x))  # Shape: [total_steps, edge_seq_len, embedding_size]

        outputs = []
        current_hidden = self.hidden  # Shape: [num_layers, batch, hidden_size]

        for t in range(embedded_x.shape[1]):  # Iterate over the sequence length dimension
            gru_input = embedded_x[:, t:t + 1, :]
            gru_output_t, current_hidden = self.gru(gru_input, current_hidden)

            query = current_hidden[-1:, :, :].permute(1, 0, 2)  # Shape: [total_steps, 1, hidden_size]
            key_value = prev_node_hiddens  # Shape: [total_steps, num_prev_nodes, node_hidden_size]

            # Prepare attention mask
            prepared_attn_mask = None
            if attn_mask is not None:
                prepared_attn_mask = attn_mask.unsqueeze(1)  # [total_steps, 1, num_prev_nodes]

            try:
                attn_output, _ = self.attention(
                    query=query,
                    key=key_value,
                    value=key_value,
                    attn_mask=prepared_attn_mask,
                    need_weights=False
                )
            except Exception as e:
                print(f"Error during attention calculation: {e}")
                print(f"Query shape: {query.shape}, Key/Value shape: {key_value.shape}")
                if prepared_attn_mask is not None:
                    print(f"Mask shape: {prepared_attn_mask.shape}")
                attn_output = torch.zeros_like(query)

            combined = torch.cat((gru_output_t, attn_output), dim=2)
            projected_output = self.relu(self.linear_combine(combined))

            out1 = self.relu(self.linear_out1(projected_output))
            out2 = self.linear_out2(out1)

            outputs.append(out2)

        final_output = torch.cat(outputs, dim=1)

        # For packed sequence data, we need to reshape back
        if final_output.shape[1] == 1:  # If we processed sequence length 1
            final_output = final_output.squeeze(1)  # Remove the sequence length dimension

        # Reshape to match expected output if needed
        if len(x.shape) == 4 and len(final_output.shape) == 3:
            # Need to reshape back to [batch, seq, m, feat]
            batch_size, seq_len, flattened_dim = final_output.shape
            m = flattened_dim // self.edge_feature_len
            final_output = final_output.view(batch_size, seq_len, m, self.edge_feature_len)

        if not return_logits:
            final_output = self.sigmoid(final_output)

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
