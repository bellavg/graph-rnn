

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import Optional, Tuple, List, Union



import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import Optional, Tuple, List, Union

class GraphLevelRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 output_size=None, edge_feature_len=1,
                 predict_node_types=False, num_node_types=None,
                 use_conditioning=False, tt_size=None,
                 max_level=None): # ADDED max_level argument
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.edge_feature_len = edge_feature_len
        self.predict_node_types = predict_node_types
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
            print(f"INFO: GraphLevelRNN using level embedding up to level {self.max_level}")

        # --- MODIFIED: Changed GRU to LSTM ---
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        # --- END MODIFICATION ---

        if output_size:
            self.linear_out1 = nn.Linear(hidden_size, embedding_size)
            self.linear_out2 = nn.Linear(embedding_size, output_size)
        else:
            self.linear_out1 = None
            self.linear_out2 = None

        self.node_type_predictor = None
        if self.predict_node_types:
            if num_node_types is None:
                raise ValueError("num_node_types must be specified if predict_node_types is True")
            self.node_type_predictor = nn.Linear(hidden_size, num_node_types)

        # --- MODIFIED: Hidden state is now a tuple (h, c) or None ---
        self.hidden = None
        # --- END MODIFICATION ---

    def reset_hidden(self):
        """Resets the hidden state (and cell state for LSTM) to None."""
        self.hidden = None # LSTM hidden state is a tuple (h, c), setting to None works

    def forward(self, x, x_lens=None, truth_table=None, levels=None):
        batch_size, seq_len, _, _ = x.shape
        x = torch.flatten(x, 2, 3)

        if self.use_conditioning and truth_table is not None:
            tt_expanded = truth_table.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat((x, tt_expanded), dim=2)

        x = self.relu(self.linear_in(x))

        if self.level_embedding is not None and levels is not None:
            if levels.shape[0] == batch_size and levels.shape[1] == seq_len:
                try:
                    clamped_levels = torch.clamp(levels, 0, self.max_level)
                    lvl_emb = self.level_embedding(clamped_levels)
                    x = x + lvl_emb
                except IndexError as e:
                    print(f"Warning: Index error during level embedding lookup (levels out of range? Max level: {self.max_level}): {e}")
                except Exception as e:
                    print(f"Warning: Error adding level embedding: {e}")
            else:
                 print(f"Warning: Shape mismatch between input x ({x.shape}) and levels ({levels.shape}). Skipping level embedding.")

        target_padded_length = None
        if x_lens is not None:
            target_padded_length = x.shape[1]
            # Ensure hidden state is detached if it's carried over explicitly
            # (though reset_hidden is typically called before each sequence)
            # if self.hidden is not None:
            #    self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
            x = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)

        # --- MODIFIED: Use LSTM and handle state tuple ---
        # self.hidden should be None or a tuple (h_0, c_0)
        output_packed, self.hidden = self.lstm(x, self.hidden)
        # self.hidden is now the tuple (h_n, c_n)
        # --- END MODIFICATION ---

        output_padded = output_packed # Initialize in case x_lens is None
        if x_lens is not None:
            output_padded, _ = pad_packed_sequence(output_packed, batch_first=True, total_length=target_padded_length)
            # Note: We assign the padded sequence back to 'x' in the original code's convention
            # For clarity, let's use 'output_padded'
            x = output_padded # Maintain consistency with original variable name 'x' for flow

        node_type_logits = None
        if self.predict_node_types and self.node_type_predictor is not None:
            node_type_logits = self.node_type_predictor(x)

        if self.linear_out1:
            x = self.relu(self.linear_out1(x))
            x = self.linear_out2(x)

        if self.predict_node_types:
            return x, node_type_logits
        else:
            return x # Return the sequence output


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


class EdgeLevelRNN(nn.Module):
    """
    Edge-Level RNN (Now LSTM) that can be optionally conditioned on truth tables.
    """
    def __init__(
            self,
            embedding_size: int,
            hidden_size: int,
            num_layers: int,
            tt_size: Optional[int] = None,
            edge_feature_len: int = 1,
            tt_embedding_size: int = 64
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = tt_size is not None

        if self.use_conditioning:
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size),
                nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size),
                nn.ReLU()
            )
            lstm_input_size = embedding_size + tt_embedding_size # GRU -> LSTM
        else:
            lstm_input_size = embedding_size # GRU -> LSTM

        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()

        # --- MODIFIED: Changed GRU to LSTM ---
        self.lstm = nn.LSTM( # Renamed from gru
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # --- END MODIFICATION ---

        self.linear_out1 = nn.Linear(hidden_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid()

        # --- MODIFIED: Hidden state is now a tuple (h, c) or None ---
        self.hidden = None # Initialize hidden state placeholder (h_0, c_0)
        # --- END MODIFICATION ---


    def set_first_layer_hidden(self, h: torch.Tensor):
        """
        Sets the hidden state of the first LSTM layer using 'h'. The cell state
        of the first layer and both hidden/cell states of other layers are zeroed.
        This should be set using the output of the graph-level RNN.

        Args:
            h: Hidden vector for the first layer, shape [batch, hidden_size] or [1, batch, hidden_size]
        """
        # Ensure h has the layer dimension (dim 0)
        if len(h.shape) == 2:
             # Unsqueeze adds dim 0: [1, batch_size, hidden_size]
            h = h.unsqueeze(0)
        elif len(h.shape) != 3 or h.shape[0] != 1:
            raise ValueError(f"Input h must have shape [batch, hidden_size] or [1, batch, hidden_size], got {h.shape}")

        # --- MODIFIED: Create initial (h_0, c_0) tuple for LSTM ---
        # h_0: Use input h for the first layer, zeros for others
        h_zeros = torch.zeros(self.num_layers - 1, h.shape[1], self.hidden_size, device=h.device)
        h_0 = torch.cat([h, h_zeros], dim=0) # Shape [num_layers, batch_size, hidden_size]

        # c_0: Initialize cell state to all zeros
        c_0 = torch.zeros_like(h_0) # Shape [num_layers, batch_size, hidden_size]

        self.hidden = (h_0, c_0) # Store the tuple
        # --- END MODIFICATION ---

    def forward(self, x, x_lens=None, return_logits=False, truth_table=None):
        if self.hidden is None:
             # This check is crucial as hidden state is expected to be set externally
             raise RuntimeError("EdgeLevelRNN hidden state not set! Call set_first_layer_hidden first.")

        batch_size = x.shape[0] # Assuming batch_first=True

        # Embed input edge features
        x_embedded = self.relu(self.linear_in(x)) # Shape: [batch, seq_len, embedding_size]

        # --- MODIFIED: Conditioning logic (apply before LSTM) ---
        if self.use_conditioning and truth_table is not None:
            if truth_table.shape[0] != batch_size:
                 raise ValueError(f"Batch size mismatch: input x ({batch_size}) vs truth_table ({truth_table.shape[0]})")
            tt_emb = self.tt_embedding(truth_table) # Shape: [batch, tt_embedding_size]
            # Expand tt_emb to match sequence length for concatenation
            tt_expanded = tt_emb.unsqueeze(1).expand(-1, x.shape[1], -1) # Shape: [batch, seq_len, tt_embedding_size]
            lstm_input = torch.cat((x_embedded, tt_expanded), dim=2) # Concatenated input
        else:
            lstm_input = x_embedded # Original embedded input
        # --- END MODIFICATION ---

        # Pack data if lengths are provided
        packed_input = lstm_input # Initialize
        if x_lens is not None:
             # Ensure hidden state tuple components are detached if needed, though set_first_layer_hidden handles this
             # self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
            packed_input = pack_padded_sequence(lstm_input, x_lens.cpu(), batch_first=True, enforce_sorted=False)

        # --- MODIFIED: Use LSTM and handle state tuple ---
        # self.hidden must be the (h_0, c_0) tuple set by set_first_layer_hidden
        output_packed, self.hidden = self.lstm(packed_input, self.hidden)
        # self.hidden is now the final (h_n, c_n) tuple
        # --- END MODIFICATION ---

        # Unpack (reintroduces padding)
        output_padded = output_packed # Initialize in case x_lens is None
        if x_lens is not None:
            output_padded, _ = pad_packed_sequence(output_packed, batch_first=True, total_length=x.shape[1]) # Use input seq len for total_length

        # Apply output layers
        x_out = self.relu(self.linear_out1(output_padded))  # [batch, seq_len, embedding_size]
        x_out = self.linear_out2(x_out)  # [batch, seq_len, edge_feature_len]

        if not return_logits:
            x_out = self.sigmoid(x_out) # Apply sigmoid if probabilities are needed

        return x_out