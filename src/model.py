

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import Optional, Tuple, List, Union



class GraphLevelRNN(nn.Module):
    # --- MODIFIED FOR CONFIGURABLE RNN TYPE ---
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 output_size=None, edge_feature_len=1,
                 predict_node_types=False, num_node_types=None,
                 use_conditioning=False, tt_size=None,
                 max_level=None,
                 rnn_type='gru'): # ADDED rnn_type argument (defaulting to GRU)
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
        self.rnn_type = rnn_type.lower() # Store the type

        # --- Keep input layer logic ---
        lin_in_features = input_size * edge_feature_len
        if self.use_conditioning and self.tt_size is not None:
            lin_in_features += self.tt_size
        self.linear_in = nn.Linear(lin_in_features, embedding_size)
        self.relu = nn.ReLU()

        # --- Keep Level Positional Embedding ---
        self.level_embedding = None
        if self.max_level is not None and self.max_level >= 0:
             self.level_embedding = nn.Embedding(self.max_level + 1, embedding_size)
             print(f"INFO: GraphLevelRNN using level embedding up to level {self.max_level}")

        # --- MODIFIED: Conditionally create RNN layer ---
        if self.rnn_type == 'lstm':
            print("INFO: GraphLevelRNN using LSTM.")
            self.rnn_layer = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                                     num_layers=num_layers, batch_first=True)
        elif self.rnn_type == 'gru':
            print("INFO: GraphLevelRNN using GRU.")
            self.rnn_layer = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                                    num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Choose 'lstm' or 'gru'.")
        # --- END MODIFIED ---

        # --- Keep optional output layers ---
        if output_size:
            self.linear_out1 = nn.Linear(hidden_size, embedding_size)
            self.linear_out2 = nn.Linear(embedding_size, output_size)
        else:
            self.linear_out1 = None
            self.linear_out2 = None

        # --- Keep optional node type predictor ---
        self.node_type_predictor = None
        if self.predict_node_types:
            if num_node_types is None:
                raise ValueError("num_node_types must be specified if predict_node_types is True")
            self.node_type_predictor = nn.Linear(hidden_size, num_node_types)

        self.hidden_state = None # Stores hidden state (or (h,c) tuple for LSTM)

    def reset_hidden(self):
        """Resets the hidden state to None (works for GRU and LSTM)."""
        self.hidden_state = None

    def forward(self, x, x_lens=None, truth_table=None, levels=None):
        batch_size, seq_len, _, _ = x.shape

        # --- Keep input processing and level embedding addition ---
        # ... (Flatten, Conditioning, Linear, ReLU, Level Embedding Addition) ...
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
                 except Exception as e: print(f"Warning: Error adding level embedding: {e}")
            else: print(f"Warning: Shape mismatch skipping level embedding.")
        # --- End kept section ---

        # Packing sequence
        target_padded_length = None
        if x_lens is not None:
            target_padded_length = x.shape[1]
            x = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)

        # --- MODIFIED: Use self.rnn_layer ---
        # Pass self.hidden_state (None or tuple/tensor)
        rnn_output, self.hidden_state = self.rnn_layer(x, self.hidden_state)
        x = rnn_output # Use the output sequence
        # --- END MODIFIED ---

        # Unpacking sequence
        if x_lens is not None:
            x, _ = pad_packed_sequence(x, batch_first=True, total_length=target_padded_length)

        # --- Keep node type prediction and optional output projection ---
        node_type_logits = None
        if self.predict_node_types and self.node_type_predictor is not None:
            node_type_logits = self.node_type_predictor(x)
        if self.linear_out1:
            x = self.relu(self.linear_out1(x))
            x = self.linear_out2(x)

        # Return based on node type prediction flag
        if self.predict_node_types:
            return x, node_type_logits
        else:
            return x


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
    Edge-Level RNN (LSTM or GRU) that can be optionally conditioned on truth tables.
    """
    # --- MODIFIED FOR CONFIGURABLE RNN TYPE ---
    def __init__(
            self,
            embedding_size: int,
            hidden_size: int,
            num_layers: int,
            tt_size: Optional[int] = None,
            edge_feature_len: int = 1,
            tt_embedding_size: int = 64,
            rnn_type: str = 'gru'): # ADDED rnn_type argument
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = tt_size is not None
        self.rnn_type = rnn_type.lower() # Store RNN type

        # --- Conditioning logic (calculates input_size_for_rnn) ---
        input_size_for_rnn = embedding_size # Base size
        if self.use_conditioning:
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size), nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size), nn.ReLU()
            )
            input_size_for_rnn += tt_embedding_size # Add TT embedding size
        # --- End conditioning ---

        # Input embedding layer
        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()

        # --- MODIFIED: Conditionally create RNN layer ---
        if self.rnn_type == 'lstm':
            print("INFO: EdgeLevelRNN using LSTM.")
            self.rnn_layer = nn.LSTM(
                input_size=input_size_for_rnn,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            print("INFO: EdgeLevelRNN using GRU.")
            self.rnn_layer = nn.GRU(
                input_size=input_size_for_rnn,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported rnn_type for EdgeLevelRNN: {rnn_type}. Choose 'lstm' or 'gru'.")
        # --- END MODIFIED ---

        # --- Output layers (Sigmoid REMOVED as recommended before) ---
        self.linear_out1 = nn.Linear(hidden_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        # self.sigmoid = nn.Sigmoid() # REMOVED

        self.hidden_state = None # Stores hidden state (or (h, c) tuple for LSTM)

    # --- MODIFIED: set_first_layer_hidden for LSTM/GRU ---
    def set_first_layer_hidden(self, h_graph_rnn: torch.Tensor):
        """
        Sets the initial hidden state(s) for the first RNN layer.
        Handles both LSTM (h, c) and GRU (h).
        """
        # Ensure h_graph_rnn has shape [1, batch_size, hidden_size]
        if len(h_graph_rnn.shape) == 2:
            h_graph_rnn = h_graph_rnn.unsqueeze(0) # Add layer dim

        # Prepare zero hidden states for remaining layers
        h_zeros = torch.zeros(
            [self.num_layers - 1, h_graph_rnn.shape[1], self.hidden_size], # Use self.hidden_size
            device=h_graph_rnn.device
        )
        # Combine for initial hidden state 'h'
        h_init = torch.cat([h_graph_rnn, h_zeros], dim=0) # Shape: [num_layers, batch, hidden]

        if self.rnn_type == 'lstm':
            # Create initial cell state (zeros) for LSTM
            c_init = torch.zeros_like(h_init)
            self.hidden_state = (h_init, c_init) # Store as tuple (h, c)
        else: # GRU
            self.hidden_state = h_init # Store only h
    # --- END MODIFIED ---

    # --- MODIFIED: forward ---
    def forward(self, x, x_lens=None, return_logits=False, truth_table=None): # return_logits less relevant now
        # Check hidden state is set appropriately
        if self.hidden_state is None:
             print("Warning: EdgeLevelRNN hidden state not set. RNN layer will initialize to zeros.")
        elif self.rnn_type == 'lstm' and not isinstance(self.hidden_state, tuple):
             print("Warning: EdgeLevelRNN hidden state is not a tuple for LSTM. Resetting.")
             self.hidden_state = None
        elif self.rnn_type == 'gru' and isinstance(self.hidden_state, tuple):
             print("Warning: EdgeLevelRNN hidden state is a tuple for GRU. Resetting.")
             self.hidden_state = None

        # --- Keep conditioning and input embedding ---
        # ... (conditioning logic as before, uses self.use_conditioning) ...
        if self.use_conditioning and truth_table is not None:
             tt_emb = self.tt_embedding(truth_table) # Calculate TT embedding
             tt_expanded = tt_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
             x = torch.cat((x, tt_expanded), dim=2) # Concatenate features

        x = self.relu(self.linear_in(x))

        # Pack sequence
        target_padded_length = None
        if x_lens is not None:
            target_padded_length = x.shape[1]
            x = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)

        # --- MODIFIED: Use self.rnn_layer ---
        rnn_output, self.hidden_state = self.rnn_layer(x, self.hidden_state)
        x = rnn_output
        # --- END MODIFIED ---

        # Unpack sequence
        if x_lens is not None:
            x, _ = pad_packed_sequence(x, batch_first=True, total_length=target_padded_length)

        # --- Keep output layers (Sigmoid Removed) ---
        x = self.relu(self.linear_out1(x))
        x = self.linear_out2(x) # Output raw logits

        return x
    # --- END MODIFIED ---