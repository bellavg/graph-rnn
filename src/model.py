import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import Optional, Tuple, List, Union


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import Optional, Tuple, List, Union


class GraphLevelRNN(nn.Module):
    # --- MODIFIED __init__ ---
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 output_size=None, edge_feature_len=1,
                 predict_node_types=False, num_node_types=None, # NEW args
                 use_conditioning=False, tt_size=None):       # NEW args
        super().__init__()
        self.input_size = input_size
        self.edge_feature_len = edge_feature_len
        self.predict_node_types = predict_node_types
        self.use_conditioning = use_conditioning
        self.tt_size = tt_size

        # Adjust input linear layer if conditioning is used
        lin_in_features = input_size * edge_feature_len
        if self.use_conditioning and self.tt_size is not None:
            lin_in_features += self.tt_size # Add truth table size to input features

        self.linear_in = nn.Linear(lin_in_features, embedding_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        # Optional output layer for original RNN edge model connection
        if output_size:
            self.linear_out1 = nn.Linear(hidden_size, embedding_size)
            self.linear_out2 = nn.Linear(embedding_size, output_size)
        else:
            self.linear_out1 = None
            self.linear_out2 = None

        # --- NEW: Optional node type prediction head ---
        self.node_type_predictor = None
        if self.predict_node_types:
            if num_node_types is None:
                raise ValueError("num_node_types must be specified if predict_node_types is True")
            # Simple linear layer on top of GRU hidden state
            self.node_type_predictor = nn.Linear(hidden_size, num_node_types)
        # --- END NEW ---

        self.hidden = None


    def reset_hidden(self):
        """Resets the hidden state to 0."""
        # By setting to None, PyTorch will automatically use a zero tensor.
        self.hidden = None

    def forward(self, x, x_lens=None, truth_table=None):  # Add truth_table arg
        # Flatten edge features
        # x shape: [batch, seq_len, input_size, edge_feature_len]
        x = torch.flatten(x, 2, 3)  # [batch, seq_len, input_size * edge_feature_len]

        # --- NEW: Concatenate truth table if conditioning ---
        if self.use_conditioning and truth_table is not None:
            # truth_table shape [batch, tt_size] -> [batch, 1, tt_size]
            tt_expanded = truth_table.unsqueeze(1).expand(-1, x.shape[1], -1)
            # Concatenate along the feature dimension
            x = torch.cat((x, tt_expanded), dim=2)
        # --- END NEW ---

        x = self.relu(self.linear_in(x))  # [batch, seq_len, embedding_dim]

        target_padded_length = None
        if x_lens is not None:
            # Store the target sequence length *before* packing
            # This should match the padding length used for the target 'y' in train.py
            target_padded_length = x.shape[1]
            x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        x, self.hidden = self.gru(x, self.hidden)

        if x_lens is not None:
            # --- EDIT: Add total_length argument ---
            x, _ = pad_packed_sequence(x, batch_first=True, total_length=target_padded_length)

        # --- NEW: Node type prediction ---
        node_type_logits = None
        if self.predict_node_types and self.node_type_predictor is not None:
            node_type_logits = self.node_type_predictor(x)  # [batch, seq_len, num_node_types]
        # --- END NEW ---

        if self.linear_out1:
            x = self.relu(self.linear_out1(x))
            x = self.linear_out2(x)

        # Return both hidden state (or final projection) and node logits if predicting types
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
    Edge-Level RNN that can be optionally conditioned on truth tables for generating AIGs.
    """

    def __init__(
            self,
            embedding_size: int,
            hidden_size: int,
            num_layers: int,
            tt_size: Optional[int] = None,        # Argument for TT size
            edge_feature_len: int = 1,
            tt_embedding_size: int = 64         # Argument for TT embedding size
    ):
        """
        Initialize the Edge-Level RNN with optional truth table conditioning.

        Args:
            embedding_size: Size of the input embedding fed to the GRU for edge features
            hidden_size: Hidden size of the GRU
            num_layers: Number of GRU layers
            tt_size: Optional size of the truth table (e.g., 8*256). If None, no conditioning.
            edge_feature_len: Number of features associated with each edge.
            tt_embedding_size: Size of the truth table embedding when conditioning is used.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = tt_size is not None # Determine if conditioning is used

        # --- NEW: Truth table conditioning embedding ---
        if self.use_conditioning:
            # Define embedding layers for the truth table
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size),
                nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size),
                nn.ReLU()
            )
            # The input to the GRU will be the concatenation of edge embedding and TT embedding
            gru_input_size = embedding_size + tt_embedding_size
        else:
            # If not conditioning, GRU input is just the edge embedding size
            gru_input_size = embedding_size
        # --- END NEW ---

        # Layer to embed the input edge features
        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()

        # GRU layer now takes potentially combined input size
        self.gru = nn.GRU(
            input_size=gru_input_size, # Use the calculated size
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Output layers (remain the same)
        self.linear_out1 = nn.Linear(hidden_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None # Initialize hidden state placeholder


    def set_first_layer_hidden(self, h: torch.Tensor):
        """
        Sets the hidden state of the first GRU layer. The hidden state of all
        other layers will be reset to 0. This should be set to the output of
        the graph-level RNN.

        Args:
            h: Hidden vector of shape [batch, hidden_size]
        """
        # Prepare zero tensor for all layers except the first
        zeros = torch.zeros([self.num_layers - 1, h.shape[-2], h.shape[-1]], device=h.device)
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        self.hidden = torch.cat([h, zeros], dim=0)  # [num_layers, batch_size, hidden_size]

    def forward(self, x, x_lens=None, return_logits=False, truth_table=None): # Add truth_table
        assert self.hidden is not None, "Hidden state not set!"

        # --- NEW: Concatenate truth table if conditioning ---
        # x shape: [batch*nodes, edge_len, edge_feature_len]
        # truth_table shape: [batch*nodes, tt_size]
        if self.use_conditioning and truth_table is not None:
             tt_expanded = truth_table.unsqueeze(1).expand(-1, x.shape[1], -1)
             x_conditioned = torch.cat((x, tt_expanded), dim=2)
        else:
             x_conditioned = x
        # --- END NEW ---

        x = self.relu(self.linear_in(x_conditioned)) # [batch*nodes, edge_len, embedding_size]

        # Pack data to increase efficiency
        if x_lens is not None:
            x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        # Process through GRU
        x, self.hidden = self.gru(x, self.hidden)  # [batch, seq_len, hidden_size]

        # Unpack (reintroduces padding)
        if x_lens is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)

        # Output layers
        x = self.relu(self.linear_out1(x))  # [batch, seq_len, embedding_size]
        x = self.linear_out2(x)  # [batch, seq_len, edge_feature_len]
        if not return_logits:
            x = self.sigmoid(x)
        return x