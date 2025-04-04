import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import Optional, Tuple, List, Union


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import Optional, Tuple, List, Union


class GraphLevelRNN(nn.Module):
    """
    A Graph-Level RNN that can optionally predict node types alongside edge connectivity for generating AIGs.
    """

    def __init__(
            self,
            input_size: int,
            embedding_size: int,
            hidden_size: int,
            num_layers: int,
            predict_node_types: bool = False,
            num_node_types: int = 4,  # ZERO, PI, AND, PO
            tt_size: Optional[int] = None,
            output_size: Optional[int] = None,
            edge_feature_len: int = 1,
            tt_embedding_size: int = 64
    ):
        """
        Initialize the Graph-Level RNN with node type prediction and optional truth table conditioning.

        Args:
            input_size: Length of the padded adjacency vector
            embedding_size: Size of the input embedding fed to the GRU
            hidden_size: Hidden size of the GRU
            num_layers: Number of GRU layers
            num_node_types: Number of different node types in the AIG (default: 4)
            tt_size: Optional size of the truth table (8x256 for 8 outputs, 2^8 input combinations)
                If None, no truth table conditioning is used
            output_size: Size of the final output. Set to None if the
                output layer should be skipped.
            edge_feature_len: Number of features associated with each edge.
                Default is 1 (i.e. scalar value 0/1 indicating whether the
                edge is set or not).
            tt_embedding_size: Size of the truth table embedding when conditioning is used
        """
        super().__init__()
        self.input_size = input_size
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = tt_size is not None
        self.predict_node_types = predict_node_types
        self.num_node_types = num_node_types

        # Adjacency vector processing
        self.linear_in = nn.Linear(input_size * edge_feature_len, embedding_size)
        self.relu = nn.ReLU()

        # Truth table conditioning (if enabled)
        if self.use_conditioning:
            # Assuming tt_size is total size (e.g., 8*256=2048 for 8 outputs with 2^8 combinations)
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size),
                nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size),
                nn.ReLU()
            )
            # Combined size with truth table embedding
            combined_size = embedding_size + tt_embedding_size
        else:
            combined_size = embedding_size

        # GRU takes combined embedding (or just adjacency embedding if no conditioning)
        self.gru = nn.GRU(
            input_size=combined_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Node type prediction layer (optional)
        if predict_node_types:
            self.node_type_predictor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_node_types)
            )
        else:
            self.node_type_predictor = None

        # Output layers for edge prediction
        if output_size:
            self.linear_out1 = nn.Linear(hidden_size, embedding_size)
            self.linear_out2 = nn.Linear(embedding_size, output_size)
        else:
            self.linear_out1 = None
            self.linear_out2 = None

        self.hidden = None

    def reset_hidden(self):
        """Resets the hidden state to 0."""
        # By setting to None, PyTorch will automatically use a zero tensor.
        self.hidden = None

    def forward(
            self,
            x: torch.Tensor,
            x_lens: Optional[List[int]] = None,
            truth_table: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional node type prediction and truth table conditioning.

        Args:
            x: Input tensor of shape [batch, seq_len, input_size, edge_feature_len].
                Should be an adjacency vector describing the connectivity of the
                previously generated node.
            x_lens: List of sequence lengths (i.e. number of graph nodes) of
                each batch entry. Should be on the CPU. This is used to pack
                the input to get rid of padding and increase efficiency.
                Set to 'None' to disable packing.
            truth_table: Optional tensor of shape [batch, tt_size] containing
                the desired output behavior of the circuit as flattened truth tables.
                Only used if model was initialized with tt_size.

        Returns:
            If predict_node_types is True:
                Tuple containing:
                - hidden: The final hidden state of the GRU of shape [batch, seq_len, hidden_size]
                - node_types: Node type predictions of shape [batch, seq_len, num_node_types]
            Otherwise:
                hidden: The final hidden state of the GRU of shape [batch, seq_len, hidden_size]
        """
        # Flatten edge features
        x = torch.flatten(x, 2, 3)  # [batch, seq_len, input_size * edge_feature_len]

        # Process adjacency vectors
        x = self.relu(self.linear_in(x))  # [batch, seq_len, embedding_dim]

        # If truth table conditioning is enabled and truth table is provided
        if self.use_conditioning and truth_table is not None:
            # Embed the truth table
            tt_embed = self.tt_embedding(truth_table)  # [batch, tt_embedding_size]

            # Expand truth table embedding to match sequence dimension
            # This broadcasts the truth table embedding to each step of the sequence
            tt_embed = tt_embed.unsqueeze(1).expand(-1, x.shape[1], -1)  # [batch, seq_len, tt_embedding_size]

            # Concatenate adjacency embedding with truth table embedding
            x = torch.cat([x, tt_embed], dim=2)  # [batch, seq_len, embedding_dim + tt_embedding_size]

        # Pack data to increase efficiency during training
        if x_lens is not None:
            x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        # Process through GRU
        x, self.hidden = self.gru(x, self.hidden)  # Packed [batch, seq_len, hidden_size]

        # Unpack (reintroduces padding)
        if x_lens is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)

        # Optional output layers for edge prediction
        if self.linear_out1:
            hidden = self.relu(self.linear_out1(x))
            hidden = self.linear_out2(hidden)
        else:
            hidden = x

        # Optionally predict node types
        if self.predict_node_types and self.node_type_predictor is not None:
            node_types = self.node_type_predictor(x)  # [batch, seq_len, num_node_types]
            return hidden, node_types
        else:
            return hidden

class EdgeLevelMLP(nn.Module):
    """
    Edge-Level MLP that can be optionally conditioned on truth tables for generating AIGs.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            tt_size: Optional[int] = None,
            edge_feature_len: int = 1,
            tt_embedding_size: int = 64
    ):
        """
        Initialize the Edge-Level MLP with optional truth table conditioning.

        Args:
            input_size: Size of the hidden state outputted by the graph-level RNN
            hidden_size: Size of the hidden layer
            output_size: Number of edges probabilities to output
            tt_size: Optional size of the truth table (8x256 for 8 outputs, 2^8 input combinations)
                If None, no truth table conditioning is used
            edge_feature_len: Number of features associated with each edge.
                Default is 1 (i.e. scalar value 0/1 indicating whether the
                edge is set or not).
            tt_embedding_size: Size of the truth table embedding when conditioning is used
        """
        super().__init__()
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = tt_size is not None

        # Truth table conditioning (if enabled)
        if self.use_conditioning:
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size),
                nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size),
                nn.ReLU()
            )
            # Enhanced input size to account for truth table embedding
            enhanced_input_size = input_size + tt_embedding_size
        else:
            enhanced_input_size = input_size

        # MLP layers
        self.linear1 = nn.Linear(enhanced_input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size * edge_feature_len)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
            self,
            x: torch.Tensor,
            return_logits: bool = False,
            truth_table: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional truth table conditioning.

        Args:
            x: Input tensor of shape [batch, seq_len, input_size]. Should be the
                hidden GRU state outputted by the graph-level RNN.
            return_logits: Set to True to output the logits without activation
            truth_table: Optional tensor of shape [batch, tt_size] containing
                the desired output behavior of the circuit as flattened truth tables.
                Only used if model was initialized with tt_size.

        Returns:
            The next edge prediction of shape [batch, seq_len, output_size, edge_feature_len].
        """
        # If truth table conditioning is enabled and truth table is provided
        if self.use_conditioning and truth_table is not None:
            batch_size, seq_len = x.shape[0], x.shape[1]

            # Embed the truth table
            tt_embed = self.tt_embedding(truth_table)  # [batch, tt_embedding_size]

            # Expand truth table embedding to match sequence dimension
            tt_embed = tt_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, tt_embedding_size]

            # Concatenate RNN hidden state with truth table embedding
            x = torch.cat([x, tt_embed], dim=2)  # [batch, seq_len, input_size + tt_embedding_size]

        # Process through MLP
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        if not return_logits:
            x = self.sigmoid(x)

        # Reshape x to get edge features into separate dimension
        x = torch.reshape(x, [x.shape[0], x.shape[1], -1,
                              self.edge_feature_len])  # [batch, seq_len, output_size, edge_feature_len]

        return x


class EdgeLevelRNN(nn.Module):
    """
    Edge-Level RNN that can be optionally conditioned on truth tables for generating AIGs.
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
        """
        Initialize the Edge-Level RNN with optional truth table conditioning.

        Args:
            embedding_size: Size of the input embedding fed to the GRU
            hidden_size: Hidden size of the GRU
            num_layers: Number of GRU layers
            tt_size: Optional size of the truth table (8x256 for 8 outputs, 2^8 input combinations)
                If None, no truth table conditioning is used
            edge_feature_len: Number of features associated with each edge.
                Default is 1 (i.e. scalar value 0/1 indicating whether the
                edge is set or not).
            tt_embedding_size: Size of the truth table embedding when conditioning is used
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = tt_size is not None

        # Truth table conditioning (if enabled)
        if self.use_conditioning:
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size),
                nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size),
                nn.ReLU()
            )
            # Enhanced embedding size to account for truth table info
            enhanced_embedding_size = embedding_size + tt_embedding_size
        else:
            enhanced_embedding_size = embedding_size

        # Edge feature embedding
        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()

        # GRU takes enhanced embedding (or just edge embedding if no conditioning)
        self.gru = nn.GRU(
            input_size=enhanced_embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Output layers
        self.linear_out1 = nn.Linear(hidden_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None

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

    def forward(
            self,
            x: torch.Tensor,
            x_lens: Optional[List[int]] = None,
            return_logits: bool = False,
            truth_table: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional truth table conditioning.

        Args:
            x: Input tensor of shape [batch, seq_len, edge_feature_len].
            x_lens: List of sequence lengths (i.e. number of graph nodes) of
                each batch entry. Should be on the CPU. This is used to pack
                the input to get rid of padding and increase efficiency.
                Set to 'None' to disable packing.
            return_logits: Set to True to output the logits without activation
            truth_table: Optional tensor of shape [batch, tt_size] containing
                the desired output behavior of the circuit as flattened truth tables.
                Only used if model was initialized with tt_size.

        Returns:
            The next edge prediction of shape [batch, seq_len, edge_feature_len].
        """
        assert self.hidden is not None, "Hidden state not set!"

        # Edge feature embedding
        x = self.relu(self.linear_in(x))  # [batch, seq_len, embedding_size]

        # If truth table conditioning is enabled and truth table is provided
        if self.use_conditioning and truth_table is not None:
            batch_size, seq_len = x.shape[0], x.shape[1]

            # Embed the truth table
            tt_embed = self.tt_embedding(truth_table)  # [batch, tt_embedding_size]

            # Expand truth table embedding to match sequence dimension
            tt_embed = tt_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, tt_embedding_size]

            # Concatenate edge embedding with truth table embedding
            x = torch.cat([x, tt_embed], dim=2)  # [batch, seq_len, embedding_size + tt_embedding_size]

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