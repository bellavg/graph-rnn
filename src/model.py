import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math # Needed for positional encoding if used, or sqrt in attention
from typing import Optional, Tuple, Union # Added Union

# --- Original GraphLevelRNN (No Attention) ---
class GraphLevelRNN(nn.Module):
    """
    Node-level GRU model for graph generation.
    Processes the graph sequence step-by-step and optionally predicts node types.
    """
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 output_size=None, edge_feature_len=1,
                 predict_node_types=False, num_node_types=None, # Flag to enable node type prediction
                 use_conditioning=False, tt_size=None,
                 max_level=None):
        """
        Initializes the GraphLevelRNN model.

        Args:
            input_size (int): Effective input size (m or max_nodes-1).
            embedding_size (int): Dimension of the node/edge embedding.
            hidden_size (int): Dimension of the GRU hidden state.
            num_layers (int): Number of GRU layers.
            output_size (int, optional): Dimension of the final output projection.
                                         If None, output is the GRU hidden state.
                                         Should match EdgeLevelRNN hidden_size if edge model is RNN. Defaults to None.
            edge_feature_len (int, optional): Number of edge features (e.g., 3 for None/Reg/Inv). Defaults to 1.
            predict_node_types (bool, optional): If True, add a head to predict node types. Defaults to False.
            num_node_types (int, optional): Number of node types to predict. Required if predict_node_types is True. Defaults to None.
            use_conditioning (bool, optional): If True, enable conditioning (e.g., on truth tables). Defaults to False.
            tt_size (int, optional): Size of the conditioning vector (e.g., truth table size). Defaults to None.
            max_level (int, optional): Maximum node level for level embedding. Defaults to None.
        """
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.edge_feature_len = edge_feature_len
        self.predict_node_types = predict_node_types # Store the flag
        self.num_node_types = num_node_types       # Store number of types
        self.use_conditioning = use_conditioning
        self.tt_size = tt_size
        self.max_level = max_level

        # Calculate input feature dimension for the initial linear layer
        lin_in_features = input_size * edge_feature_len
        if self.use_conditioning and self.tt_size is not None:
            lin_in_features += self.tt_size

        # Initial embedding layer
        self.linear_in = nn.Linear(lin_in_features, embedding_size)
        self.relu = nn.ReLU()

        # Optional level embedding layer
        self.level_embedding = None
        if self.max_level is not None and self.max_level >= 0:
             self.level_embedding = nn.Embedding(self.max_level + 1, embedding_size)
             print(f"INFO: GraphLevelRNN using level embedding up to level {self.max_level}")

        # Core GRU layer
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        # Optional output projection layers
        self.linear_out1 = None
        self.linear_out2 = None
        if output_size:
            self.linear_out1 = nn.Linear(hidden_size, embedding_size)
            self.linear_out2 = nn.Linear(embedding_size, output_size)

        # Optional node type predictor head
        self.node_type_predictor = None
        if self.predict_node_types:
            if self.num_node_types is None or self.num_node_types <= 0:
                raise ValueError("num_node_types must be a positive integer if predict_node_types is True")
            # Predicts node type logits from the GRU hidden state
            self.node_type_predictor = nn.Linear(hidden_size, self.num_node_types)
            print(f"INFO: GraphLevelRNN initialized with node type predictor head ({self.num_node_types} types).")

        # Stored hidden state for sequential processing
        self.hidden = None

    def reset_hidden(self):
        """Resets the GRU hidden state to None."""
        self.hidden = None

    def forward(self, x, x_lens=None, truth_table=None, levels=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the GraphLevelRNN.

        Args:
            x (torch.Tensor): Input sequence tensor shape [batch, seq_len, input_size, edge_feature_len].
            x_lens (torch.Tensor or list, optional): Tensor/list of actual sequence lengths [batch]. Defaults to None.
            truth_table (torch.Tensor, optional): Optional conditioning tensor [batch, tt_size]. Defaults to None.
            levels (torch.Tensor, optional): Optional node level tensor [batch, seq_len]. Defaults to None.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                If predict_node_types is True:
                    Tuple: (final_output, node_type_logits)
                        - final_output: Output sequence for edge model [batch, seq_len, output_size or hidden_size].
                        - node_type_logits: Logits for node type prediction [batch, seq_len, num_node_types].
                Else:
                    torch.Tensor: final_output sequence.
        """
        batch_size, seq_len, _, _ = x.shape
        device = x.device # Get device from input

        # 1. Flatten edge features
        # Shape: [batch, seq_len, input_size * edge_feature_len]
        x_flat = torch.flatten(x, 2, 3)

        # 2. Concatenate conditioning vector (if enabled)
        if self.use_conditioning and truth_table is not None:
            # Expand truth table to match sequence length
            tt_expanded = truth_table.unsqueeze(1).expand(-1, seq_len, -1)
            # Concatenate along the feature dimension
            x_flat = torch.cat((x_flat, tt_expanded), dim=2)

        # 3. Apply initial embedding
        # Shape: [batch, seq_len, embedding_size]
        embedded_input = self.relu(self.linear_in(x_flat))

        # 4. Add level embedding (if enabled)
        if self.level_embedding is not None and levels is not None:
            # Ensure levels tensor has the correct shape
            if levels.shape[0] == batch_size and levels.shape[1] == seq_len:
                try:
                    # Clamp levels to be within the embedding range
                    clamped_levels = torch.clamp(levels, 0, self.max_level)
                    # Get embeddings and add to the input
                    lvl_emb = self.level_embedding(clamped_levels)
                    embedded_input = embedded_input + lvl_emb
                except Exception as e:
                    # Log warning if level embedding fails
                    print(f"Warning: Error adding level embedding in GraphLevelRNN: {e}")
            else:
                print(f"Warning: Levels tensor shape mismatch in GraphLevelRNN. Expected [{batch_size}, {seq_len}], got {levels.shape}. Skipping level embedding.")


        # 5. Prepare GRU input and handle padding
        gru_input = embedded_input
        target_padded_length = gru_input.shape[1] # Store original sequence length

        if x_lens is not None:
            # Ensure x_lens is a tensor on the correct device before moving to CPU
            if not isinstance(x_lens, torch.Tensor):
                x_lens_tensor = torch.tensor(x_lens, device=device)
            else:
                x_lens_tensor = x_lens.to(device)

            x_lens_cpu = x_lens_tensor.cpu() # Move lengths to CPU for packing
            try:
                # Pack the sequence to handle variable lengths efficiently
                gru_input = pack_padded_sequence(gru_input, x_lens_cpu, batch_first=True, enforce_sorted=False)
            except RuntimeError as e:
                print(f"Error packing sequence in GraphLevelRNN: {e}")
                raise e # Re-raise error

        # 6. Process through GRU
        # Initialize hidden state if it's None (first step or reset)
        if self.hidden is None:
            # Initialize hidden state with zeros on the correct device
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            self.hidden = h_0

        # Pass input and hidden state to GRU
        # gru_output_packed contains packed outputs for each time step
        # self.hidden is updated with the hidden state of the last time step
        gru_output_packed, self.hidden = self.gru(gru_input, self.hidden)

        # 7. Unpack GRU output
        gru_output = gru_output_packed
        if x_lens is not None:
             try:
                # Pad the packed sequence back to the original length
                gru_output, _ = pad_packed_sequence(gru_output_packed, batch_first=True, total_length=target_padded_length)
             except RuntimeError as e:
                 print(f"Error unpacking sequence in GraphLevelRNN: {e}")
                 raise e # Re-raise error
        # gru_output shape: [batch, seq_len, hidden_size]

        # 8. Predict node types (if enabled)
        node_type_logits = None
        if self.predict_node_types and self.node_type_predictor is not None:
            # Pass GRU output through the predictor head
            node_type_logits = self.node_type_predictor(gru_output)
            # node_type_logits shape: [batch, seq_len, num_node_types]

        # 9. Apply final output projection (if configured)
        final_output = gru_output # Start with GRU output
        if self.linear_out1 and self.linear_out2:
            # Apply projection layers, often used to match edge model's hidden size
            final_output = self.relu(self.linear_out1(final_output))
            final_output = self.linear_out2(final_output)
            # final_output shape: [batch, seq_len, output_size]
        # else: final_output shape remains [batch, seq_len, hidden_size]

        # 10. Return appropriate output based on configuration
        if self.predict_node_types:
            if node_type_logits is None:
                # Defensive check: This should not happen if initialized correctly
                raise RuntimeError("predict_node_types is True, but node_type_logits were not calculated.")
            # Return both the sequence output (for edge model) and node type predictions
            return final_output, node_type_logits
        else:
            # Return only the sequence output
            return final_output


# --- EdgeLevelRNN ---
class EdgeLevelRNN(nn.Module):
    """
    Edge-level GRU model for graph generation.
    Predicts the edge type for the current node based on its context
    (provided by the node-level model's hidden state).
    """
    def __init__(self, embedding_size, hidden_size, num_layers,
                 edge_feature_len=3,  # Number of edge types (e.g., 3 for None/Reg/Inv)
                 use_conditioning=False,
                 tt_size=None,
                 tt_embedding_size=64):
        """
        Initializes the EdgeLevelRNN model.

        Args:
            embedding_size (int): Dimension of the edge input embedding.
            hidden_size (int): Dimension of the GRU hidden state. Must match the
                               output size of the node-level model's hidden state
                               used for initialization via set_first_layer_hidden.
            num_layers (int): Number of GRU layers.
            edge_feature_len (int, optional): Number of edge types to predict. Defaults to 3.
            use_conditioning (bool, optional): If True, enable conditioning. Defaults to False.
            tt_size (int, optional): Size of the conditioning vector. Defaults to None.
            tt_embedding_size (int, optional): Dimension for the conditioning embedding. Defaults to 64.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.edge_feature_len = edge_feature_len
        self.use_conditioning = use_conditioning and (tt_size is not None)

        # Optional conditioning embedding layers
        self.tt_embedding = None
        gru_input_size = embedding_size # Start with base embedding size
        if self.use_conditioning:
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size),
                nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size),
                nn.ReLU()
            )
            # Add conditioning embedding size to GRU input size
            gru_input_size += tt_embedding_size
            print(f"INFO: EdgeLevelRNN using TT conditioning. GRU input size: {gru_input_size}")

        # Input embedding layer for edge features
        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()

        # Core GRU layer
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        # Output projection layers
        self.linear_out1 = nn.Linear(hidden_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len) # Output logits for each edge type
        self.sigmoid = nn.Sigmoid() # Sigmoid for potential binary case (BCELoss)

        # Stored hidden state
        self.hidden = None

    def set_first_layer_hidden(self, h):
        """
        Initializes the hidden state of the first GRU layer using the output
        from the node-level model. Assumes h is the relevant context.

        Args:
            h (torch.Tensor): Hidden state or output from the node-level model.
                              Expected shape compatible with GRU hidden state initialization.
        """
        # Validate input hidden state dimension
        if h.shape[-1] != self.hidden_size:
            raise ValueError(f"Hidden state dimension mismatch in set_first_layer_hidden. "
                             f"Node model output ({h.shape[-1]}) != EdgeRNN hidden ({self.hidden_size})")

        # Handle different possible input shapes for h
        if len(h.shape) == 3 and h.shape[0] > 1: # Shape [num_layers, batch, hidden_size]
            h_first_layer = h[0:1, :, :] # Take only the first layer's state
        elif len(h.shape) == 2: # Shape [batch, hidden_size]
            h_first_layer = h.unsqueeze(0) # Add layer dimension
        elif len(h.shape) == 3 and h.shape[0] == 1: # Shape [1, batch, hidden_size]
            h_first_layer = h # Already correct shape
        else:
            raise ValueError(f"Unexpected shape for hidden state h: {h.shape}")

        # Initialize remaining layers' hidden states with zeros if num_layers > 1
        if self.num_layers > 1:
            zeros = torch.zeros([self.num_layers - 1, h_first_layer.shape[1], h_first_layer.shape[2]], device=h.device)
            # Concatenate the first layer state with zeros for other layers
            self.hidden = torch.cat([h_first_layer, zeros], dim=0)
        else:
            self.hidden = h_first_layer # Use directly if only one layer

    def reset_hidden(self):
        """Resets the GRU hidden state to None."""
        self.hidden = None

    def forward(self, x, x_lens=None, return_logits=False, truth_table=None):
        """
        Forward pass for the EdgeLevelRNN.

        Args:
            x (torch.Tensor): Input sequence tensor (edge features)
                              Shape [batch, seq_len, edge_feature_len].
            x_lens (torch.Tensor or list, optional): Actual sequence lengths [batch]. Defaults to None.
            return_logits (bool, optional): If True, return raw logits before sigmoid/softmax.
                                           Required for CrossEntropyLoss. Defaults to False.
            truth_table (torch.Tensor, optional): Conditioning tensor [batch, tt_size]. Defaults to None.

        Returns:
            torch.Tensor: Output tensor containing edge predictions (probabilities or logits).
                          Shape [batch, seq_len, edge_feature_len].
        """
        # 1. Initialize hidden state if necessary
        if self.hidden is None:
            # Determine batch size from input x
            batch_size = x.shape[0] if x.dim() > 1 else 1
            device = x.device
            print(f"INFO: Initializing hidden state for EdgeLevelRNN (Batch: {batch_size}, Device: {device})")
            # Initialize hidden state with zeros
            self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        # 2. Embed edge features
        # Shape: [batch, seq_len, embedding_size]
        embedded_x = self.relu(self.linear_in(x))
        gru_input = embedded_x

        # 3. Add conditioning (if enabled)
        if self.use_conditioning and truth_table is not None and self.tt_embedding is not None:
            # Ensure truth_table is on the correct device
            truth_table = truth_table.to(embedded_x.device)
            # Embed the truth table
            tt_emb = self.tt_embedding(truth_table)
            # Expand embedded truth table to match sequence length
            seq_len_edge = embedded_x.shape[1]
            tt_emb_expanded = tt_emb.unsqueeze(1).expand(-1, seq_len_edge, -1)
            # Concatenate along the feature dimension
            gru_input = torch.cat((embedded_x, tt_emb_expanded), dim=2)

        # 4. Prepare for GRU and handle padding
        target_padded_length = gru_input.shape[1] # Store original sequence length
        if x_lens is not None:
            # Ensure x_lens is a tensor on the correct device before moving to CPU
            if not isinstance(x_lens, torch.Tensor):
                x_lens_tensor = torch.tensor(x_lens, device=device)
            else:
                x_lens_tensor = x_lens.to(device)

            x_lens_cpu = x_lens_tensor.cpu() # Move to CPU for packing
            try:
                # Pack the sequence
                gru_input = pack_padded_sequence(gru_input, x_lens_cpu, batch_first=True, enforce_sorted=False)
            except RuntimeError as e:
                print(f"Error packing sequence in EdgeLevelRNN: {e}")
                raise e # Re-raise

        # 5. Process through GRU
        # Pass input and current hidden state; update hidden state
        gru_output_packed, self.hidden = self.gru(gru_input, self.hidden)

        # 6. Unpack GRU output
        gru_output = gru_output_packed
        if x_lens is not None:
            try:
                # Pad back to original length
                gru_output, _ = pad_packed_sequence(gru_output_packed, batch_first=True,
                                                    total_length=target_padded_length)
            except RuntimeError as e:
                print(f"Error unpacking sequence in EdgeLevelRNN: {e}")
                raise e # Re-raise
        # gru_output shape: [batch, seq_len, hidden_size]

        # 7. Apply output projection layers
        # Shape: [batch, seq_len, embedding_size]
        out = self.relu(self.linear_out1(gru_output))
        # Shape: [batch, seq_len, edge_feature_len]
        logits = self.linear_out2(out)

        # 8. Return logits or probabilities
        if return_logits:
            # Return raw logits (needed for CrossEntropyLoss)
            return logits
        else:
            # Apply sigmoid (for BCELoss or if probabilities are desired)
            # Note: For multi-class classification with CrossEntropyLoss,
            # you should use return_logits=True and apply Softmax implicitly
            # within the loss function or explicitly after this forward pass if needed elsewhere.
            return self.sigmoid(logits)

