import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math  # Needed for positional encoding if used, or sqrt in attention
from typing import Optional, Tuple, Union  # Added Union


# --- GraphLevelRNN ---
class GraphLevelRNN(nn.Module):
    """
    Node-level GRU model for graph generation.
    Processes the graph sequence step-by-step and optionally predicts node types.
    """

    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 output_size=None, edge_feature_len=1,
                 predict_node_types=False, num_node_types=None,
                 use_conditioning=False, tt_size=None,
                 max_level=None):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.edge_feature_len = edge_feature_len
        self.predict_node_types = predict_node_types
        self.num_node_types = num_node_types
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
            # Embedding for levels 0 to max_level, plus one for potential SOS token or padding.
            # If SOS is represented by level 0, and actual levels are 1 to max_level+1,
            # then num_embeddings should be max_level + 2.
            self.level_embedding = nn.Embedding(self.max_level + 2, embedding_size)
            print(
                f"INFO: GraphLevelRNN using level embedding up to level {self.max_level} (Embedding table size: {self.max_level + 2})")

        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        self.linear_out1 = None
        self.linear_out2 = None
        if output_size:
            self.linear_out1 = nn.Linear(hidden_size, embedding_size)
            self.linear_out2 = nn.Linear(embedding_size, output_size)

        self.node_type_predictor = None
        if self.predict_node_types:
            if self.num_node_types is None or self.num_node_types <= 0:
                raise ValueError("num_node_types must be a positive integer if predict_node_types is True")
            self.node_type_predictor = nn.Linear(hidden_size, self.num_node_types)
            print(f"INFO: GraphLevelRNN initialized with node type predictor head ({self.num_node_types} types).")

        self.hidden = None

    def reset_hidden(self):
        self.hidden = None

    def forward(self, x, x_lens=None, truth_table=None, levels=None) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
                    # Ensure levels are within the valid range for the embedding layer
                    # Max index for embedding is (self.max_level + 2) - 1 = self.max_level + 1
                    clamped_levels = torch.clamp(levels, 0, self.max_level + 1)
                    lvl_emb = self.level_embedding(clamped_levels)
                    embedded_input = embedded_input + lvl_emb
                except IndexError as e:
                    print(
                        f"Warning: IndexError adding level embedding in GraphLevelRNN: {e}. Max level in batch: {levels.max()}, Embedding size: {self.level_embedding.num_embeddings}. Clamped max: {clamped_levels.max()}")
                except Exception as e:
                    print(f"Warning: Error adding level embedding in GraphLevelRNN: {e}.")
            else:
                print(
                    f"Warning: Levels tensor shape mismatch in GraphLevelRNN. Expected [{batch_size}, {seq_len}], got {levels.shape}. Skipping level embedding.")

        gru_input = embedded_input
        target_padded_length = gru_input.shape[1]

        if x_lens is not None:
            if not isinstance(x_lens, torch.Tensor):
                x_lens_tensor = torch.tensor(x_lens, dtype=torch.long,
                                             device='cpu')  # Lengths must be on CPU for pack_padded
            else:
                x_lens_tensor = x_lens.cpu()

            try:
                gru_input = pack_padded_sequence(gru_input, x_lens_tensor, batch_first=True, enforce_sorted=False)
            except RuntimeError as e:
                print(
                    f"Error packing sequence in GraphLevelRNN: {e}. Max lens: {x_lens_tensor.max() if x_lens_tensor.numel() > 0 else 'N/A'}, Seq len: {target_padded_length}")
                raise e

        if self.hidden is None or self.hidden.shape[1] != batch_size:  # Re-initialize if batch size changes
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            self.hidden = h_0

        gru_output_packed, self.hidden = self.gru(gru_input, self.hidden)

        gru_output = gru_output_packed
        if x_lens is not None:
            try:
                gru_output, _ = pad_packed_sequence(gru_output_packed, batch_first=True,
                                                    total_length=target_padded_length)
            except RuntimeError as e:
                print(f"Error unpacking sequence in GraphLevelRNN: {e}")
                raise e

        node_type_logits = None
        if self.predict_node_types and self.node_type_predictor is not None:
            node_type_logits = self.node_type_predictor(gru_output)

        final_output = gru_output
        if self.linear_out1 and self.linear_out2:
            final_output = self.relu(self.linear_out1(final_output))
            final_output = self.linear_out2(final_output)

        if self.predict_node_types:
            if node_type_logits is None:  # Should not happen if initialized correctly
                raise RuntimeError("predict_node_types is True, but node_type_logits were not calculated.")
            return final_output, node_type_logits
        else:
            return final_output


# --- EdgeLevelRNN ---
class EdgeLevelRNN(nn.Module):
    """
    Edge-level GRU model for graph generation.
    """

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
        gru_input_size = embedding_size
        if self.use_conditioning:
            self.tt_embedding = nn.Sequential(
                nn.Linear(tt_size, tt_embedding_size),
                nn.ReLU(),
                nn.Linear(tt_embedding_size, tt_embedding_size),
                nn.ReLU()
            )
            gru_input_size += tt_embedding_size
            # print(f"INFO: EdgeLevelRNN using TT conditioning. GRU input size: {gru_input_size}")

        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.linear_out1 = nn.Linear(hidden_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None

    def set_first_layer_hidden(self, h_context):
        """
        Initializes the hidden state of the GRU layers using the context from the node-level model.
        Args:
            h_context (torch.Tensor): Context tensor.
                - During training (from packed sequence): [current_batch_size, features] (2D)
                  where current_batch_size is sum of nodes in batch.
                - During generation (output of GraphLevelRNN for one node): [1, 1, features] (3D)
        """
        if h_context.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Feature dimension of h_context ({h_context.shape[-1]}) "
                f"must match EdgeLevelRNN hidden_size ({self.hidden_size})."
            )

        current_batch_for_edge_rnn = h_context.shape[0]

        if h_context.ndim == 3:
            # Expected from generation: [1, 1, self.hidden_size]
            if h_context.shape[0] == 1 and h_context.shape[1] == 1:
                # Squeeze to [self.hidden_size], then treat as batch_size=1 for EdgeRNN
                h_processed_for_gru_input = h_context.squeeze(0)  # Shape [1, self.hidden_size]
                current_batch_for_edge_rnn = 1  # This is the 'batch size' for this single node's edge generation
            else:
                raise ValueError(
                    f"Unexpected 3D shape for h_context: {h_context.shape}. Expected [1, 1, features] during single item generation.")
        elif h_context.ndim == 2:
            # Expected from training (packed data): [sum_of_nodes, self.hidden_size]
            h_processed_for_gru_input = h_context  # Shape [sum_of_nodes, self.hidden_size]
            current_batch_for_edge_rnn = h_context.shape[0]
        else:
            raise ValueError(
                f"Unexpected number of dimensions for h_context: {h_context.ndim}. Shape: {h_context.shape}")

        # h_processed_for_gru_input is now [current_batch_for_edge_rnn, self.hidden_size]
        # We need to make it [1, current_batch_for_edge_rnn, self.hidden_size] for the first GRU layer's hidden state
        h_init_first_layer = h_processed_for_gru_input.unsqueeze(
            0)  # Shape [1, current_batch_for_edge_rnn, self.hidden_size]

        if self.num_layers > 1:
            zeros = torch.zeros(
                self.num_layers - 1,
                current_batch_for_edge_rnn,
                self.hidden_size,
                device=h_init_first_layer.device
            )
            self.hidden = torch.cat([h_init_first_layer, zeros], dim=0)
        else:
            self.hidden = h_init_first_layer

        # self.hidden is now [num_layers, current_batch_for_edge_rnn, self.hidden_size]

    def reset_hidden(self):
        self.hidden = None

    def forward(self, x, x_lens=None, return_logits=False, truth_table=None):
        device = x.device
        current_input_batch_size = x.shape[0]  # This is 'total_nodes' in training or 1 in generation

        if self.hidden is None or self.hidden.shape[1] != current_input_batch_size:
            # print(f"INFO: Initializing/resetting hidden state for EdgeLevelRNN (Batch: {current_input_batch_size}, Device: {device})")
            self.hidden = torch.zeros(self.num_layers, current_input_batch_size, self.hidden_size, device=device)

        embedded_x = self.relu(self.linear_in(x))
        gru_input = embedded_x

        if self.use_conditioning and truth_table is not None and self.tt_embedding is not None:
            truth_table = truth_table.to(device)
            tt_emb = self.tt_embedding(truth_table)
            seq_len_edge = embedded_x.shape[1]
            # Ensure tt_emb can be broadcast or expanded correctly
            if tt_emb.shape[0] != current_input_batch_size and tt_emb.shape[0] == 1:  # If tt_emb is [1, features]
                tt_emb_expanded = tt_emb.expand(current_input_batch_size,
                                                -1)  # Expand to [current_input_batch_size, features]
                tt_emb_expanded = tt_emb_expanded.unsqueeze(1).expand(-1, seq_len_edge, -1)
            elif tt_emb.shape[0] == current_input_batch_size:
                tt_emb_expanded = tt_emb.unsqueeze(1).expand(-1, seq_len_edge, -1)
            else:
                raise ValueError(
                    f"Truth table embedding batch size {tt_emb.shape[0]} doesn't match input batch size {current_input_batch_size}")

            gru_input = torch.cat((embedded_x, tt_emb_expanded), dim=2)

        target_padded_length = gru_input.shape[1]
        if x_lens is not None:
            if not isinstance(x_lens, torch.Tensor):
                x_lens_tensor = torch.tensor(x_lens, dtype=torch.long, device='cpu')
            else:
                x_lens_tensor = x_lens.cpu()
            try:
                gru_input = pack_padded_sequence(gru_input, x_lens_tensor, batch_first=True, enforce_sorted=False)
            except RuntimeError as e:
                print(
                    f"Error packing sequence in EdgeLevelRNN: {e}. Input shape: {gru_input.shape}, Lengths max: {x_lens_tensor.max() if x_lens_tensor.numel() > 0 else 'N/A'}")
                raise e

        gru_output_packed, self.hidden = self.gru(gru_input, self.hidden)

        gru_output = gru_output_packed
        if x_lens is not None:
            try:
                gru_output, _ = pad_packed_sequence(gru_output_packed, batch_first=True,
                                                    total_length=target_padded_length)
            except RuntimeError as e:
                print(f"Error unpacking sequence in EdgeLevelRNN: {e}")
                raise e

        out = self.relu(self.linear_out1(gru_output))
        logits = self.linear_out2(out)

        if return_logits:
            return logits
        else:
            return self.sigmoid(logits)
