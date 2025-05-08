import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math  # Needed for positional encoding if used, or sqrt in attention
from typing import Optional, Tuple, Union  # Added Union


# --- Original GraphLevelRNN (No Attention) ---
class GraphLevelRNN(nn.Module):
    """
    Node-level GRU model for graph generation.
    Processes the graph sequence step-by-step and optionally predicts node types.
    """

    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 output_size=None, edge_feature_len=1,
                 predict_node_types=False, num_node_types=None,  # Flag to enable node type prediction
                 use_conditioning=False, tt_size=None,
                 max_level=None):
        """
        Initializes the GraphLevelRNN model.
        """
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.edge_feature_len = edge_feature_len
        self.predict_node_types = predict_node_types  # Store the flag
        self.num_node_types = num_node_types  # Store number of types
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
            self.level_embedding = nn.Embedding(self.max_level + 2, embedding_size)  # Max level + SOS + 0-indexed
            print(
                f"INFO: GraphLevelRNN using level embedding up to level {self.max_level} (Embedding size: {self.max_level + 2})")

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
        device = x.device  # Get device from input early

        x_flat = torch.flatten(x, 2, 3)

        if self.use_conditioning and truth_table is not None:
            tt_expanded = truth_table.unsqueeze(1).expand(-1, seq_len, -1)
            x_flat = torch.cat((x_flat, tt_expanded), dim=2)

        embedded_input = self.relu(self.linear_in(x_flat))

        if self.level_embedding is not None and levels is not None:
            if levels.shape[0] == batch_size and levels.shape[1] == seq_len:
                try:
                    # Clamp levels: 0 for SOS, 1 to max_level+1 for actual levels
                    # Assuming levels input are 0-indexed for actual levels, map them to 1 to max_level+1
                    # SOS level is implicitly handled if levels tensor aligns with x (which includes SOS)
                    # Max embedding index is max_level + 1 (0 for SOS, 1..max_level+1 for levels 0..max_level)
                    clamped_levels = torch.clamp(levels, 0, self.max_level + 1)
                    lvl_emb = self.level_embedding(clamped_levels)
                    embedded_input = embedded_input + lvl_emb
                except Exception as e:
                    print(
                        f"Warning: Error adding level embedding in GraphLevelRNN: {e}. Levels shape: {levels.shape}, Clamped max: {clamped_levels.max()}, Emb size: {self.level_embedding.num_embeddings}")
            else:
                print(
                    f"Warning: Levels tensor shape mismatch in GraphLevelRNN. Expected [{batch_size}, {seq_len}], got {levels.shape}. Skipping level embedding.")

        gru_input = embedded_input
        target_padded_length = gru_input.shape[1]

        if x_lens is not None:
            if not isinstance(x_lens, torch.Tensor):
                x_lens_tensor = torch.tensor(x_lens, dtype=torch.long, device=device)  # Ensure long type for lengths
            else:
                x_lens_tensor = x_lens.to(device)

            # Filter out zero lengths to prevent pack_padded_sequence error
            # This can happen if a batch contains empty sequences, though ideally handled in data loading
            # For now, we assume x_lens contains valid lengths > 0 if provided.
            # If x_lens_tensor can contain zeros, it needs careful handling.
            # Example: valid_indices = x_lens_tensor > 0; gru_input = gru_input[valid_indices]; x_lens_cpu = x_lens_tensor[valid_indices].cpu()

            x_lens_cpu = x_lens_tensor.cpu()
            try:
                gru_input = pack_padded_sequence(gru_input, x_lens_cpu, batch_first=True, enforce_sorted=False)
            except RuntimeError as e:
                print(
                    f"Error packing sequence in GraphLevelRNN: {e}. Max lens: {x_lens_cpu.max()}, Seq len: {target_padded_length}")
                # Consider what to do here: re-raise, or return a zero tensor?
                # For now, re-raising to make the error visible.
                raise e

        if self.hidden is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            self.hidden = h_0
        elif self.hidden.shape[1] != batch_size:  # Handle batch size change if any (e.g. last batch)
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
            if node_type_logits is None:
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
            print(f"INFO: EdgeLevelRNN using TT conditioning. GRU input size: {gru_input_size}")

        self.linear_in = nn.Linear(edge_feature_len, embedding_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.linear_out1 = nn.Linear(hidden_size, embedding_size)
        self.linear_out2 = nn.Linear(embedding_size, edge_feature_len)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None

    def set_first_layer_hidden(self, h):
        if h.shape[-1] != self.hidden_size:
            raise ValueError(f"Hidden state dimension mismatch in set_first_layer_hidden. "
                             f"Node model output ({h.shape[-1]}) != EdgeRNN hidden ({self.hidden_size})")

        # h is expected to be [total_nodes, hidden_size]
        # GRU expects hidden as [num_layers, batch_size (total_nodes here), hidden_size]
        h_first_layer = h.unsqueeze(0)  # [1, total_nodes, hidden_size]

        if self.num_layers > 1:
            zeros = torch.zeros([self.num_layers - 1, h_first_layer.shape[1], h_first_layer.shape[2]], device=h.device)
            self.hidden = torch.cat([h_first_layer, zeros], dim=0)
        else:
            self.hidden = h_first_layer

    def reset_hidden(self):
        self.hidden = None

    def forward(self, x, x_lens=None, return_logits=False, truth_table=None):
        # ******** CORE FIX FOR UnboundLocalError ********
        # Define device from an input tensor at the beginning of the method.
        device = x.device
        # ************************************************

        if self.hidden is None:
            batch_size = x.shape[0] if x.dim() > 1 else 1  # x is [total_nodes, seq_len, features]
            # print(f"INFO: Initializing hidden state for EdgeLevelRNN (Batch: {batch_size}, Device: {device})")
            self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        elif self.hidden.shape[1] != x.shape[0]:  # Handle batch size change (total_nodes can vary)
            # print(f"INFO: EdgeLevelRNN hidden batch size mismatch. Re-initializing. Old: {self.hidden.shape[1]}, New: {x.shape[0]}")
            self.hidden = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=device)

        embedded_x = self.relu(self.linear_in(x))
        gru_input = embedded_x

        if self.use_conditioning and truth_table is not None and self.tt_embedding is not None:
            truth_table = truth_table.to(device)  # Ensure conditioning is on correct device
            tt_emb = self.tt_embedding(truth_table)
            seq_len_edge = embedded_x.shape[1]
            tt_emb_expanded = tt_emb.unsqueeze(1).expand(-1, seq_len_edge, -1)
            gru_input = torch.cat((embedded_x, tt_emb_expanded), dim=2)

        target_padded_length = gru_input.shape[1]
        if x_lens is not None:
            if not isinstance(x_lens, torch.Tensor):
                # x_lens here are edge lengths for each node, should be on CPU for pack_padded_sequence
                x_lens_tensor = torch.tensor(x_lens, dtype=torch.long, device='cpu')  # Explicitly CPU
            else:
                x_lens_tensor = x_lens.cpu()  # Ensure it's on CPU

            # Filter out zero lengths before packing if they can occur
            # valid_indices = x_lens_tensor > 0
            # gru_input_filtered = gru_input[valid_indices]
            # x_lens_cpu_filtered = x_lens_tensor[valid_indices]
            # if gru_input_filtered.shape[0] == 0: # All sequences were length 0
            #     # Handle empty input case: return zero logits or appropriate shape
            #     # This depends on how downstream code handles it.
            #     # For now, assume lengths are > 0 or pack_padded will handle it if not empty.
            #     pass

            try:
                # x_lens_tensor should be on CPU
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
