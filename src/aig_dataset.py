"""
AIGDataset: Dataset implementation for training GraphRNN models on And-Inverter Graphs.

This module provides the AIGDataset class that handles AIG-specific structures
and features, loading from multiple PKL files and adapting to different
feature encodings during preprocessing. It prepares data for GraphRNN using
topological sorting and node level information. Now includes target node types.
"""

import os
import pickle
import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset # Use base Dataset
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any, Union
import warnings # Added for warnings
import random   # Added for shuffling

# --- Constants for Original GraphRNN Structure & Output ---
# Node types used INTERNALLY by this class and expected by some logic
NODE_TYPES_INTERNAL = {
    "NODE_CONST0": 0, # CHANGED: Renamed from "ZERO" to match aig_config.py
    "PI": 1,          # Primary Input
    "AND": 2,
    "PO": 3,          # Primary Output
    "UNKNOWN": -1     # For internal use when conversion fails
}
# Edge types used INTERNALLY (0=no edge, 1=regular, 2=inverted)
EDGE_TYPES_INTERNAL = {
    "NONE": 0,
    "REGULAR": 1,
    "INVERTED": 2
}
# The final dimension expected by the GraphRNN model's input sequence `x`
NUM_EDGE_FEATURES_RNN = 3

# --- Constants Reflecting NEW Graph Data Format (Based on G2PT/configs/aig.py) ---
# These should match the output of your data generation script (generate_dataset.py)
NUM_NODE_FEATURES_NEW = 4      # Expected length of node 'type' vector (e.g., [1.0, 0.0, 0.0, 0.0])
NUM_EXPLICIT_EDGE_TYPES_NEW = 2 # Expected length of edge 'type' vector (e.g., [1.0, 0.0])

# Define encoding based on NEW format (assuming this order matches aig.py config)
# These map the NEW one-hot vector (as tuple) to the INTERNAL integer type
NODE_TYPE_ENCODING_NEW = {
    # Tuple keys represent the one-hot vector (e.g., [1.0, 0.0, 0.0, 0.0])
    # IMPORTANT: Ensure this order matches your generated data exactly!
    (1.0, 0.0, 0.0, 0.0): NODE_TYPES_INTERNAL["NODE_CONST0"], # Use the updated key
    (0.0, 1.0, 0.0, 0.0): NODE_TYPES_INTERNAL["PI"],
    (0.0, 0.0, 1.0, 0.0): NODE_TYPES_INTERNAL["AND"],
    (0.0, 0.0, 0.0, 1.0): NODE_TYPES_INTERNAL["PO"]
}
# Map based on NEW format [REGULAR, INVERTED] -> internal integer
# IMPORTANT: Verify this order matches your data generation script!
EDGE_TYPE_ENCODING_NEW = {
    # Tuple keys represent the one-hot vector (e.g., [1.0, 0.0] for REG)
    (1.0, 0.0): EDGE_TYPES_INTERNAL["REGULAR"], # Assuming index 0 is REGULAR
    (0.0, 1.0): EDGE_TYPES_INTERNAL["INVERTED"] # Assuming index 1 is INVERTED
}
# --- End New Constants ---

def _calculate_levels(g: nx.DiGraph) -> Tuple[Dict[int, int], int]:
    """Calculates node levels (longest path from source). Requires DAG."""
    if not g: return {}, -1
    levels = {}
    max_level = 0
    source_nodes = [n for n in g.nodes() if g.in_degree(n) == 0]
    for node in source_nodes: levels[node] = 0
    try:
        for node in nx.topological_sort(g):
            if node in levels: continue
            current_max_pred_level = -1
            for pred in g.predecessors(node):
                if pred in levels:
                    current_max_pred_level = max(current_max_pred_level, levels[pred])
            if current_max_pred_level != -1:
                levels[node] = current_max_pred_level + 1
                max_level = max(max_level, levels[node])
            else:
                 warnings.warn(f"Node {node} unreachable in level calc. Assigning level 0.")
                 levels[node] = 0
    except nx.NetworkXUnfeasible:
        warnings.warn("Graph not a DAG in _calculate_levels. Returning default levels.")
        return {n: 0 for n in g.nodes()}, 0
    for node in g.nodes(): levels.setdefault(node, 0)
    return levels, max_level


class AIGDataset(Dataset):

    def __init__(self,
                 graph_files: List[str],
                 training: bool = True,
                 train_split: float = 0.9,
                 max_graphs: Optional[int] = None,
                 max_train_graphs: Optional[int] = None):
        """
        Initializes AIG dataset. Loads from PKL files, preprocesses features,
        calculates levels, determines splits, and computes edge weights.

        Args:
            graph_files: List of paths to pickle files.
            training: True for training set, False for testing set.
            train_split: Fraction for training split.
            max_graphs: Max total graphs to load.
            max_train_graphs: Max training graphs (overrides split if smaller).
        """
        self.graph_files = graph_files
        self.m_internal = None
        self.max_train_graphs = max_train_graphs
        self.edge_weights = None # Initialize edge_weights

        # Load raw graphs
        print(f"Loading AIG graphs from {len(graph_files)} file(s)...")
        all_raw_graphs = []
        for file_path in graph_files:
             print(f" Loading from: {file_path}")
             if not os.path.exists(file_path):
                 warnings.warn(f"Dataset file not found: {file_path}. Skipping.")
                 continue
             try:
                 with open(file_path, 'rb') as f:
                     graphs_in_file = pickle.load(f)
                     if isinstance(graphs_in_file, list):
                         all_raw_graphs.extend(graphs_in_file)
                         print(f"  -> Loaded {len(graphs_in_file)} graphs.")
                     else:
                         warnings.warn(f" Expected list in {file_path}, got {type(graphs_in_file)}. Skipping.")
             except Exception as e:
                 warnings.warn(f" Failed to load/process {file_path}: {e}. Skipping.")
        if not all_raw_graphs: raise ValueError("No graphs loaded.")
        self.raw_graphs_temp = all_raw_graphs

        # Limit graphs if requested
        if max_graphs is not None and max_graphs < len(self.raw_graphs_temp):
            random.shuffle(self.raw_graphs_temp)
            self.raw_graphs_temp = self.raw_graphs_temp[:max_graphs]
            print(f"Limited to random subset of {max_graphs} graphs.")

        # Preprocess graphs (converts features to internal format)
        print("Preprocessing graphs...")
        self.graphs = self._preprocess_graphs()
        if not self.graphs: raise ValueError("No graphs preprocessed successfully.")
        del self.raw_graphs_temp # Free memory

        # Determine max_node_count and m_internal
        self.max_node_count = 0
        for g in self.graphs: self.max_node_count = max(self.max_node_count, g.number_of_nodes())
        # --- Set max_node_count to 64 if it's less, to match config ---
        if self.max_node_count < 64:
            print(f"Warning: Calculated max_node_count ({self.max_node_count}) is less than config (64). Using 64.")
            self.max_node_count = 64
        # --- End modification ---
        self.m_internal = max(1, self.max_node_count - 1)
        print(f"Dataset Stats: Max Nodes={self.max_node_count}, Effective m={self.m_internal}")


        # Calculate levels and filter graphs
        print("Calculating node levels...")
        self.max_level = 0
        graphs_with_levels = []
        for i, g in enumerate(self.graphs):
            if isinstance(g, nx.DiGraph) and g.number_of_nodes() > 0:
                try:
                    node_to_level, graph_max_level = _calculate_levels(g)
                    g.graph['levels'] = node_to_level # Store levels in graph metadata
                    self.max_level = max(self.max_level, graph_max_level)
                    graphs_with_levels.append(g)
                except Exception as e: print(f"\nWarning: Level calc failed for graph {i}: {e}. Skipping.")
            elif isinstance(g, nx.DiGraph): print(f"Warning: Skipping empty graph {i}.")
        self.graphs = graphs_with_levels
        if not self.graphs: raise ValueError("No graphs remaining after level calculation.")
        # --- Set max_level to 22 if it's less, to match config ---
        if self.max_level < 22:
             print(f"Warning: Calculated max_level ({self.max_level}) is less than config (22). Using 22.")
             self.max_level = 22
        # --- End modification ---
        print(f"Maximum node level across dataset: {self.max_level}")

        # Calculate Edge Weights for Training Split
        self._calculate_and_set_edge_weights(train_split)

        # Set up train/test split indices
        np.random.seed(42); np.random.shuffle(self.graphs)
        final_num_graphs = len(self.graphs)
        train_size_by_split = int(final_num_graphs * train_split)
        train_size = min(train_size_by_split, self.max_train_graphs) if self.max_train_graphs is not None else train_size_by_split
        self.start_idx = 0 if training else train_size
        self.length = train_size if training else final_num_graphs - train_size
        print(f"Dataset ready: {self.length} graphs ({'training' if training else 'testing'} split).")
        print(f"Total available graphs: {final_num_graphs}")
        print(f"Training graphs: {train_size}, Testing graphs: {final_num_graphs - train_size}")


    def _preprocess_graphs(self) -> List[nx.DiGraph]:
        """Converts raw graphs to internal format (int types)."""
        processed_graphs = []
        skipped_count, node_err, edge_err = 0, 0, 0
        num_raw = len(self.raw_graphs_temp)
        print(f"Starting preprocessing of {num_raw} raw graphs...")
        for i, g_raw in enumerate(self.raw_graphs_temp):
            # Basic checks
            if not isinstance(g_raw, nx.Graph): skipped_count += 1; continue
            if not g_raw.is_directed(): g_raw = g_raw.to_directed()
            if not nx.is_directed_acyclic_graph(g_raw): skipped_count += 1; continue
            if g_raw.number_of_nodes() == 0: skipped_count += 1; continue

            g_processed = nx.DiGraph(**g_raw.graph)
            node_list = list(g_raw.nodes())
            node_mapping = {old_id: new_id for new_id, old_id in enumerate(node_list)}
            valid_graph = True

            # Process Nodes
            for old_node_id in node_list:
                new_node_id = node_mapping[old_node_id]
                node_data = g_raw.nodes[old_node_id]
                node_type_int = NODE_TYPES_INTERNAL["UNKNOWN"]
                if 'type' in node_data:
                    type_val = node_data['type']
                    if isinstance(type_val, (list, np.ndarray)): # Handle one-hot
                        try:
                            type_tuple = tuple(round(x, 1) for x in type_val[:NUM_NODE_FEATURES_NEW])
                            if len(type_tuple) == NUM_NODE_FEATURES_NEW:
                                node_type_int = NODE_TYPE_ENCODING_NEW.get(type_tuple, NODE_TYPES_INTERNAL["UNKNOWN"])
                                if node_type_int == NODE_TYPES_INTERNAL["UNKNOWN"]: node_err += 1; warnings.warn(f"G{i}: Unknown node vec {type_tuple}")
                            else: node_err += 1; warnings.warn(f"G{i}: Node vec len {len(type_val)}!= {NUM_NODE_FEATURES_NEW}"); valid_graph = False; break
                        except Exception as e: node_err += 1; warnings.warn(f"G{i}: Node type conv err {e}"); valid_graph = False; break
                    elif isinstance(type_val, int) and type_val in NODE_TYPES_INTERNAL.values(): node_type_int = type_val # Allow pre-converted int
                    else: node_err += 1; warnings.warn(f"G{i}: Invalid node type format {type(type_val)}"); valid_graph = False; break
                else: node_err += 1; warnings.warn(f"G{i}: Node missing 'type'"); valid_graph = False; break
                g_processed.add_node(new_node_id, type=node_type_int) # Add node with INTERNAL int type
            if not valid_graph: skipped_count += 1; continue

            # Process Edges
            for u_old, v_old, edge_data in g_raw.edges(data=True):
                try: u_new, v_new = node_mapping[u_old], node_mapping[v_old]
                except KeyError: warnings.warn(f"G{i}: Edge node missing"); valid_graph = False; break
                edge_type_int = EDGE_TYPES_INTERNAL["REGULAR"] # Default
                if 'type' in edge_data:
                    type_val = edge_data['type']
                    if isinstance(type_val, (list, np.ndarray)): # Handle one-hot
                        try:
                            type_tuple = tuple(round(x, 1) for x in type_val[:NUM_EXPLICIT_EDGE_TYPES_NEW])
                            if len(type_tuple) == NUM_EXPLICIT_EDGE_TYPES_NEW:
                                edge_type_int = EDGE_TYPE_ENCODING_NEW.get(type_tuple, EDGE_TYPES_INTERNAL["REGULAR"])
                                if edge_type_int == EDGE_TYPES_INTERNAL["REGULAR"] and type_tuple not in EDGE_TYPE_ENCODING_NEW: edge_err += 1; warnings.warn(f"G{i}: Unknown edge vec {type_tuple}")
                            else: edge_err += 1; warnings.warn(f"G{i}: Edge vec len {len(type_val)}!= {NUM_EXPLICIT_EDGE_TYPES_NEW}")
                        except Exception as e: edge_err += 1; warnings.warn(f"G{i}: Edge type conv err {e}")
                    elif isinstance(type_val, int) and type_val in EDGE_TYPES_INTERNAL.values(): edge_type_int = type_val # Allow pre-converted int
                    else: edge_err += 1; warnings.warn(f"G{i}: Invalid edge type format {type(type_val)}")
                # Add edge with INTERNAL int type (0, 1, or 2)
                if u_new in g_processed and v_new in g_processed: g_processed.add_edge(u_new, v_new, type=edge_type_int)
                else: warnings.warn(f"G{i}: Skip edge ({u_new}-{v_new}) missing node"); valid_graph = False; break
            if not valid_graph: skipped_count += 1; continue

            processed_graphs.append(g_processed)

        print(f"\nPreprocessing complete. Processed {len(processed_graphs)} graphs.")
        print(f"Skipped {skipped_count} raw graphs.")
        if node_err > 0: print(f"Encountered {node_err} node type issues.")
        if edge_err > 0: print(f"Encountered {edge_err} edge type issues.")
        return processed_graphs

    def _calculate_and_set_edge_weights(self, train_split):
        """Calculates edge weights based on the training portion of the dataset."""
        edge_type_counts = Counter({i: 0 for i in range(NUM_EDGE_FEATURES_RNN)})
        total_potential_edges = 0
        num_graphs_for_weights = int(len(self.graphs) * train_split)
        if self.max_train_graphs is not None:
            num_graphs_for_weights = min(num_graphs_for_weights, self.max_train_graphs)

        print(f"Calculating edge weights based on first {num_graphs_for_weights} graphs...")
        for i in range(num_graphs_for_weights):
            g = self.graphs[i]
            n = g.number_of_nodes()
            if n <= 1: continue
            try: node_ordering = list(nx.topological_sort(g))
            except nx.NetworkXUnfeasible: continue
            node_to_idx = {node_id: i for i, node_id in enumerate(node_ordering)}
            adj_tensor = np.zeros((n, n, NUM_EDGE_FEATURES_RNN), dtype=np.float32)
            for u, v, data in g.edges(data=True):
                try:
                    source_idx, target_idx = node_to_idx[u], node_to_idx[v]
                    edge_type_int = data.get('type', EDGE_TYPES_INTERNAL["REGULAR"])
                    if 0 <= edge_type_int < NUM_EDGE_FEATURES_RNN: adj_tensor[target_idx, source_idx, edge_type_int] = 1.0
                    else: adj_tensor[target_idx, source_idx, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
                except KeyError: continue

            for target_idx in range(1, n):
                num_preds_considered = min(target_idx, self.m_internal)
                all_prev_connections = adj_tensor[target_idx, 0:target_idx, :]
                connections_slice = all_prev_connections[-num_preds_considered:, :] if num_preds_considered > 0 else np.zeros((0, NUM_EDGE_FEATURES_RNN))
                for k in range(num_preds_considered):
                    edge_class = np.argmax(connections_slice[k, :])
                    if np.sum(connections_slice[k, :]) == 0: edge_class = EDGE_TYPES_INTERNAL["NONE"] # Handle all-zero case -> NONE
                    if 0 <= edge_class < NUM_EDGE_FEATURES_RNN: edge_type_counts[edge_class] += 1; total_potential_edges += 1
                    else: warnings.warn(f"Invalid edge class {edge_class} in weight count.")
                padding_len = self.m_internal - num_preds_considered
                if padding_len > 0: edge_type_counts[EDGE_TYPES_INTERNAL["NONE"]] += padding_len; total_potential_edges += padding_len

        # Calculate weights
        total_edges_counted = sum(edge_type_counts.values())
        self.edge_weights = torch.ones(NUM_EDGE_FEATURES_RNN) # Default to uniform
        if total_edges_counted > 0:
            print(f"Total edge slots for weights: {total_potential_edges}")
            print(f"Raw edge counts (0=None, 1=Reg, 2=Inv): {dict(edge_type_counts)}")
            temp_weights = torch.zeros(NUM_EDGE_FEATURES_RNN)
            all_counts_positive = True
            for i in range(NUM_EDGE_FEATURES_RNN):
                if edge_type_counts[i] > 0: temp_weights[i] = 1.0 / edge_type_counts[i]
                else: all_counts_positive = False; temp_weights[i] = 0 # Handle zero count

            if all_counts_positive:
                sum_weights = torch.sum(temp_weights)
                if sum_weights > 1e-6: self.edge_weights = temp_weights / sum_weights * NUM_EDGE_FEATURES_RNN
                else: warnings.warn("Inverse weights sum zero. Using uniform.")
            else: # Fallback if some counts were zero
                warnings.warn(f"Zero counts for some edge types: {dict(edge_type_counts)}. Using fallback weights.")
                max_count = max(edge_type_counts.values()) if any(edge_type_counts.values()) else 1
                for i in range(NUM_EDGE_FEATURES_RNN):
                    temp_weights[i] = float(max_count) / edge_type_counts[i] if edge_type_counts[i] > 0 else float(max_count)
                sum_weights = torch.sum(temp_weights)
                if sum_weights > 1e-6: self.edge_weights = temp_weights / sum_weights * NUM_EDGE_FEATURES_RNN
                else: warnings.warn("Fallback weights sum zero. Using uniform.")
        else: warnings.warn("No edges found for weight calc. Using uniform.")
        print(f"Calculated edge weights: {self.edge_weights.tolist()}")


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Get a specific graph converted to sequence format for GraphRNN.
        Includes edge sequence (x), sequence length (len), node levels (levels),
        and target node types (y_node_type).
        """
        actual_idx = self.start_idx + idx
        if actual_idx >= len(self.graphs):
            raise IndexError(f"Index {idx} out of bounds for current dataset split.")
        g = self.graphs[actual_idx]

        n = g.number_of_nodes()
        effective_m = self.m_internal

        if effective_m is None:
            raise ValueError("self.m_internal not set. Dataset improperly initialized.")

        # Target padded length for all sequences
        target_padded_len = self.max_node_count - 1

        # Handle graphs too small or invalid
        if n <= 1:
            return {
                'x': torch.zeros((target_padded_len, effective_m, NUM_EDGE_FEATURES_RNN), dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long),
                'levels': torch.zeros(target_padded_len, dtype=torch.long),
                'y_node_type': torch.zeros(target_padded_len, dtype=torch.long)
            }

        # Get topological sort
        try:
            node_ordering = list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible:
            warnings.warn(f"Graph {idx} (original index {actual_idx}) not DAG. Returning dummy.")
            return {
                'x': torch.zeros((target_padded_len, effective_m, NUM_EDGE_FEATURES_RNN), dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long),
                'levels': torch.zeros(target_padded_len, dtype=torch.long),
                'y_node_type': torch.zeros(target_padded_len, dtype=torch.long)
            }

        if len(node_ordering) != n:
             warnings.warn(f"Ordering length mismatch graph {idx}. Returning dummy.")
             return {
                'x': torch.zeros((target_padded_len, effective_m, NUM_EDGE_FEATURES_RNN), dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long),
                'levels': torch.zeros(target_padded_len, dtype=torch.long),
                'y_node_type': torch.zeros(target_padded_len, dtype=torch.long)
            }
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ordering)}

        # Create Adjacency Tensor
        adj_tensor = np.zeros((n, n, NUM_EDGE_FEATURES_RNN), dtype=np.float32)
        adj_tensor[:, :, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
        for u, v, data in g.edges(data=True):
            try:
                source_idx, target_idx = node_to_idx[u], node_to_idx[v]
                edge_type_int = data.get('type', EDGE_TYPES_INTERNAL["REGULAR"])
                if 0 <= edge_type_int < NUM_EDGE_FEATURES_RNN:
                     adj_tensor[target_idx, source_idx, :] = 0.0
                     adj_tensor[target_idx, source_idx, edge_type_int] = 1.0
                else:
                    adj_tensor[target_idx, source_idx, :] = 0.0
                    adj_tensor[target_idx, source_idx, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
            except KeyError: continue

        # Create Edge Sequence 'x'
        sequence = []
        for i in range(1, n):
            target_node_idx = i
            all_prev_connections = adj_tensor[target_node_idx, 0:i, :]
            num_preds_available = all_prev_connections.shape[0]
            padding_len = max(0, effective_m - num_preds_available)
            num_preds_to_take = min(num_preds_available, effective_m)
            connections_slice = all_prev_connections[-num_preds_to_take:, :] if num_preds_to_take > 0 else np.zeros((0, NUM_EDGE_FEATURES_RNN))
            padded_connections = np.pad(connections_slice, ((padding_len, 0), (0, 0)), 'constant', constant_values=0)
            if padding_len > 0:
                padded_connections[:padding_len, :] = 0.0
                padded_connections[:padding_len, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
            padded_connections_reversed = padded_connections[::-1, :]
            sequence.append(padded_connections_reversed)

        # Convert edge sequence to padded tensor
        seq_len = len(sequence)
        if seq_len > 0: seq_tensor = torch.tensor(np.stack(sequence, axis=0), dtype=torch.float32)
        else: seq_tensor = torch.zeros((0, effective_m, NUM_EDGE_FEATURES_RNN), dtype=torch.float32)
        total_pad_len_x = max(0, target_padded_len - seq_len)
        padded_seq_tensor = torch.nn.functional.pad(seq_tensor, (0, 0, 0, 0, 0, total_pad_len_x))
        if total_pad_len_x > 0:
             pad_indices = torch.arange(seq_len, target_padded_len)
             padded_seq_tensor[pad_indices, :, :] = 0.0
             padded_seq_tensor[pad_indices, :, EDGE_TYPES_INTERNAL["NONE"]] = 1.0

        # Prepare Levels Tensor
        node_to_level = g.graph.get('levels', {})
        levels_ordered = [node_to_level.get(node_id, 0) for node_id in node_ordering]
        padded_levels_tensor = torch.zeros(target_padded_len, dtype=torch.long)
        if n > 1:
            levels_for_sequence = levels_ordered[1:]
            len_to_copy = min(len(levels_for_sequence), target_padded_len)
            levels_array = np.array(levels_for_sequence[:len_to_copy], dtype=np.int64)
            padded_levels_tensor[:len_to_copy] = torch.tensor(levels_array, dtype=torch.long)

        # Prepare Node Type Target Tensor
        padded_node_types_tensor = torch.zeros(target_padded_len, dtype=torch.long)
        if n > 1:
            node_types_for_sequence = []
            for node_idx_in_order in range(1, n):
                original_node_id = node_ordering[node_idx_in_order]
                node_data = g.nodes.get(original_node_id, {})
                node_type_int = node_data.get('type', NODE_TYPES_INTERNAL["UNKNOWN"])
                if node_type_int == NODE_TYPES_INTERNAL["UNKNOWN"]:
                    warnings.warn(f"Graph {idx}: Found UNKNOWN node type at index {node_idx_in_order} during target creation. Using 0.")
                    node_type_int = 0 # Default to CONST0 index
                node_types_for_sequence.append(node_type_int)
            len_to_copy_types = min(len(node_types_for_sequence), target_padded_len)
            if len_to_copy_types > 0:
                types_array = np.array(node_types_for_sequence[:len_to_copy_types], dtype=np.int64)
                padded_node_types_tensor[:len_to_copy_types] = torch.tensor(types_array, dtype=torch.long)

        # Prepare result dictionary
        result = {
            'x': padded_seq_tensor,
            'len': torch.tensor(seq_len, dtype=torch.long),
            'levels': padded_levels_tensor,
            'y_node_type': padded_node_types_tensor
        }

        return result

