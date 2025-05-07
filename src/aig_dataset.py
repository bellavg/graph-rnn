"""
AIGDataset: Dataset implementation for training GraphRNN models on And-Inverter Graphs.

This module provides the AIGDataset class that handles AIG-specific structures
and features, loading from multiple PKL files and adapting to different
feature encodings during preprocessing. It prepares data for GraphRNN using
topological sorting and node level information.
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
    "ZERO": 0,
    "PI": 1,    # Primary Input
    "AND": 2,
    "PO": 3,    # Primary Output
    "UNKNOWN": -1 # Add UNKNOWN for internal use
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
    (1.0, 0.0, 0.0, 0.0): NODE_TYPES_INTERNAL["ZERO"], # Assuming index 0 is ZERO/CONST0
    (0.0, 1.0, 0.0, 0.0): NODE_TYPES_INTERNAL["PI"],   # Assuming index 1 is PI
    (0.0, 0.0, 1.0, 0.0): NODE_TYPES_INTERNAL["AND"],  # Assuming index 2 is AND
    (0.0, 0.0, 0.0, 1.0): NODE_TYPES_INTERNAL["PO"]    # Assuming index 3 is PO
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
    """
    Calculates the level of each node (longest path from a source node).
    Source nodes are nodes with in-degree 0.
    Requires the graph to be a DAG.

    Args:
        g: NetworkX directed acyclic graph.

    Returns:
        Tuple: (Dictionary mapping node_id to level, maximum level in graph)
               Returns ({}, -1) for empty graphs.
    """
    if not g:
        return {}, -1

    levels = {}
    max_level = 0
    source_nodes = [n for n in g.nodes() if g.in_degree(n) == 0]

    # Initialize levels for source nodes
    for node in source_nodes:
        levels[node] = 0

    # Process nodes in topological order
    try:
        for node in nx.topological_sort(g):
            if node in levels: # Already processed (source node)
                continue

            current_max_pred_level = -1
            for pred in g.predecessors(node):
                if pred in levels: # Ensure predecessor level is known
                    current_max_pred_level = max(current_max_pred_level, levels[pred])

            if current_max_pred_level != -1:
                levels[node] = current_max_pred_level + 1
                max_level = max(max_level, levels[node])
            else:
                 # Node is unreachable from any source node with level 0? Assign default level 0.
                 warnings.warn(f"Node {node} seems unreachable from a source node in level calculation. Assigning level 0.")
                 levels[node] = 0

    except nx.NetworkXUnfeasible:
        # Should not happen if preprocessing ensures DAGs, but handle defensively
        warnings.warn("Graph provided to _calculate_levels is not a DAG. Returning default levels.")
        return {n: 0 for n in g.nodes()}, 0

    # Ensure all nodes have a level assigned (fallback for safety)
    for node in g.nodes():
        levels.setdefault(node, 0)

    return levels, max_level


class AIGDataset(Dataset): # Inherit from torch.utils.data.Dataset

    def __init__(self,
                 graph_files: List[str], # MODIFIED: Accept list of files
                 training: bool = True,
                 train_split: float = 0.9,
                 max_graphs: Optional[int] = None,
                 max_train_graphs: Optional[int] = None):
        """
        Initialize the AIG dataset using Topological Sort from multiple PKL files.
        Adapts to new graph feature formats (one-hot nodes/edges). Calculates node levels
        and edge type counts for class weighting.

        Args:
            graph_files: List of paths to the pickle files containing the graphs.
            training: Whether this dataset is for training (True) or testing (False).
            train_split: Fraction of data to use for training (between 0 and 1).
            max_graphs: Maximum number of graphs to load in total from all files.
            max_train_graphs: Maximum number of graphs to use for training.
                              Only applies when training=True. If None, uses train_split.
        """
        # Basic setup
        self.graph_files = graph_files
        self.m_internal = None
        self.max_train_graphs = max_train_graphs

        # --- MODIFIED: Load raw graph data from multiple files ---
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
                         warnings.warn(f" Expected a list of graphs in {file_path}, got {type(graphs_in_file)}. Skipping file.")
             except Exception as e:
                 warnings.warn(f" Failed to load or process {file_path}: {e}. Skipping.")

        if not all_raw_graphs:
             raise ValueError("No graphs loaded from any provided file path.")

        self.raw_graphs_temp = all_raw_graphs
        # --- END Load ---

        # Limit number of graphs if needed
        if max_graphs is not None and max_graphs < len(self.raw_graphs_temp):
            random.shuffle(self.raw_graphs_temp) # Shuffle before slicing for random subset
            self.raw_graphs_temp = self.raw_graphs_temp[:max_graphs]
            print(f"Limited to a random subset of {max_graphs} graphs for processing.")

        # Preprocess graphs (populates self.graphs)
        print("Preprocessing graphs (adapting to new features)...")
        self.graphs = self._preprocess_graphs() # self.graphs is List[nx.DiGraph]
        if not self.graphs:
            raise ValueError("No graphs loaded or preprocessed successfully.")
        # Clear temporary raw graphs if memory is a concern
        del self.raw_graphs_temp

        # Determine maximum node count from processed graphs
        self.max_node_count = 64

        # Calculate effective M for TopSort (NOW self.m_internal is set)
        self.m_internal = max(1, self.max_node_count - 1)  # Ensure m_internal >= 1
        print(f"INFO: Topological Sort mode. Effective input size (m_internal): {self.m_internal}")

        # Calculate levels and max_level (Requires self.graphs)
        print("Calculating node levels...")
        self.max_level = 22
        graphs_with_levels = []
        for i, g in enumerate(self.graphs):
            if isinstance(g, nx.DiGraph) and g.number_of_nodes() > 0:
                try:
                    node_to_level, graph_max_level = _calculate_levels(g)
                    g.graph['levels'] = node_to_level
                    self.max_level = max(self.max_level, graph_max_level)
                    graphs_with_levels.append(g)
                except Exception as e:
                    print(f"\nWarning: Failed to calculate levels for graph {i}: {e}. Skipping graph.")
            elif isinstance(g, nx.DiGraph) and g.number_of_nodes() == 0:
                print(f"Warning: Skipping empty graph {i}.")

        self.graphs = graphs_with_levels
        if not self.graphs:
            raise ValueError("No graphs remaining after level calculation and filtering.")
        print(f"Maximum node level across dataset: {self.max_level}")

        # --- Calculate Edge Type Counts (Using internal 3-class representation) ---
        print("Calculating edge type counts for class weighting...")
        edge_type_counts = Counter({i: 0 for i in range(NUM_EDGE_FEATURES_RNN)}) # Use 3 features for weights
        total_potential_edges = 0

        # Determine the range of graphs to use for weights (training split)
        final_num_graphs_before_split = len(self.graphs)

        # Calculate the train size based on either train_split or max_train_graphs
        train_size_by_split = int(final_num_graphs_before_split * train_split)
        if max_train_graphs is not None:
            train_size = min(train_size_by_split, max_train_graphs)
        else:
            train_size = train_size_by_split

        start_idx_weights = 0
        num_graphs_for_weights = train_size

        print(f"Calculating weights based on {num_graphs_for_weights} training graphs...")
        for i in range(num_graphs_for_weights):
            g = self.graphs[start_idx_weights + i]
            n = g.number_of_nodes()
            if n <= 1: continue

            try:
                node_ordering = list(nx.topological_sort(g))
            except nx.NetworkXUnfeasible:
                warnings.warn(f"Graph {i} in weight calculation is not a DAG. Skipping.")
                continue

            node_to_idx = {node_id: i for i, node_id in enumerate(node_ordering)}
            # Recreate adj_tensor needed for counting (using the 3 internal edge types)
            adj_tensor = np.zeros((n, n, NUM_EDGE_FEATURES_RNN), dtype=np.float32)
            for u, v, data in g.edges(data=True):
                try:
                    source_idx, target_idx = node_to_idx[u], node_to_idx[v]
                    # --- Use the PREPROCESSED edge type (should be 0, 1, or 2) ---
                    edge_type_int = data.get('type', EDGE_TYPES_INTERNAL["REGULAR"])
                    # --- ---
                    if 0 <= edge_type_int < NUM_EDGE_FEATURES_RNN:
                        adj_tensor[target_idx, source_idx, edge_type_int] = 1.0
                    else:
                        warnings.warn(f"Invalid edge type {edge_type_int} found during weight calculation for graph {i}. Defaulting to NONE.")
                        adj_tensor[target_idx, source_idx, EDGE_TYPES_INTERNAL["NONE"]] = 1.0 # Use NONE index
                except KeyError:
                    continue # Skip if node not found (shouldn't happen after preprocessing)


            # Count edges using self.m_internal
            for target_idx in range(1, n):
                num_preds_considered = min(target_idx, self.m_internal)
                all_prev_connections = adj_tensor[target_idx, 0:target_idx, :]
                connections_slice = all_prev_connections[-num_preds_considered:, :] if num_preds_considered > 0 else np.zeros((0, NUM_EDGE_FEATURES_RNN))

                for k in range(num_preds_considered):
                    edge_class = np.argmax(connections_slice[k, :])
                    if 0 <= edge_class < NUM_EDGE_FEATURES_RNN:
                        edge_type_counts[edge_class] += 1
                        total_potential_edges += 1
                    else:
                         warnings.warn(f"Invalid edge class {edge_class} detected during weight counting. Skipping count.")


                padding_len = self.m_internal - num_preds_considered
                if padding_len > 0:
                    edge_type_counts[EDGE_TYPES_INTERNAL["NONE"]] += padding_len
                    total_potential_edges += padding_len

        # --- Calculate Weights (using same logic as before) ---
        total_edges_counted = sum(edge_type_counts.values())
        self.edge_weights = torch.zeros(NUM_EDGE_FEATURES_RNN)

        if total_edges_counted > 0 and all(count > 0 for count in edge_type_counts.values()):
            print(f"Total edge slots considered for weights: {total_potential_edges}")
            print(f"Raw edge counts (0=None, 1=Reg, 2=Inv): {dict(edge_type_counts)}")
            for i in range(NUM_EDGE_FEATURES_RNN):
                weight = 1.0 / edge_type_counts[i] if edge_type_counts[i] > 0 else 0 # Avoid division by zero
                self.edge_weights[i] = weight
            # Normalize weights
            sum_weights = torch.sum(self.edge_weights)
            if sum_weights > 1e-6:
                 self.edge_weights = self.edge_weights / sum_weights * NUM_EDGE_FEATURES_RNN
            else:
                 warnings.warn("Sum of inverse weights is zero. Using uniform weights.")
                 self.edge_weights = torch.ones(NUM_EDGE_FEATURES_RNN)
            print(f"Calculated edge weights (Inverse Frequency): {self.edge_weights.tolist()}")

        elif total_edges_counted > 0:
            warnings.warn(f"Zero counts detected for some edge types: {dict(edge_type_counts)}. Using fallback weights.")
            max_count = 1
            for i in range(NUM_EDGE_FEATURES_RNN):
                if edge_type_counts[i] > 0: max_count = max(max_count, edge_type_counts[i])
            for i in range(NUM_EDGE_FEATURES_RNN):
                if edge_type_counts[i] > 0: self.edge_weights[i] = float(max_count) / edge_type_counts[i]
                else: self.edge_weights[i] = float(max_count)
            sum_weights = torch.sum(self.edge_weights)
            if sum_weights > 1e-6:
                 self.edge_weights = self.edge_weights / sum_weights * NUM_EDGE_FEATURES_RNN
            else:
                 warnings.warn("Sum of fallback weights is zero. Using uniform weights.")
                 self.edge_weights = torch.ones(NUM_EDGE_FEATURES_RNN)
            print(f"Calculated edge weights (Fallback for Zero Counts): {self.edge_weights.tolist()}")
        else:
            warnings.warn("No edges/padding found in training split to calculate weights. Using default weights [1, 1, 1].")
            self.edge_weights = torch.ones(NUM_EDGE_FEATURES_RNN)
        # --- End Weights ---

        # Set up train/test split indices
        np.random.seed(42)
        np.random.shuffle(self.graphs)

        final_num_graphs = len(self.graphs)

        train_size_by_split = int(final_num_graphs * train_split)
        if max_train_graphs is not None:
            train_size = min(train_size_by_split, max_train_graphs)
        else:
            train_size = train_size_by_split

        self.start_idx = 0 if training else train_size
        self.length = train_size if training else final_num_graphs - train_size

        print(f"Dataset ready: {self.length} graphs ({'training' if training else 'testing'} split).")
        print(f"Total available graphs: {final_num_graphs}")
        print(
            f"Training graphs: {train_size} (limited by {'max_train_graphs' if max_train_graphs is not None and train_size < train_size_by_split else 'split ratio'})")
        print(f"Testing graphs: {final_num_graphs - train_size}")
        print(f"Ordering: Topological Sort")

    def _preprocess_graphs(self) -> List[nx.DiGraph]:
        """
        Converts raw graphs (potentially with new one-hot features) into NetworkX DiGraphs
        with INTERNAL integer representations for node and edge types.
        """
        processed_graphs = []
        skipped_count = 0
        node_type_conversion_errors = 0
        edge_type_conversion_errors = 0
        num_raw_graphs = len(self.raw_graphs_temp)

        print(f"Starting preprocessing of {num_raw_graphs} raw graphs...")

        for i, g_raw in enumerate(self.raw_graphs_temp):
            if i % 5000 == 0 and i > 0: print(f" Preprocessed {i}/{num_raw_graphs} graphs...")

            # Basic graph checks
            if not isinstance(g_raw, nx.Graph):
                skipped_count += 1; continue
            if not g_raw.is_directed(): g_raw = g_raw.to_directed()
            if not nx.is_directed_acyclic_graph(g_raw):
                skipped_count += 1; continue
            if g_raw.number_of_nodes() == 0:
                skipped_count += 1; continue

            # Create new graph, remap nodes to 0..N-1 integers
            g_processed = nx.DiGraph(**g_raw.graph)
            node_list = list(g_raw.nodes())
            node_mapping = {old_id: new_id for new_id, old_id in enumerate(node_list)}

            valid_graph = True
            # Process nodes: map NEW one-hot format to INTERNAL integer format
            for old_node_id in node_list:
                new_node_id = node_mapping[old_node_id]
                node_data = g_raw.nodes[old_node_id]
                node_type_int = NODE_TYPES_INTERNAL["UNKNOWN"] # Default

                if 'type' in node_data:
                    type_val = node_data['type']
                    # --- NEW NODE TYPE HANDLING ---
                    if isinstance(type_val, (list, np.ndarray)):
                        try:
                            # Convert to tuple, handle float precision, slice to expected length
                            type_tuple = tuple(round(x, 1) for x in type_val[:NUM_NODE_FEATURES_NEW])
                            # Check length AFTER slicing
                            if len(type_tuple) == NUM_NODE_FEATURES_NEW:
                                node_type_int = NODE_TYPE_ENCODING_NEW.get(type_tuple, NODE_TYPES_INTERNAL["UNKNOWN"])
                                if node_type_int == NODE_TYPES_INTERNAL["UNKNOWN"]:
                                    node_type_conversion_errors += 1
                                    warnings.warn(f"Graph {i}: Unknown node type vector {type_tuple} for node {old_node_id}. Treating as UNKNOWN.")
                            else:
                                node_type_conversion_errors += 1
                                warnings.warn(f"Graph {i}: Node type vector {type_val} has wrong length ({len(type_val)} vs {NUM_NODE_FEATURES_NEW}) for node {old_node_id}. Skipping graph.")
                                valid_graph = False; break
                        except Exception as e:
                            node_type_conversion_errors += 1
                            warnings.warn(f"Graph {i}: Error converting node type {type_val} for node {old_node_id}: {e}. Skipping graph.")
                            valid_graph = False; break
                    else: # Allow direct integer type if already matches internal format
                        if isinstance(type_val, int) and type_val in NODE_TYPES_INTERNAL.values():
                             node_type_int = type_val
                        else:
                             node_type_conversion_errors += 1
                             warnings.warn(f"Graph {i}: Invalid node type format {type(type_val)} for node {old_node_id}. Skipping graph.")
                             valid_graph = False; break
                    # --- END ---
                else:
                    node_type_conversion_errors += 1
                    warnings.warn(f"Graph {i}: Node {old_node_id} missing 'type' attribute. Skipping graph.")
                    valid_graph = False; break # Skip this graph

                # Add node with the internal integer type
                g_processed.add_node(new_node_id, type=node_type_int)

            if not valid_graph:
                skipped_count += 1
                continue # Move to next graph

            # Process edges: map NEW one-hot format to INTERNAL integer format
            for u_old, v_old, edge_data in g_raw.edges(data=True):
                try:
                    u_new, v_new = node_mapping[u_old], node_mapping[v_old]
                except KeyError:
                    warnings.warn(f"Graph {i}: Edge ({u_old}-{v_old}) refers to node not in mapping. Skipping graph.")
                    valid_graph = False; break

                # Default to REGULAR if type missing, matches old behavior better than NONE
                edge_type_int = EDGE_TYPES_INTERNAL["REGULAR"]

                if 'type' in edge_data:
                    type_val = edge_data['type']
                    # --- NEW EDGE TYPE HANDLING ---
                    if isinstance(type_val, (list, np.ndarray)):
                        try:
                             # Convert to tuple, handle floats, slice to expected length
                             type_tuple = tuple(round(x, 1) for x in type_val[:NUM_EXPLICIT_EDGE_TYPES_NEW])
                             if len(type_tuple) == NUM_EXPLICIT_EDGE_TYPES_NEW:
                                 # Map new encoding to internal int, default to REGULAR on lookup miss
                                 edge_type_int = EDGE_TYPE_ENCODING_NEW.get(type_tuple, EDGE_TYPES_INTERNAL["REGULAR"])
                                 if edge_type_int == EDGE_TYPES_INTERNAL["REGULAR"] and type_tuple not in EDGE_TYPE_ENCODING_NEW:
                                      edge_type_conversion_errors += 1
                                      warnings.warn(f"Graph {i}: Unknown edge type vector {type_tuple} for edge ({u_old}-{v_old}). Treating as REGULAR.")
                             else:
                                  edge_type_conversion_errors += 1
                                  warnings.warn(f"Graph {i}: Edge type vector {type_val} has wrong length ({len(type_val)} vs {NUM_EXPLICIT_EDGE_TYPES_NEW}) for edge ({u_old}-{v_old}). Treating as REGULAR.")
                        except Exception as e:
                             edge_type_conversion_errors += 1
                             warnings.warn(f"Graph {i}: Error converting edge type {type_val} for edge ({u_old}-{v_old}): {e}. Treating as REGULAR.")
                    elif isinstance(type_val, int) and type_val in EDGE_TYPES_INTERNAL.values():
                        # Allow direct integer type if already matches internal format
                        edge_type_int = type_val
                    else:
                        edge_type_conversion_errors += 1
                        warnings.warn(f"Graph {i}: Invalid edge type format {type(type_val)} for edge ({u_old}-{v_old}). Treating as REGULAR.")
                    # --- END ---
                else:
                    # If 'type' missing, assume REGULAR
                    # warnings.warn(f"Graph {i}: Edge ({u_old}-{v_old}) missing 'type' attribute. Assuming REGULAR.")
                    pass # Already defaulted to REGULAR

                # Add edge with the internal integer type (0, 1, or 2)
                # Check if nodes exist before adding edge
                if u_new in g_processed and v_new in g_processed:
                    g_processed.add_edge(u_new, v_new, type=edge_type_int)
                else:
                    warnings.warn(f"Graph {i}: Skipping edge ({u_new}-{v_new}) due to missing node after remapping.")
                    valid_graph = False; break


            if not valid_graph: # Check again if edge processing failed
                skipped_count += 1
                continue

            processed_graphs.append(g_processed)

        print(f"\nGraph preprocessing complete. Processed {len(processed_graphs)} graphs.")
        print(f"Skipped {skipped_count} raw graphs (invalid format, not DAG, empty, or feature error).")
        if node_type_conversion_errors > 0:
             print(f"Encountered {node_type_conversion_errors} node type conversion issues (affected graphs skipped or defaulted).")
        if edge_type_conversion_errors > 0:
             print(f"Encountered {edge_type_conversion_errors} edge type conversion issues (affected edges defaulted).")
        return processed_graphs


    def __len__(self):
        return self.length

    # __getitem__ remains the same as it uses the internally stored integer edge types
    # and NUM_EDGE_FEATURES_RNN (=3) to construct the one-hot sequence `x`
    def __getitem__(self, idx):
        """
        Get a specific graph converted to the sequence format required by GraphRNN
        using Topological Sort ordering. Includes node level information.
        """
        actual_idx = self.start_idx + idx
        if actual_idx >= len(self.graphs):
            raise IndexError(f"Index {idx} out of bounds for current dataset split.")
        g = self.graphs[actual_idx]

        n = g.number_of_nodes()
        effective_m = self.m_internal

        # Ensure m_internal is set
        if effective_m is None:
            raise ValueError("self.m_internal has not been calculated. Dataset might be improperly initialized.")

        if n <= 1:
            dummy_m = effective_m
            return {
                'x': torch.zeros((0, dummy_m, NUM_EDGE_FEATURES_RNN), dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long),
                'levels': torch.zeros(0, dtype=torch.long)
            }

        try:
            node_ordering = list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible:
            warnings.warn(f"Graph {idx} (original index {actual_idx}) is not a DAG in getitem. Returning dummy.")
            dummy_m = effective_m
            return {
                'x': torch.zeros((0, dummy_m, NUM_EDGE_FEATURES_RNN), dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long),
                'levels': torch.zeros(0, dtype=torch.long)
            }

        if len(node_ordering) != n:
            warnings.warn(f"Ordering length mismatch graph {idx} (original index {actual_idx}). Returning dummy.")
            dummy_m = effective_m
            return {
                'x': torch.zeros((0, dummy_m, NUM_EDGE_FEATURES_RNN), dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long),
                'levels': torch.zeros(0, dtype=torch.long)
            }
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ordering)}

        # Create Adjacency Tensor (using NUM_EDGE_FEATURES_RNN = 3)
        adj_tensor = np.zeros((n, n, NUM_EDGE_FEATURES_RNN), dtype=np.float32)
        # Initialize the 'NONE' channel to 1, others to 0
        adj_tensor[:, :, EDGE_TYPES_INTERNAL["NONE"]] = 1.0

        for u, v, data in g.edges(data=True):
            try:
                source_idx, target_idx = node_to_idx[u], node_to_idx[v]
                edge_type_int = data.get('type', EDGE_TYPES_INTERNAL["REGULAR"]) # Should be 0, 1, or 2 now

                if 0 <= edge_type_int < NUM_EDGE_FEATURES_RNN:
                     # Set the appropriate channel to 1 and the 'NONE' channel to 0
                     adj_tensor[target_idx, source_idx, :] = 0.0 # Zero out all channels first
                     adj_tensor[target_idx, source_idx, edge_type_int] = 1.0
                else:
                    # If invalid type somehow got through preprocessing, keep NONE channel as 1
                    adj_tensor[target_idx, source_idx, :] = 0.0
                    adj_tensor[target_idx, source_idx, EDGE_TYPES_INTERNAL["NONE"]] = 1.0
            except KeyError:
                continue # Skip if node mapping failed (shouldn't happen)

        # Create Sequence
        sequence = []
        for i in range(1, n):
            target_node_idx = i
            # Get connections FROM predecessors TO current node (target_idx)
            all_prev_connections = adj_tensor[target_node_idx, 0:i, :] # Shape [i, num_features]

            # Determine padding and slice
            num_preds_available = all_prev_connections.shape[0]
            padding_len = max(0, effective_m - num_preds_available)
            num_preds_to_take = min(num_preds_available, effective_m)

            # Slice the connections from the end (most recent predecessors relative to the current node i)
            # These are connections from nodes i-1, i-2, ..., i-num_preds_to_take
            connections_slice = all_prev_connections[-num_preds_to_take:, :] if num_preds_to_take > 0 else np.zeros((0, NUM_EDGE_FEATURES_RNN))

            # Pad at the beginning (representing connections from nodes further back than m)
            padded_connections = np.pad(
                connections_slice, ((padding_len, 0), (0, 0)),
                'constant', constant_values=0
            )
            # Ensure the 'NONE' feature is 1 for padded entries
            if padding_len > 0:
                padded_connections[:padding_len, :] = 0.0 # Zero out features
                padded_connections[:padding_len, EDGE_TYPES_INTERNAL["NONE"]] = 1.0 # Set NONE type

            # Reverse the order so index 0 corresponds to connection from node i-1, index 1 from i-2, ..., index m-1 from i-m
            padded_connections_reversed = padded_connections[::-1, :]
            sequence.append(padded_connections_reversed)

        # Convert sequence to tensor
        seq_len = len(sequence) # Actual sequence length (n-1 if n>0, else 0)
        if seq_len > 0:
            sequence_array = np.stack(sequence, axis=0)
            seq_tensor = torch.tensor(sequence_array, dtype=torch.float32)
        else:
            # Handle case where n=1 (no sequence generated)
            seq_tensor = torch.zeros((0, effective_m, NUM_EDGE_FEATURES_RNN), dtype=torch.float32)


        # Pad sequence tensor to max_node_count - 1 length
        target_padded_len = self.max_node_count - 1
        total_pad_len = max(0, target_padded_len - seq_len)

        padded_seq_tensor = torch.nn.functional.pad(
            seq_tensor, (0, 0, 0, 0, 0, total_pad_len) # Pad sequence dimension (dim 0)
        )
        # --- Fill padding with 'NONE' edge type ---
        if total_pad_len > 0:
             pad_indices = torch.arange(seq_len, target_padded_len)
             padded_seq_tensor[pad_indices, :, :] = 0.0 # Zero out features
             padded_seq_tensor[pad_indices, :, EDGE_TYPES_INTERNAL["NONE"]] = 1.0 # Set NONE type
        # --- End Padding Fill ---

        # Prepare Levels Tensor (padding logic simplified)
        node_to_level = g.graph.get('levels', {})
        levels_ordered = [node_to_level.get(node_id, 0) for node_id in node_ordering]
        padded_levels_tensor = torch.zeros(target_padded_len, dtype=torch.long) # Initialize with zeros
        if n > 1:
            levels_for_sequence = levels_ordered[1:] # Levels for nodes 1 to n-1
            len_to_copy = min(len(levels_for_sequence), target_padded_len)
            levels_array = np.array(levels_for_sequence[:len_to_copy], dtype=np.int64)
            padded_levels_tensor[:len_to_copy] = torch.tensor(levels_array, dtype=torch.long)

        # Prepare result dictionary
        result = {
            'x': padded_seq_tensor,
            'len': torch.tensor(seq_len, dtype=torch.long),
            'levels': padded_levels_tensor
        }

        return result
