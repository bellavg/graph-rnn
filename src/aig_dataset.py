"""
AIGDataset: Dataset implementation for training GraphRNN models on And-Inverter Graphs.

This module provides the AIGDataset class that extends GraphRNN's DirectedGraphDataSet
to handle AIG-specific structures and truth tables, with optional node type information.
"""

import os
import pickle
import numpy as np
import networkx as nx
import torch
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any, Union

from extension_data import DirectedGraphDataSet

# Node and edge type constants for AIGs
NODE_TYPES = {
    "ZERO": 0,
    "PI": 1,    # Primary Input
    "AND": 2,
    "PO": 3     # Primary Output
}

# Edge types: 0=no edge, 1=regular edge, 2=inverted edge
EDGE_TYPES = {
    "NONE": 0,
    "REGULAR": 1,
    "INVERTED": 2
}

NUM_EDGE_FEATURES = 3
NUM_NODE_TYPES = 4

def _calculate_levels(g: nx.DiGraph) -> (Dict[int, int], int):
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
                # Else: predecessor level unknown? This shouldn't happen in topo sort of DAG connected from sources.

            if current_max_pred_level != -1:
                levels[node] = current_max_pred_level + 1
                max_level = max(max_level, levels[node])
            else:
                # Node is unreachable from any source node with level 0? Assign default level 0.
                 print(f"Warning: Node {node} seems unreachable from a source node in level calculation. Assigning level 0.")
                 levels[node] = 0

    except nx.NetworkXUnfeasible:
        # Should not happen if preprocessing ensures DAGs, but handle defensively
        print("Warning: Graph provided to _calculate_levels is not a DAG. Returning default levels.")
        return {n: 0 for n in g.nodes()}, 0

    # Ensure all nodes have a level assigned (fallback for safety)
    for node in g.nodes():
        levels.setdefault(node, 0)

    return levels, max_level

import numpy as np
import torch
from collections import Counter # Import Counter
import networkx as nx # Ensure networkx is imported
from typing import List, Dict, Tuple, Optional, Any, Union # Ensure these are imported

# ... (other imports, constants like NODE_TYPES, EDGE_TYPES, NUM_EDGE_FEATURES, _calculate_levels) ...
class AIGDataset(torch.utils.data.Dataset):

    def __init__(self,
                 graph_file: str,
                 training: bool = True,
                 train_split: float = 0.9,
                 max_graphs: Optional[int] = None,
                 max_train_graphs: Optional[int] = None):
        """
        Initialize the AIG dataset using Topological Sort. Calculates node levels
        and edge type counts for class weighting.

        Args:
            graph_file: Path to the pickle file containing the graphs.
            training: Whether this dataset is for training (True) or testing (False).
            train_split: Fraction of data to use for training (between 0 and 1).
            include_node_types: Whether to include node type information.
            max_graphs: Maximum number of graphs to load in total.
            max_train_graphs: Maximum number of graphs to use for training.
                              Only applies when training=True. If None, uses train_split.
        """
        # Basic setup
        self.graph_file = graph_file
        self.m_internal = None  # Will be calculated later
        self.max_train_graphs = max_train_graphs

        # Load raw graph data
        print(f"Loading AIG graphs from {graph_file}...")
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Dataset file not found: {graph_file}")
        with open(graph_file, 'rb') as f:
            # Store raw graphs first
            self.raw_graphs_temp = pickle.load(f)  # Load into a temporary variable

        # Limit number of graphs if needed
        if max_graphs is not None and max_graphs < len(self.raw_graphs_temp):
            self.raw_graphs_temp = self.raw_graphs_temp[:max_graphs]
            print(f"Limited to {max_graphs} graphs for processing.")

        # Preprocess graphs (populates self.graphs)
        print("Preprocessing graphs...")
        self.graphs = self._preprocess_graphs()  # self.graphs is now List[nx.DiGraph]
        if not self.graphs:
            raise ValueError("No graphs loaded or preprocessed successfully.")
        # Clear temporary raw graphs if memory is a concern
        # del self.raw_graphs_temp

        # Determine maximum node count from processed graphs
        self.max_node_count = 0
        for g in self.graphs:
            if isinstance(g, nx.DiGraph): self.max_node_count = max(self.max_node_count, g.number_of_nodes())
        print(f"Maximum node count in processed dataset: {self.max_node_count}")
        if self.max_node_count <= 0: raise ValueError("Max node count <= 0.")

        # Calculate effective M for TopSort (NOW self.m_internal is set)
        self.m_internal = max(1, self.max_node_count - 1)  # Ensure m_internal >= 1
        print(f"INFO: Topological Sort mode. Effective input size (m_internal): {self.m_internal}")

        # Calculate levels and max_level (Requires self.graphs)
        print("Calculating node levels...")
        self.max_level = 0
        graphs_with_levels = []
        for i, g in enumerate(self.graphs):
            if isinstance(g, nx.DiGraph) and g.number_of_nodes() > 0:
                try:
                    # Calculate levels using the helper function
                    node_to_level, graph_max_level = _calculate_levels(g)
                    g.graph['levels'] = node_to_level  # Store levels in graph attributes
                    self.max_level = max(self.max_level, graph_max_level)
                    graphs_with_levels.append(g)  # Keep graphs where levels were calculated
                except Exception as e:
                    print(f"\nWarning: Failed to calculate levels for graph {i}: {e}. Skipping graph.")
            elif isinstance(g, nx.DiGraph) and g.number_of_nodes() == 0:
                print(f"Warning: Skipping empty graph {i}.")
            # Else: non-graph object handled in _preprocess_graphs

        self.graphs = graphs_with_levels  # Update self.graphs to only include valid ones
        if not self.graphs:
            raise ValueError("No graphs remaining after level calculation and filtering.")
        print(f"Maximum node level across dataset: {self.max_level}")
        # --- END Level Calculation ---

        # --- MOVED & CORRECTED: Calculate Edge Type Counts ---
        print("Calculating edge type counts for class weighting...")
        edge_type_counts = Counter({i: 0 for i in range(NUM_EDGE_FEATURES)})  # Initialize counter
        total_potential_edges = 0

        # Determine the range of graphs to use for weights (training split)
        final_num_graphs_before_split = len(self.graphs)  # Use length after level filtering

        # Calculate the train size based on either train_split or max_train_graphs
        train_size_by_split = int(final_num_graphs_before_split * train_split)
        if max_train_graphs is not None:
            train_size = min(train_size_by_split, max_train_graphs)
        else:
            train_size = train_size_by_split

        start_idx_weights = 0
        # Always calculate weights based on the training portion
        num_graphs_for_weights = train_size

        print(f"Calculating weights based on {num_graphs_for_weights} training graphs...")
        for i in range(num_graphs_for_weights):
            # Access graphs using the index directly, assumes self.graphs is the final list
            g = self.graphs[start_idx_weights + i]
            n = g.number_of_nodes()
            if n <= 1: continue

            try:
                node_ordering = list(nx.topological_sort(g))
            except nx.NetworkXUnfeasible:
                print(f"Warning: Graph {i} in weight calculation is not a DAG. Skipping.")
                continue  # Skip non-DAGs that might have slipped through

            node_to_idx = {node_id: i for i, node_id in enumerate(node_ordering)}
            # Recreate adj_tensor needed for counting
            adj_tensor = np.zeros((n, n, NUM_EDGE_FEATURES), dtype=np.float32)
            for u, v, data in g.edges(data=True):
                try:
                    source_idx, target_idx = node_to_idx[u], node_to_idx[v]
                except KeyError:
                    continue
                edge_type = data.get('type', EDGE_TYPES["REGULAR"])
                if 0 <= edge_type < NUM_EDGE_FEATURES:
                    adj_tensor[target_idx, source_idx, edge_type] = 1.0
                else:
                    adj_tensor[target_idx, source_idx, 0] = 1.0  # Default NO_EDGE

            # Count edges using self.m_internal
            for target_idx in range(1, n):  # Iterate over nodes 1 to n-1
                num_preds_considered = min(target_idx, self.m_internal)  # Use self.m_internal
                # Get the actual connections to the relevant predecessors
                all_prev_connections = adj_tensor[target_idx, 0:target_idx, :]
                connections_slice = all_prev_connections[-num_preds_considered:,
                                    :] if num_preds_considered > 0 else np.zeros((0, NUM_EDGE_FEATURES))

                # Iterate over the slice corresponding to relevant predecessors
                for k in range(num_preds_considered):
                    edge_class = np.argmax(connections_slice[k, :])  # Find the index (0, 1, or 2)
                    edge_type_counts[edge_class] += 1
                    total_potential_edges += 1

                # Count padding slots up to m_internal as "NONE" edges
                padding_len = self.m_internal - num_preds_considered  # Use self.m_internal
                if padding_len > 0:
                    edge_type_counts[EDGE_TYPES["NONE"]] += padding_len
                    total_potential_edges += padding_len

        # --- START Replace Weights Calculation ---
        total_edges_counted = sum(edge_type_counts.values())  # Use actual counted edges
        self.edge_weights = torch.zeros(NUM_EDGE_FEATURES)

        if total_edges_counted > 0 and all(count > 0 for count in edge_type_counts.values()):  # Check all counts > 0
            print(f"Total edge slots considered for weights: {total_potential_edges}")
            print(f"Raw edge counts: {dict(edge_type_counts)}")

            # --- Simple Inverse Frequency Weighting ---
            for i in range(NUM_EDGE_FEATURES):
                # weight = total_edges_counted / edge_type_counts[i] # Basic inverse frequency
                # Alternative: Use 1 / count, often works well
                weight = 1.0 / edge_type_counts[i]
                self.edge_weights[i] = weight
            # --- End Simple Inverse Frequency ---

            # Normalize weights so they sum to NUM_EDGE_FEATURES (optional, but can help stabilize)
            # Or simply normalize to sum to 1: self.edge_weights / torch.sum(self.edge_weights)
            self.edge_weights = self.edge_weights / torch.sum(self.edge_weights) * NUM_EDGE_FEATURES

            print(f"Calculated edge weights (Inverse Frequency): {self.edge_weights.tolist()}")

        elif total_edges_counted > 0:  # Handle if some classes have zero counts
            print(
                f"Warning: Zero counts detected for some edge types: {dict(edge_type_counts)}. Using fallback weights.")
            # Fallback: Give non-zero classes inverse weight, zero classes maybe weight 1? Or max weight?
            max_count = 1
            for i in range(NUM_EDGE_FEATURES):
                if edge_type_counts[i] > 0:
                    max_count = max(max_count, edge_type_counts[i])

            for i in range(NUM_EDGE_FEATURES):
                if edge_type_counts[i] > 0:
                    self.edge_weights[i] = float(max_count) / edge_type_counts[i]  # Relative inverse freq
                else:
                    self.edge_weights[i] = float(max_count)  # Assign max weight to zero-count class
            # Normalize
            self.edge_weights = self.edge_weights / torch.sum(self.edge_weights) * NUM_EDGE_FEATURES
            print(f"Calculated edge weights (Fallback for Zero Counts): {self.edge_weights.tolist()}")

        else:
            print(
                "Warning: No edges/padding found in training split to calculate weights. Using default weights [1, 1, 1].")
            self.edge_weights = torch.ones(NUM_EDGE_FEATURES)

        # Set up train/test split indices (needed for __len__ and __getitem__)
        np.random.seed(42)  # Ensure consistent shuffle for split definition
        np.random.shuffle(self.graphs)  # Shuffle the final list of graphs

        final_num_graphs = len(self.graphs)

        # Calculate train_size based on split or max_train_graphs
        train_size_by_split = int(final_num_graphs * train_split)
        if max_train_graphs is not None:
            train_size = min(train_size_by_split, max_train_graphs)
        else:
            train_size = train_size_by_split

        self.start_idx = 0 if training else train_size
        # Adjust length based on whether we want train or test split
        self.length = train_size if training else final_num_graphs - train_size

        # Log detailed information about the dataset split
        print(f"Dataset ready: {self.length} graphs ({'training' if training else 'testing'} split).")
        print(f"Total available graphs: {final_num_graphs}")
        print(
            f"Training graphs: {train_size} (limited by {'max_train_graphs' if max_train_graphs is not None and train_size < train_size_by_split else 'split ratio'})")
        print(f"Testing graphs: {final_num_graphs - train_size}")
        print(f"Ordering: Topological Sort")

    # The rest of the methods remain unchanged


    def _preprocess_graphs(self) -> List[nx.DiGraph]:
        # (Make sure this uses self.raw_graphs_temp as implemented previously)
        processed_graphs = []
        skipped_count = 0
        num_raw_graphs = len(self.raw_graphs_temp)

        for i, g_raw in enumerate(self.raw_graphs_temp):
            # Basic check: Ensure it's a NetworkX graph object
            if not isinstance(g_raw, nx.Graph):
                skipped_count += 1
                continue
            # Ensure it's directed if not already
            if not g_raw.is_directed():
                g_raw = g_raw.to_directed()
            # Check DAG property
            if not nx.is_directed_acyclic_graph(g_raw):
                skipped_count += 1
                continue
            # Check for empty graph
            if g_raw.number_of_nodes() == 0:
                skipped_count += 1
                continue

            # Create a new graph to store processed data
            g_processed = nx.DiGraph(**g_raw.graph)
            node_list = list(g_raw.nodes())
            node_mapping = {old_id: new_id for new_id, old_id in enumerate(node_list)}

            # Process nodes
            for old_node_id in node_list:
                new_node_id = node_mapping[old_node_id]
                node_data = g_raw.nodes[old_node_id]
                node_type_int = -1
                if 'type' in node_data:
                    type_val = node_data['type']
                    if isinstance(type_val, (list, np.ndarray)):
                        # Your existing type mapping logic here...
                        if np.array_equal(type_val, [0, 0, 0]):
                            node_type_int = NODE_TYPES.get("ZERO", 0)
                        elif np.array_equal(type_val, [1, 0, 0]):
                            node_type_int = NODE_TYPES.get("PI", 1)
                        elif np.array_equal(type_val, [0, 1, 0]):
                            node_type_int = NODE_TYPES.get("AND", 2)
                        elif np.array_equal(type_val, [0, 0, 1]):
                            node_type_int = NODE_TYPES.get("PO", 3)
                    elif isinstance(type_val, int) and type_val in NODE_TYPES.values():
                        node_type_int = type_val
                g_processed.add_node(new_node_id, type=node_type_int)

            # Process edges
            for u_old, v_old, edge_data in g_raw.edges(data=True):
                try:
                    u_new, v_new = node_mapping[u_old], node_mapping[v_old]
                except KeyError:
                    continue
                edge_type_int = EDGE_TYPES["REGULAR"]
                if 'type' in edge_data:
                    type_val = edge_data['type']
                    if isinstance(type_val, (list, np.ndarray)):
                        # Your existing type mapping logic here...
                        if np.array_equal(type_val, [1, 0]):
                            edge_type_int = EDGE_TYPES["INVERTED"]
                        elif np.array_equal(type_val, [0, 1]):
                            edge_type_int = EDGE_TYPES["REGULAR"]
                    elif isinstance(type_val, int) and type_val in EDGE_TYPES.values():
                        edge_type_int = type_val
                g_processed.add_edge(u_new, v_new, type=edge_type_int)

            processed_graphs.append(g_processed)

        print(f"Graph preprocessing complete. {len(processed_graphs)} graphs processed, {skipped_count} skipped.")
        return processed_graphs


    def __len__(self):
        """Return the number of graphs in the dataset split."""
        return self.length


    def __getitem__(self, idx):
        """
        Get a specific graph converted to the sequence format required by GraphRNN
        using Topological Sort ordering. Includes node level and node type information.
        """
        # --- Ensure self.graphs is accessed correctly based on self.start_idx ---
        actual_idx = self.start_idx + idx
        if actual_idx >= len(self.graphs):
            raise IndexError(f"Index {idx} out of bounds for current dataset split.")
        g = self.graphs[actual_idx]
        # --- End Indexing Fix ---

        n = g.number_of_nodes()
        effective_m = self.m_internal

        if n <= 1:
            # Return dummy data for small graphs
            dummy_m = effective_m if effective_m is not None else 1
            return {
                'x': torch.zeros((0, dummy_m, NUM_EDGE_FEATURES), dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long),
                'levels': torch.zeros(0, dtype=torch.long),
                'node_types': torch.zeros(0, dtype=torch.long) # Add dummy node_types
            }

        try:
            node_ordering = list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible:
            # Handle non-DAG error
            print(f"\nError: Graph {idx} (original index {actual_idx}) is not a DAG in getitem.")
            dummy_m = effective_m if effective_m is not None else 1
            return {
                'x': torch.zeros((0, dummy_m, NUM_EDGE_FEATURES), dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long),
                'levels': torch.zeros(0, dtype=torch.long),
                'node_types': torch.zeros(0, dtype=torch.long) # Add dummy node_types
            }

        if len(node_ordering) != n:
            # Handle length mismatch error
            print(f"\nWarning: Ordering length mismatch graph {idx} (original index {actual_idx}).")
            dummy_m = effective_m if effective_m is not None else 1
            return {
                'x': torch.zeros((0, dummy_m, NUM_EDGE_FEATURES), dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long),
                'levels': torch.zeros(0, dtype=torch.long),
                'node_types': torch.zeros(0, dtype=torch.long) # Add dummy node_types
            }
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ordering)}

        # Create Adjacency Tensor (code remains the same)
        adj_tensor = np.zeros((n, n, NUM_EDGE_FEATURES), dtype=np.float32)
        for u, v, data in g.edges(data=True):
            try:
                source_idx, target_idx = node_to_idx[u], node_to_idx[v]
            except KeyError:
                continue
            edge_type = data.get('type', EDGE_TYPES["REGULAR"])
            if 0 <= edge_type < NUM_EDGE_FEATURES:
                adj_tensor[target_idx, source_idx, edge_type] = 1.0
            else:
                adj_tensor[target_idx, source_idx, 0] = 1.0

        # Create Sequence (code remains the same)
        sequence = []
        if effective_m is None: raise ValueError("self.m_internal is None.")

        for i in range(1, n):
            target_node_idx = i
            all_prev_connections = adj_tensor[target_node_idx, 0:i, :]
            # Pad preceding connections based on effective_m
            connections_slice = all_prev_connections  # All predecessors up to i-1
            padding_len = effective_m - connections_slice.shape[0]
            if padding_len < 0: padding_len = 0
            # Pad at the beginning before reversing
            padded_connections = np.pad(
                connections_slice, ((padding_len, 0), (0, 0)),
                'constant', constant_values=0
            )[::-1, :]  # Reverse order
            sequence.append(padded_connections)

        # Convert sequence to tensor (code remains the same)
        if sequence:
            sequence_array = np.stack(sequence, axis=0)
            seq_tensor = torch.tensor(sequence_array, dtype=torch.float32)
        else:
            seq_tensor = torch.zeros((0, effective_m, NUM_EDGE_FEATURES), dtype=torch.float32)

        # Pad sequence tensor to max_node_count - 1 length (code remains the same)
        seq_len = seq_tensor.shape[0]
        target_padded_len = self.max_node_count - 1
        total_pad_len = target_padded_len - seq_len
        if total_pad_len < 0: total_pad_len = 0

        padded_seq_tensor = torch.nn.functional.pad(
            seq_tensor, (0, 0, 0, 0, 0, total_pad_len)
        )

        # Prepare Levels Tensor (code remains the same)
        node_to_level = g.graph.get('levels', {})  # Use default {}
        levels_ordered = [node_to_level.get(node_id, 0) for node_id in node_ordering]
        if n > 1:
            levels_for_sequence = levels_ordered[1:]
            levels_array = np.array(levels_for_sequence, dtype=np.int64)
            level_pad_len = target_padded_len - len(levels_array)
            if level_pad_len < 0: level_pad_len = 0
            padded_levels_array = np.pad(levels_array, (0, level_pad_len), 'constant', constant_values=0)
            padded_levels_tensor = torch.tensor(padded_levels_array, dtype=torch.long)
        else:
            padded_levels_tensor = torch.zeros(0, dtype=torch.long)

        # --- NEW: Prepare Node Types Tensor ---
        # Get the stored integer 'type' attribute for each node in the topological order
        node_types_ordered = [g.nodes[node_id].get('type', -1) for node_id in node_ordering] # Default to -1 if 'type' is missing

        if n > 1:
            # Slice to match the sequence length (excluding the first node)
            node_types_for_sequence = node_types_ordered[1:]
            node_types_array = np.array(node_types_for_sequence, dtype=np.int64)

            # Pad to the target length (max_node_count - 1)
            node_type_pad_len = target_padded_len - len(node_types_array)
            if node_type_pad_len < 0: node_type_pad_len = 0
            padded_node_types_array = np.pad(node_types_array, (0, node_type_pad_len), 'constant', constant_values=-1) # Pad with -1 (or another indicator)
            padded_node_types_tensor = torch.tensor(padded_node_types_array, dtype=torch.long)
        else:
            padded_node_types_tensor = torch.zeros(0, dtype=torch.long)
        # --- END NEW ---

        # Prepare result dictionary
        result = {
            'x': padded_seq_tensor,
            'len': torch.tensor(seq_len, dtype=torch.long),
            'levels': padded_levels_tensor,
            'node_types': padded_node_types_tensor # Add the node types tensor
        }

        return result