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


class AIGDataset(torch.utils.data.Dataset): # Or inherit from DirectedGraphDataSet if needed

    def __init__(self,
                 graph_file: str,
                 m: Optional[int] = None, # M value for BFS mode
                 training: bool = True,
                 train_split: float = 0.8,
                 use_bfs: bool = True, # If False, uses Topological Sort
                 include_node_types: bool = False, # For potential future use
                 max_graphs: Optional[int] = None):
        """
        Initialize the AIG dataset.

        Args:
            graph_file: Path to the pickle file containing AIG graph data.
            m: Maximum lookback for BFS mode. Ignored if use_bfs is False. Required if use_bfs is True.
            training: Whether this is for training (True) or testing (False).
            train_split: Percentage of data to use for training.
            use_bfs: Whether to use BFS ordering (True) or topological sort (False).
            include_node_types: Whether to include node type info (requires model support).
            max_graphs: Maximum number of graphs to load (None for all).
        """
        # Basic setup
        self.dataset_type = 'aig-directed-multiclass'
        self.max_node_count = -1 # Will be determined from loaded graphs
        self.m = m # Store original m value passed
        self.use_bfs = use_bfs
        self.graph_file = graph_file
        self.include_node_types = include_node_types
        self.m_internal = None # Will store the effective M value

        # Validate M value based on mode
        if self.use_bfs and self.m is None:
             raise ValueError("Parameter 'm' must be provided when use_bfs=True.")
        if not self.use_bfs and self.m is not None:
             print(f"INFO: use_bfs is False (Topological Sort mode). Provided 'm' ({self.m}) will be ignored.")
             self.m = None # Explicitly set m to None if not used

        # Load raw graph data
        if training:
            print(f"Loading AIG graphs from {graph_file}...")
        if not os.path.exists(graph_file):
             raise FileNotFoundError(f"Dataset file not found: {graph_file}")
        with open(graph_file, 'rb') as f:
            self.raw_graphs = pickle.load(f)

        # Limit number of graphs
        if max_graphs is not None and max_graphs < len(self.raw_graphs):
            self.raw_graphs = self.raw_graphs[:max_graphs]
            print(f"Limited to {max_graphs} graphs")

        #print(f"Loaded {len(self.raw_graphs)} raw AIG graphs")

        # Preprocess graphs (convert types, etc.)
        # Ensure _preprocess_graphs handles potential errors and returns a list
        self.graphs = self._preprocess_graphs()

        # Determine maximum node count *after* preprocessing
        if not self.graphs:
             raise ValueError("No graphs loaded or preprocessed successfully.")
        for g in self.graphs:
            # Ensure g is a graph object before calling number_of_nodes
            if isinstance(g, nx.DiGraph):
                 self.max_node_count = max(self.max_node_count, g.number_of_nodes())
            else:
                 print(f"Warning: Item in self.graphs is not a DiGraph: {type(g)}. Skipping for max_node_count.")

        print(f"Maximum node count in processed dataset: {self.max_node_count}")
        if self.max_node_count <= 0:
             raise ValueError("Maximum node count is not positive. Check dataset and preprocessing.")

        # --- Set the internal effective M value ---
        if self.use_bfs:
             # For BFS, use the provided m
             self.m_internal = self.m
             print(f"INFO: BFS mode - Using internal M = {self.m_internal}")
        else:
             # For Topological Sort, M should effectively be max possible predecessors
             if self.max_node_count <= 1:
                  # Handle edge case of dataset with only single-node graphs
                  self.m_internal = 1 # Or 0, depending on how sequence is handled
                  #print(f"INFO: max_node_count is {self.max_node_count}. Setting internal M for TopSort to {self.m_internal}.")
             else:
                  self.m_internal = self.max_node_count - 1

        # Check if m_internal is set
        if self.m_internal is None or self.m_internal < 0:
            raise ValueError(f"Internal M (m_internal) calculation failed. Value: {self.m_internal}")

        # Set up train/test split
        np.random.seed(42) # Ensure consistent shuffle
        # Ensure self.graphs is a list before shuffling
        if isinstance(self.graphs, list):
            np.random.shuffle(self.graphs)
        else:
            print("Warning: self.graphs is not a list, cannot shuffle.")

        train_size = int(len(self.graphs) * train_split)
        self.start_idx = 0 if training else train_size
        self.length = train_size if training else len(self.graphs) - train_size
        if training:
            print(f"Dataset ready: {self.length} graphs ({'training' if training else 'testing'} split). Ordering: {'BFS' if use_bfs else 'Topological Sort'}")
        # print(f"Node type information will be {'included' if include_node_types else 'excluded'}")

    def _preprocess_graphs(self) -> List[nx.DiGraph]:
        """
        Process the raw graphs loaded from the pickle file into NetworkX DiGraphs
        with standardized integer node/edge types and node labels from 0 to N-1.

        Returns:
            List of preprocessed NetworkX DiGraphs.
        """
        processed_graphs = []
        skipped_count = 0
        num_raw_graphs = len(self.raw_graphs)

        #print(f"Preprocessing {num_raw_graphs} raw graphs...")

        for i, g_raw in enumerate(self.raw_graphs):
            # Basic check: Ensure it's a NetworkX graph object
            if not isinstance(g_raw, nx.Graph):  # Check for base Graph class
                print(f"\nWarning: Skipping item {i} - not a NetworkX graph object (type: {type(g_raw)}).")
                skipped_count += 1
                continue

            # Ensure it's directed if not already
            if not g_raw.is_directed():
                g_raw = g_raw.to_directed()  # Convert if undirected

            # Check for empty graph
            if g_raw.number_of_nodes() == 0:
                # print(f"\nWarning: Skipping graph {i} - empty graph.")
                skipped_count += 1
                continue

            # Create a new graph to store processed data, copying graph attributes
            g_processed = nx.DiGraph(**g_raw.graph)  # Copies graph-level attributes

            # Create a consistent node mapping from original IDs to 0..N-1 integers
            node_list = list(g_raw.nodes())  # Get a fixed order of original nodes
            node_mapping = {old_id: new_id for new_id, old_id in enumerate(node_list)}

            # Process nodes and add them with integer types
            for old_node_id in node_list:
                new_node_id = node_mapping[old_node_id]
                node_data = g_raw.nodes[old_node_id]

                # Determine integer node type
                node_type_int = -1  # Default or unknown type
                if 'type' in node_data:
                    type_val = node_data['type']
                    # Map from potential raw formats (adjust conditions as needed)
                    if isinstance(type_val, (list, np.ndarray)):
                        if np.array_equal(type_val, [0, 0, 0]):
                            node_type_int = NODE_TYPES.get("ZERO", 0)
                        elif np.array_equal(type_val, [1, 0, 0]):
                            node_type_int = NODE_TYPES.get("PI", 1)
                        elif np.array_equal(type_val, [0, 1, 0]):
                            node_type_int = NODE_TYPES.get("AND", 2)
                        elif np.array_equal(type_val, [0, 0, 1]):
                            node_type_int = NODE_TYPES.get("PO", 3)
                        # Add more elif conditions if other raw types exist
                    elif isinstance(type_val, int) and type_val in NODE_TYPES.values():
                        node_type_int = type_val  # Assume int type is already correct
                    # else: print(f"Warning: Unknown node type format {type_val} for node {old_node_id}")

                # Add the node to the processed graph with its new ID and integer type
                g_processed.add_node(
                    new_node_id,
                    type=node_type_int
                    # Add other original node attributes if needed: **{k:v for k,v in node_data.items() if k != 'type'}
                )

            # Process edges and add them with integer types
            for u_old, v_old, edge_data in g_raw.edges(data=True):
                # Map node IDs to the new 0..N-1 range
                try:
                    u_new, v_new = node_mapping[u_old], node_mapping[v_old]
                except KeyError:
                    print(f"\nWarning: Skipping edge ({u_old}, {v_old}) - node ID not found in mapping.")
                    continue

                # Determine integer edge type
                edge_type_int = EDGE_TYPES["REGULAR"]  # Default to regular
                if 'type' in edge_data:
                    type_val = edge_data['type']
                    # Map from potential raw formats (adjust conditions as needed)
                    if isinstance(type_val, (list, np.ndarray)):
                        # Example mapping based on user's previous code
                        if np.array_equal(type_val, [1, 0]):
                            edge_type_int = EDGE_TYPES["INVERTED"]
                        elif np.array_equal(type_val, [0, 1]):
                            edge_type_int = EDGE_TYPES["REGULAR"]
                        # Add more conditions if other raw formats exist
                    elif isinstance(type_val, int) and type_val in EDGE_TYPES.values():
                        edge_type_int = type_val  # Assume int type is already correct
                    # else: print(f"Warning: Unknown edge type format {type_val} for edge ({u_old},{v_old})")

                # Add the edge to the processed graph with its new IDs and integer type
                g_processed.add_edge(u_new, v_new, type=edge_type_int)
                # Add other original edge attributes if needed: **{k:v for k,v in edge_data.items() if k != 'type'}

            processed_graphs.append(g_processed)

            # Optional: Print progress less frequently
            # if (i + 1) % 100 == 0 or (i + 1) == num_raw_graphs:
            #     print(f"Processed {i + 1}/{num_raw_graphs} graphs...", end='\r')

        #print(f"\nGraph preprocessing complete. {len(processed_graphs)} graphs processed, {skipped_count} skipped.")
        return processed_graphs

    def __len__(self):
        """Return the number of graphs in the dataset split."""
        return self.length

    def __getitem__(self, idx):
        """
        Get a specific graph converted to the sequence format required by GraphRNN.
        Handles both BFS and Topological Sort orderings.
        """
        g = self.graphs[self.start_idx + idx]
        n = g.number_of_nodes()

        # Determine effective_m based on the mode set during initialization
        effective_m = self.m_internal

        if n <= 1:
            # Handle graphs with 0 or 1 node (cannot generate sequence)
            print(f"Warning: Graph {idx} has {n} nodes, returning dummy data.")
            dummy_m = effective_m if effective_m is not None else 1
            # Return empty tensors instead of None
            return {
                'x': torch.zeros((0, dummy_m, NUM_EDGE_FEATURES), dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long)
                # Don't include 'y' or 'node_types' keys at all
            }

        # 1. Get Node Ordering (BFS or Topological Sort)
        node_ordering = []
        if self.use_bfs:
            node_ordering = self.get_bfs_ordering(g)
        else:  # Topological sort
            try:
                node_ordering = list(nx.topological_sort(g))
            except nx.NetworkXUnfeasible:
                print(f"\nError: Graph {idx} is not a DAG, cannot perform topological sort. Skipping graph.")
                dummy_m = effective_m if effective_m is not None else 1
                # Return empty tensors instead of None
                return {
                    'x': torch.zeros((0, dummy_m, NUM_EDGE_FEATURES), dtype=torch.float),
                    'len': torch.tensor(0, dtype=torch.long)
                    # Don't include 'y' or 'node_types' keys at all
                }

        if len(node_ordering) != n:
            print(f"\nWarning: Ordering length ({len(node_ordering)}) mismatch with node count ({n}) for graph {idx}.")
            # Handle error appropriately, e.g., return dummy data

        # Create mapping from node ID to its index in the ordering
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ordering)}

        # 2. Create Adjacency Tensor with Edge Types (One-Hot)
        adj_tensor = np.zeros((n, n, NUM_EDGE_FEATURES), dtype=np.float32)
        for u, v, data in g.edges(data=True):
            try:
                source_idx = node_to_idx[u]
                target_idx = node_to_idx[v]
            except KeyError:
                print(f"\nWarning: Node ID {u} or {v} not found in ordering for graph {idx}. Skipping edge.")
                continue
            edge_type = data.get('type', EDGE_TYPES["REGULAR"])
            if 0 <= edge_type < NUM_EDGE_FEATURES:
                adj_tensor[target_idx, source_idx, edge_type] = 1.0
            else:
                print(f"\nWarning: Invalid edge type {edge_type} for edge ({u},{v}). Using type 0.")
                adj_tensor[target_idx, source_idx, 0] = 1.0

        # 3. Create Sequence for GraphRNN
        sequence = []
        max_seq_len = n - 1

        for i in range(1, n):  # Iterate for target nodes 1 to n-1
            target_node_idx = i
            all_prev_connections = adj_tensor[target_node_idx, 0:i, :]

            if self.use_bfs:
                # --- BFS Logic ---
                if effective_m is None:  # Should not happen if __init__ is correct
                    raise ValueError("self.m_internal (effective_m) is None in BFS mode.")
                start_idx = max(0, i - effective_m)
                connections_slice = all_prev_connections[start_idx:i, :]
                padding_len = effective_m - connections_slice.shape[0]
                padded_connections = np.pad(
                    connections_slice,
                    ((padding_len, 0), (0, 0)),
                    'constant', constant_values=0
                )[::-1, :]
            else:
                # --- Topological Sort Logic ---
                if effective_m is None:  # Should not happen if __init__ is correct
                    raise ValueError("self.m_internal (effective_m) is None in TopSort mode.")
                connections_slice = all_prev_connections  # Use all predecessors
                padding_len = effective_m - connections_slice.shape[0]  # Pad up to max_node_count-1
                padded_connections = np.pad(
                    connections_slice,
                    ((padding_len, 0), (0, 0)),
                    'constant', constant_values=0
                )[::-1, :]  # Reverse the order

            sequence.append(padded_connections)

        # Convert sequence list to tensor
        if sequence:
            sequence_array = np.stack(sequence, axis=0)
            seq_tensor = torch.tensor(sequence_array, dtype=torch.float32)
        else:
            # Use effective_m for shape consistency even if sequence is empty
            dummy_m = effective_m if effective_m is not None else 1
            seq_tensor = torch.zeros((0, dummy_m, NUM_EDGE_FEATURES), dtype=torch.float32)

        # 4. Pad the whole sequence tensor to max_node_count length
        seq_len = seq_tensor.shape[0]
        total_pad_len = (self.max_node_count - 1) - seq_len
        # Ensure padding is not negative
        if total_pad_len < 0:
            print(
                f"\nWarning: Negative padding calculated ({total_pad_len}) for graph {idx}. seq_len={seq_len}, max_node_count={self.max_node_count}. Clamping padding to 0.")
            total_pad_len = 0

        padded_seq_tensor = torch.nn.functional.pad(
            seq_tensor,
            (0, 0,  # Feature dim
             0, 0,  # Predecessor dim
             0, total_pad_len)  # Sequence length dim
        )

        # 5. Prepare result dictionary - ONLY include keys we absolutely need
        result = {
            'x': padded_seq_tensor,
            'len': torch.tensor(seq_len, dtype=torch.long)
        }

        # Conditionally include other data keys
        if self.include_node_types:
            node_types_ordered = [g.nodes[node_id].get('type', 0) for node_id in node_ordering]
            if len(node_types_ordered) > 1:
                node_types_array = np.array(node_types_ordered[1:], dtype=np.int64)
                node_types_tensor = torch.tensor(node_types_array, dtype=torch.long)
            else:
                node_types_tensor = torch.zeros(0, dtype=torch.long)

            node_type_pad_len = (self.max_node_count - 1) - node_types_tensor.shape[0]
            if node_type_pad_len < 0: node_type_pad_len = 0  # Clamp padding
            padded_node_types = torch.nn.functional.pad(
                node_types_tensor,
                (0, node_type_pad_len),
                value=0
            )
            result['node_types'] = padded_node_types

        # We're not using conditioning for now, so don't include 'y' key at all

        return result


    def get_bfs_ordering(self, g: nx.DiGraph) -> List[int]:
        """
        Get a BFS ordering of the nodes in a directed graph.

        Args:
            g: NetworkX directed graph

        Returns:
            List of node IDs in BFS order
        """
        # Find a node with no incoming edges to start BFS (ideally an input)
        start_nodes = []
        for node in g.nodes():
            if g.in_degree(node) == 0:
                start_nodes.append(node)

        # If no such node, just start from node 0
        if not start_nodes:
            start_nodes = [0]

        # Run BFS from start nodes
        visited = set()
        ordering = []

        for start in start_nodes:
            if start in visited:
                continue

            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue

                visited.add(node)
                ordering.append(node)

                # Add successors to queue
                for successor in g.successors(node):
                    if successor not in visited:
                        queue.append(successor)

        # Add any remaining nodes not visited by BFS
        for node in g.nodes():
            if node not in visited:
                ordering.append(node)

        return ordering