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


class AIGDataset(DirectedGraphDataSet):
    """
    Dataset for loading and processing AIGs for GraphRNN training.

    This class extends DirectedGraphDataSet from the GraphRNN codebase
    to handle AIG-specific data structures and converts them to the
    format required for training GraphRNN models.
    """

    def __init__(self,
                 graph_file: str,
                 m: Optional[int] = None,
                 training: bool = True,
                 train_split: float = 0.8,
                 use_bfs: bool = True,
                 include_node_types: bool = False,
                 max_graphs: Optional[int] = None):
        """
        Initialize the AIG dataset.

        Args:
            graph_file: Path to the pickle file containing AIG graph data
            m: Maximum number of previous nodes a node can connect to (M-value)
            training: Whether this is for training (True) or testing (False)
            train_split: Percentage of data to use for training
            use_bfs: Whether to use BFS ordering (True) or topological sort (False)
            include_node_types: Whether to include node type information in the dataset
            max_graphs: Maximum number of graphs to load (None for all)
        """
        self.dataset_type = 'aig-directed-multiclass'
        self.max_node_count = -1
        self.m = m
        self.use_bfs = use_bfs
        self.graph_file = graph_file
        self.include_node_types = include_node_types

        # Load raw graph data from pickle file
        print(f"Loading AIG graphs from {graph_file}...")
        with open(graph_file, 'rb') as f:
            self.raw_graphs = pickle.load(f)

        # Limit number of graphs if specified
        if max_graphs is not None and max_graphs < len(self.raw_graphs):
            self.raw_graphs = self.raw_graphs[:max_graphs]
            print(f"Limited to {max_graphs} graphs")

        print(f"Loaded {len(self.raw_graphs)} AIG graphs")

        # Convert to GraphRNN compatible format
        self.graphs = self._preprocess_graphs()

        # Determine maximum node count for padding
        for g in self.graphs:
            self.max_node_count = max(self.max_node_count, g.number_of_nodes())

        print(f"Maximum node count: {self.max_node_count}")

        # Standard initialization for train/test split
        np.random.seed(42)
        np.random.shuffle(self.graphs)

        train_size = int(len(self.graphs) * train_split)
        self.start_idx = 0 if training else train_size
        self.length = train_size if training else len(self.graphs) - train_size

        print(f"Dataset initialized with {self.length} {'training' if training else 'testing'} graphs")
        print(f"Node type information is {'included' if include_node_types else 'not included'}")

    def _preprocess_graphs(self) -> List[nx.DiGraph]:
        """
        Process the raw graphs into the format needed by GraphRNN.

        Returns:
            List of preprocessed NetworkX DiGraphs
        """
        processed_graphs = []

        for i, g in enumerate(self.raw_graphs):
            print(f"Processing graph {i+1}/{len(self.raw_graphs)}...", end='\r')

            # Create a new graph with the same properties
            new_g = nx.DiGraph(
                inputs=g.graph.get('inputs', 0),
                outputs=g.graph.get('outputs', 0),
                output_tts=g.graph.get('output_tts', [])
            )

            # Map node types from one-hot encoding to integer classes
            for node_id, node_data in g.nodes(data=True):
                # Determine node type based on one-hot encoding
                node_type = -1
                if 'type' in node_data:
                    type_arr = node_data['type']

                    # Map from your one-hot encoding to our integer classes
                    if np.array_equal(type_arr, [0, 0, 0]):
                        node_type = NODE_TYPES["ZERO"]
                    elif np.array_equal(type_arr, [1, 0, 0]):
                        node_type = NODE_TYPES["PI"]
                    elif np.array_equal(type_arr, [0, 1, 0]):
                        node_type = NODE_TYPES["AND"]
                    elif np.array_equal(type_arr, [0, 0, 1]):
                        node_type = NODE_TYPES["PO"]

                # Add node with integer type and original features
                new_g.add_node(
                    node_id,
                    type=node_type,
                    feature=node_data.get('feature', [])
                )

            # Map edge types from one-hot encoding to integer classes
            for source, target, edge_data in g.edges(data=True):
                # Default to regular edge if type not specified
                edge_type = EDGE_TYPES["REGULAR"]

                if 'type' in edge_data:
                    type_arr = edge_data['type']

                    # Map from your encoding to our integer classes
                    if np.array_equal(type_arr, [1, 0]):
                        edge_type = EDGE_TYPES["INVERTED"]
                    elif np.array_equal(type_arr, [0, 1]):
                        edge_type = EDGE_TYPES["REGULAR"]

                # Add edge with integer type
                new_g.add_edge(source, target, type=edge_type)

            processed_graphs.append(new_g)

        print("\nGraph preprocessing complete")
        return processed_graphs

    def __len__(self):
        """Return the number of graphs in the dataset split."""
        return self.length

    def __getitem__(self, idx):
        """
        Get a specific graph converted to the sequence format required by GraphRNN.
        Fixed version with performance optimization for tensor creation.

        Args:
            idx: Index of the graph to retrieve

        Returns:
            Dictionary containing:
            - 'x': Tensor of shape [seq_len, input_size, edge_feature_len]
                    containing the adjacency vectors for each node
            - 'len': The actual length of the sequence (number of nodes - 1)
            - 'y': Target truth table tensor if available
            - 'node_types': Node type labels if include_node_types is True
        """
        g = self.graphs[self.start_idx + idx]
        n = g.number_of_nodes()

        # Use either BFS or topological sort to get a node ordering
        if self.use_bfs:
            node_ordering = self.get_bfs_ordering(g)
        else:
            node_ordering = list(nx.topological_sort(g))

        # Create adjacency matrix with edge type information
        adj_matrix = np.zeros((n, n, 3))  # 3 edge types: none, regular, inverted

        # For each edge, set the corresponding entry in the adjacency matrix
        for u, v, data in g.edges(data=True):
            # Get positions in the ordering
            try:
                i = node_ordering.index(u)
                j = node_ordering.index(v)

                # Get edge type (default to regular if not specified)
                edge_type = data.get('type', EDGE_TYPES["REGULAR"])

                # Set the edge in the adjacency matrix
                adj_matrix[j, i, edge_type] = 1  # Directed edge from i to j
            except ValueError:
                # Skip edges where either node is not in the ordering
                continue

        # Create the sequence for GraphRNN
        # Each row represents connections to previous nodes
        sequence = []
        for i in range(1, n):
            # For each node, extract its connections to previous nodes
            start_idx = max(0, i - self.m)
            connections = adj_matrix[i, start_idx:i, :]

            # Pad to input_size and reverse (GraphRNN convention)
            padded = np.pad(
                connections,
                ((self.m - (i - start_idx), 0), (0, 0))
            )[::-1]

            sequence.append(padded)

        # PERFORMANCE FIX: Convert list of numpy arrays to a single numpy array first
        # This avoids the slow tensor creation warning
        if sequence:
            sequence_array = np.stack(sequence, axis=0)
            seq_tensor = torch.tensor(sequence_array, dtype=torch.float)
        else:
            # Handle empty sequence case
            seq_tensor = torch.zeros((0, self.m, 3), dtype=torch.float)

        # Pad sequence to max_node_count
        padded_tensor = torch.nn.functional.pad(
            seq_tensor,
            (0, 0, 0, 0, 0, self.max_node_count - len(sequence))
        )

        # Get output truth table if available
        output_tt = None
        if 'output_tts' in g.graph and g.graph['output_tts']:
            output_tt = torch.tensor(g.graph['output_tts'], dtype=torch.float)

        # Prepare result dictionary
        result = {
            'x': padded_tensor,
            'len': len(sequence),
            'y': output_tt  # Output truth table for conditioning
        }

        # Add node type information if required
        if self.include_node_types:
            # Extract node types from the ordered nodes
            node_types = []
            for node_id in node_ordering:
                node_type = g.nodes[node_id].get('type', 0)  # Default to ZERO if not specified
                node_types.append(node_type)

            # Convert to tensor and create correct format for GraphRNN
            # (skip the first node as GraphRNN starts from second node)
            if len(node_types) > 1:
                node_types_array = np.array(node_types[1:], dtype=np.int64)
                node_types_tensor = torch.tensor(node_types_array, dtype=torch.long)
            else:
                node_types_tensor = torch.zeros(0, dtype=torch.long)

            # Pad to max_node_count
            padded_node_types = torch.nn.functional.pad(
                node_types_tensor,
                (0, self.max_node_count - len(node_types_tensor)),
                value=0  # Pad with ZERO type
            )

            result['node_types'] = padded_node_types

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