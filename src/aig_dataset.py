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

class AIGDataset(torch.utils.data.Dataset):

    # --- MODIFIED __init__ ---
    def __init__(self,
                 graph_file: str,
                 training: bool = True,      # Removed m, use_bfs defaults to False
                 train_split: float = 0.9,
                 include_node_types: bool = False,
                 max_graphs: Optional[int] = None):
        """
        Initialize the AIG dataset using Topological Sort. Calculates node levels.

        Args:
            graph_file: Path to the pickle file containing AIG graph data.
            training: Whether this is for training (True) or testing (False).
            train_split: Percentage of data to use for training.
            include_node_types: Whether to include node type info (requires model support).
            max_graphs: Maximum number of graphs to load (None for all).
        """
        # Basic setup
        self.dataset_type = 'aig-directed-multiclass'
        self.max_node_count = -1
        self.use_bfs = False # Hardcoded as per user request
        self.graph_file = graph_file
        self.include_node_types = include_node_types
        self.m_internal = None # Effective M for TopSort = max_nodes - 1

        #print("INFO: Initializing AIGDataset for Topological Sort only.")

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

        # Preprocess graphs
        self.graphs = self._preprocess_graphs()

        # Determine maximum node count
        if not self.graphs:
             raise ValueError("No graphs loaded or preprocessed successfully.")
        for g in self.graphs:
            if isinstance(g, nx.DiGraph):
                 self.max_node_count = max(self.max_node_count, g.number_of_nodes())
            else:
                 print(f"Warning: Item in self.graphs is not a DiGraph: {type(g)}. Skipping for max_node_count.")
        print(f"Maximum node count in processed dataset: {self.max_node_count}")
        if self.max_node_count <= 0:
             raise ValueError("Maximum node count is not positive. Check dataset and preprocessing.")

        # Calculate effective M for TopSort
        if self.max_node_count <= 1:
             self.m_internal = 1
        else:
             self.m_internal = self.max_node_count - 1
        print(f"INFO: Topological Sort mode. Effective input size (max_nodes-1): {self.m_internal}")

        # --- Calculate levels and max_level ---
        #print("Calculating node levels for all graphs...")
        self.max_level = 0
        graphs_with_levels = []
        for i, g in enumerate(self.graphs):
            if isinstance(g, nx.DiGraph) and g.number_of_nodes() > 0:
                try:
                    # Calculate levels using the helper function
                    node_to_level, graph_max_level = _calculate_levels(g)
                    g.graph['levels'] = node_to_level # Store levels in graph attributes
                    self.max_level = max(self.max_level, graph_max_level)
                    graphs_with_levels.append(g) # Keep graphs where levels were calculated
                except Exception as e:
                    print(f"\nWarning: Failed to calculate levels for graph {i}: {e}. Skipping graph.")
            elif isinstance(g, nx.DiGraph) and g.number_of_nodes() == 0:
                 print(f"Warning: Skipping empty graph {i}.")
            # Else: non-graph object already warned about earlier

        self.graphs = graphs_with_levels # Update self.graphs to only include valid ones
        if not self.graphs:
            raise ValueError("No graphs remaining after level calculation and filtering.")
        #print(f"Maximum node level across dataset: {self.max_level}")
        # --- END Level Calculation ---

        # Set up train/test split
        np.random.seed(42)
        np.random.shuffle(self.graphs) # Now self.graphs is guaranteed to be a list
        train_size = int(len(self.graphs) * train_split)
        self.start_idx = 0 if training else train_size
        self.length = train_size if training else len(self.graphs) - train_size
        print(f"Dataset ready: {self.length} graphs ({'training' if training else 'testing'} split). Ordering: Topological Sort")


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

            if not nx.is_directed_acyclic_graph(g_raw):
                print(f"\nWarning: Skipping graph {i} - not a DAG.")
                skipped_count += 1
                continue

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
        Get a specific graph converted to the sequence format required by GraphRNN
        using Topological Sort ordering. Includes node level information.
        """
        g = self.graphs[self.start_idx + idx]
        n = g.number_of_nodes()
        effective_m = self.m_internal # Should be max_node_count - 1

        # Handle graphs with 0 or 1 node
        if n <= 1:
            print(f"Warning: Graph {idx} has {n} nodes, returning dummy data.")
            dummy_m = effective_m if effective_m is not None else 1
            return {
                'x': torch.zeros((0, dummy_m, NUM_EDGE_FEATURES), dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long),
                'levels': torch.zeros(0, dtype=torch.long)
            }

        # 1. Get Topological Sort Node Ordering
        try:
            # Ensure we use a specific topological sort if needed, otherwise default is fine
            node_ordering = list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible:
             # This should have been caught in preprocessing, but handle again
             print(f"\nError: Graph {idx} is not a DAG in getitem. Skipping graph.")
             dummy_m = effective_m if effective_m is not None else 1
             return {
                 'x': torch.zeros((0, dummy_m, NUM_EDGE_FEATURES), dtype=torch.float),
                 'len': torch.tensor(0, dtype=torch.long),
                 'levels': torch.zeros(0, dtype=torch.long)
             }

        if len(node_ordering) != n:
             print(f"\nWarning: Ordering length mismatch graph {idx}.")
             # Handle error, e.g., return dummy data
             dummy_m = effective_m if effective_m is not None else 1
             return {
                'x': torch.zeros((0, dummy_m, NUM_EDGE_FEATURES), dtype=torch.float),
                'len': torch.tensor(0, dtype=torch.long),
                'levels': torch.zeros(0, dtype=torch.long)
             }
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ordering)}

        # 2. Create Adjacency Tensor (same as before)
        adj_tensor = np.zeros((n, n, NUM_EDGE_FEATURES), dtype=np.float32)
        for u, v, data in g.edges(data=True):
            try: source_idx, target_idx = node_to_idx[u], node_to_idx[v]
            except KeyError: continue
            edge_type = data.get('type', EDGE_TYPES["REGULAR"])
            if 0 <= edge_type < NUM_EDGE_FEATURES: adj_tensor[target_idx, source_idx, edge_type] = 1.0
            else: adj_tensor[target_idx, source_idx, 0] = 1.0 # Default to NO_EDGE if type invalid

        # 3. Create Sequence for GraphRNN (Topological Sort Logic Only)
        sequence = []
        if effective_m is None: # Should be set in __init__
             raise ValueError("self.m_internal (effective_m) is None in TopSort mode.")

        for i in range(1, n): # Iterate for target nodes 1 to n-1
            target_node_idx = i
            # Connections from all preceding nodes (0 to i-1) to target node i
            all_prev_connections = adj_tensor[target_node_idx, 0:i, :] # Shape [i, Feat]

            # Pad preceding connections to effective_m length
            connections_slice = all_prev_connections # Use all predecessors up to i-1
            padding_len = effective_m - connections_slice.shape[0] # Pad up to max_node_count-1
            if padding_len < 0: padding_len = 0 # Safety clamp

            # Pad before reversing to match original logic convention
            padded_connections = np.pad(
                connections_slice,
                ((padding_len, 0), (0, 0)), # Pad at the beginning
                'constant', constant_values=0
            )[::-1, :] # Reverse the order so index 0 is connection to node i-1 etc.

            sequence.append(padded_connections)

        # Convert sequence list to tensor
        if sequence:
            sequence_array = np.stack(sequence, axis=0) # Shape [n-1, effective_m, Feat]
            seq_tensor = torch.tensor(sequence_array, dtype=torch.float32)
        else: # Should only happen if n=1, handled above, but included for safety
            dummy_m = effective_m
            seq_tensor = torch.zeros((0, dummy_m, NUM_EDGE_FEATURES), dtype=torch.float32)

        # 4. Pad the whole sequence tensor to max_node_count - 1 length
        seq_len = seq_tensor.shape[0] # Actual sequence length (should be n-1)
        # Target padded length is self.max_node_count - 1
        target_padded_len = self.max_node_count - 1
        total_pad_len = target_padded_len - seq_len
        if total_pad_len < 0:
             print(f"\nWarning: Negative sequence padding calculated ({total_pad_len}) for graph {idx}. Clamping.")
             total_pad_len = 0

        padded_seq_tensor = torch.nn.functional.pad(
            seq_tensor,
            (0, 0, # Feature dim
             0, 0, # Predecessor dim (already size effective_m)
             0, total_pad_len) # Sequence length dim
        )

        # --- 5. Prepare Levels Tensor ---
        node_to_level = g.graph.get('levels')
        if node_to_level is None: # Fallback if levels weren't calculated/stored
             print(f"CRITICAL Warning: Levels not found for graph {idx} in getitem. Defaulting to 0.")
             node_to_level = {node: 0 for node in g.nodes()}

        levels_ordered = [node_to_level.get(node_id, 0) for node_id in node_ordering]

        # Sequence corresponds to nodes 1 to n-1
        if n > 1:
             levels_for_sequence = levels_ordered[1:]
             levels_array = np.array(levels_for_sequence, dtype=np.int64)
             # Pad to target_padded_len (max_node_count - 1)
             level_pad_len = target_padded_len - len(levels_array)
             if level_pad_len < 0: level_pad_len = 0 # Clamp padding
             # Pad with 0, assuming level 0 is valid padding or unused level for padding
             padded_levels_array = np.pad(levels_array, (0, level_pad_len), 'constant', constant_values=0)
             padded_levels_tensor = torch.tensor(padded_levels_array, dtype=torch.long)
        else: # n=1 case
            padded_levels_tensor = torch.zeros(0, dtype=torch.long)
        # --- END Levels ---

        # 6. Prepare result dictionary
        result = {
            'x': padded_seq_tensor,             # Shape [max_n-1, effective_m, Feat]
            'len': torch.tensor(seq_len, dtype=torch.long), # Scalar: actual seq length (n-1)
            'levels': padded_levels_tensor       # Shape [max_n-1]
        }

        # Conditionally include node types (if requested)
        if self.include_node_types:
            node_types_ordered = [g.nodes[node_id].get('type', -1) for node_id in node_ordering] # Use -1 default?
            if n > 1:
                node_types_array = np.array(node_types_ordered[1:], dtype=np.int64)
                type_pad_len = target_padded_len - len(node_types_array)
                if type_pad_len < 0: type_pad_len = 0
                # Use a distinct padding value if 0 is a valid type, e.g., -1 or NODE_TYPES["ZERO"]
                padded_types_array = np.pad(node_types_array, (0, type_pad_len), 'constant', constant_values=NODE_TYPES["ZERO"])
                padded_node_types = torch.tensor(padded_types_array, dtype=torch.long)
            else:
                 padded_node_types = torch.zeros(0, dtype=torch.long)
            result['node_types'] = padded_node_types # Shape [max_n-1]


        # NOTE: If you use TT conditioning, add 'y': truth_table here

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