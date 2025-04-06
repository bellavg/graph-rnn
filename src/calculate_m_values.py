import numpy as np
import networkx as nx
from tqdm import tqdm
import argparse

# Import your AIGDataset class
from aig_dataset import AIGDataset

def calculate_m_for_aig_dataset(dataset):
    """
    Calculates the maximum required 'm' value for the AIGDataset based on its
    specific BFS ordering and graph structures.

    'm' is the maximum lookback distance required in the node sequence, i.e.,
    the maximum difference between a node's index and the index of its
    earliest predecessor in the BFS ordering.

    Args:
        dataset: An initialized AIGDataset object.

    Returns:
        The maximum 'm' value required for this dataset.
    """
    overall_max_m = 0
    print(f"Calculating 'm' for {len(dataset.graphs)} graphs using dataset's BFS ordering...")

    for g in tqdm(dataset.graphs):
        if g.number_of_nodes() <= 1:
            continue # Skip graphs with 0 or 1 node

        # --- IMPORTANT: Ensure this method exists and is accessible ---
        # Option 1: Rename _get_bfs_ordering to get_bfs_ordering in aig_dataset.py
        # Option 2: Access private method (less ideal): node_ordering = dataset._AIGDataset__get_bfs_ordering(g)
        try:
            # Assuming you rename it to get_bfs_ordering
            node_ordering = dataset.get_bfs_ordering(g)
        except AttributeError:
             print("\nError: Could not find 'get_bfs_ordering'.")
             print("Please rename '_get_bfs_ordering' to 'get_bfs_ordering' in aig_dataset.py")
             return -1 # Indicate error


        if not node_ordering:
            print(f"Warning: Empty node ordering for graph {g}. Skipping.")
            continue

        node_to_idx = {node: idx for idx, node in enumerate(node_ordering)}
        max_m_for_graph = 0

        # Iterate through nodes starting from the second one (index 1)
        for i in range(1, len(node_ordering)):
            current_node = node_ordering[i]
            predecessors = list(g.predecessors(current_node))

            # Find predecessors that appear *before* current_node in the ordering
            valid_pred_indices = []
            for p in predecessors:
                pred_idx = node_to_idx.get(p)
                # Check if predecessor exists in ordering and comes before current node
                if pred_idx is not None and pred_idx < i:
                    valid_pred_indices.append(pred_idx)

            if valid_pred_indices:
                min_pred_idx = min(valid_pred_indices)
                # Calculate the lookback distance needed for this node
                m_for_node = i - min_pred_idx
                max_m_for_graph = max(max_m_for_graph, m_for_node)

        overall_max_m = max(overall_max_m, max_m_for_graph)

    return overall_max_m

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate 'm' value for AIGDataset.")
    parser.add_argument('--graph_file', type=str, default="dataset/inputs8_outputs2.pkl",
                        help='Path to the dataset pickle file (e.g., dataset/inputs8_outputs2.pkl)')
    # Add other necessary arguments for AIGDataset if needed (e.g., max_graphs)
    # parser.add_argument('--max_graphs', type=int, default=None, help='Maximum number of graphs to load')

    args = parser.parse_args()

    print(f"Loading dataset from: {args.graph_file}")
    # Instantiate AIGDataset - set m=None as we are calculating it
    # Set training=True/False as needed, use_bfs must be True here
    dset = AIGDataset(
        graph_file=args.graph_file,
        m=None, # We are calculating m
        training=True, # Or False, shouldn't matter for graph structure
        use_bfs=True, # MUST be True to match the calculation logic
        # max_graphs=args.max_graphs # Optional: load fewer graphs for faster testing
    )

    # Calculate and print the m value
    calculated_m = calculate_m_for_aig_dataset(dset)

    if calculated_m != -1:
        print(f"\n-----------------------------------------")
        print(f"Calculated maximum 'm' value: {calculated_m}")
        print(f"-----------------------------------------")
        print(f"You should set 'm: {calculated_m}' in your config file.")