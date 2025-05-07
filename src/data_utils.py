import yaml
import os
import pickle
import networkx as nx
import numpy as np # Added for np.max
from typing import List # Added for type hinting

def load_config(config_file):
    """Loads configuration from a YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print("Loaded config:")
    print(yaml.dump(config, sort_keys=False))
    # Basic config validation
    if 'data' not in config or 'model' not in config or 'train' not in config:
        raise ValueError("Config file must contain 'data', 'model', and 'train' sections.")
    # Check for the new graph_files key
    if 'graph_files' not in config['data'] or not isinstance(config['data']['graph_files'], list):
         print("Warning: Config 'data.graph_files' is missing or not a list. Ensure it's defined in your YAML.")
         # Optionally raise ValueError if it's strictly required
         # raise ValueError("Config 'data.graph_files' must be a list of file paths.")
    return config


def _calculate_levels_for_single_graph(g: nx.DiGraph) -> int:
    """
    Calculates the maximum level for a single NetworkX DiGraph.
    Helper function for get_max_level_from_pkl.
    Returns -1 if graph is invalid or not a DAG.
    """
    if not isinstance(g, nx.Graph) or g.number_of_nodes() == 0:
        return -1 # Skip non-graphs or empty graphs

    if not g.is_directed():
        g = g.to_directed()

    if not nx.is_directed_acyclic_graph(g):
        return -1 # Skip non-DAGs

    levels = {}
    max_level = 0
    source_nodes = [n for n in g.nodes() if g.in_degree(n) == 0]

    for node in source_nodes:
        levels[node] = 0

    try:
        for node in nx.topological_sort(g):
            if node in levels: continue
            pred_levels = [levels.get(pred, -1) for pred in g.predecessors(node)]
            max_pred_level = max(pred_levels) if pred_levels else -1
            if max_pred_level >= 0:
                levels[node] = max_pred_level + 1
                max_level = max(max_level, levels[node])
            else:
                levels[node] = 0 # Unreachable node gets level 0
    except Exception: # Catch potential errors during sort/level calculation
        return -1

    return max_level


def get_max_level_from_pkl(graph_files: List[str]) -> int:
    """
    Efficiently loads graphs from a LIST of pickle files and finds the maximum level
    across all valid DAGs in all files.

    Args:
        graph_files: List of paths to the pickle files containing NetworkX DiGraph objects

    Returns:
        int: The maximum level found across all graphs, or 0 if none are valid.
    """
    overall_max_level = 0
    total_loaded_count = 0
    total_valid_dag_count = 0

    print(f"Calculating max level from {len(graph_files)} file(s)...")
    for file_path in graph_files:
        if not os.path.exists(file_path):
            print(f"Warning: Dataset file not found: {file_path}. Skipping.")
            continue

        print(f" Processing levels in: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                raw_graphs = pickle.load(f)
            if not isinstance(raw_graphs, list):
                 print(f"Warning: Expected a list in {file_path}, found {type(raw_graphs)}. Skipping.")
                 continue

            file_loaded_count = 0
            file_valid_dag_count = 0
            file_max_level = 0

            for i, g in enumerate(raw_graphs):
                graph_max_level = _calculate_levels_for_single_graph(g)
                if graph_max_level != -1: # Check if graph was valid and processed
                    file_max_level = max(file_max_level, graph_max_level)
                    file_valid_dag_count += 1
                file_loaded_count += 1 # Count all attempts within the file

            print(f"  -> Found max level {file_max_level} in this file ({file_valid_dag_count}/{file_loaded_count} valid DAGs processed).")
            overall_max_level = max(overall_max_level, file_max_level)
            total_loaded_count += file_loaded_count
            total_valid_dag_count += file_valid_dag_count

        except (pickle.UnpicklingError, EOFError, MemoryError) as e:
            print(f"Error reading pickle file {file_path}: {e}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing levels in {file_path}: {e}")

    print(f"\nOverall max level calculated: {overall_max_level} from {total_valid_dag_count}/{total_loaded_count} valid DAGs across all files.")
    if total_valid_dag_count == 0:
        print("Warning: No valid DAGs found to calculate max level. Returning 0.")
    return overall_max_level


def get_max_node_count_from_pkl(graph_files: List[str]) -> int:
    """
    Efficiently loads raw graphs from a LIST of pickle files and finds the maximum node count
    across all graphs in all files.

    Args:
        graph_files: List of paths to the pickle files.

    Returns:
        int: The maximum node count found, or 0 if no valid graphs are found.
    """
    overall_max_nodes = 0
    total_loaded_count = 0
    total_valid_graph_count = 0

    print(f"Calculating max node count from {len(graph_files)} file(s)...")
    for file_path in graph_files:
        if not os.path.exists(file_path):
            print(f"Warning: Dataset file not found: {file_path}. Skipping.")
            continue

        print(f" Processing node counts in: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                raw_graphs = pickle.load(f)
            if not isinstance(raw_graphs, list):
                 print(f"Warning: Expected a list in {file_path}, found {type(raw_graphs)}. Skipping.")
                 continue

            file_loaded_count = 0
            file_valid_graph_count = 0
            file_max_nodes = 0

            for i, g in enumerate(raw_graphs):
                if hasattr(g, 'number_of_nodes'):
                    file_max_nodes = max(file_max_nodes, g.number_of_nodes())
                    file_valid_graph_count += 1
                else:
                    print(f"Warning: Item {i} in {file_path} doesn't seem to be a graph. Skipping.")
                file_loaded_count += 1

            print(f"  -> Found max nodes {file_max_nodes} in this file ({file_valid_graph_count}/{file_loaded_count} graphs processed).")
            overall_max_nodes = max(overall_max_nodes, file_max_nodes)
            total_loaded_count += file_loaded_count
            total_valid_graph_count += file_valid_graph_count

        except (pickle.UnpicklingError, EOFError, MemoryError) as e:
            print(f"Error reading pickle file {file_path}: {e}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while reading {file_path}: {e}")

    print(f"\nOverall max node count calculated: {overall_max_nodes} from {total_valid_graph_count}/{total_loaded_count} graphs across all files.")
    if total_valid_graph_count == 0:
        print("Warning: No valid graphs found to calculate max node count. Returning 0.")

    return overall_max_nodes
