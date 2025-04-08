
import yaml

import os

import pickle





def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print("Loaded config:")
    print(yaml.dump(config, sort_keys=False))
    # Basic config validation
    if 'data' not in config or 'model' not in config or 'train' not in config:
        raise ValueError("Config file must contain 'data', 'model', and 'train' sections.")
    # Removed use_bfs check here as dataset now handles TopSort default
    # if 'use_bfs' not in config['data']:
    #      raise ValueError("Config must specify 'data.use_bfs' (true or false).")
    # if config['data']['use_bfs'] and 'm' not in config['data']:
    #      raise ValueError("Config must specify 'data.m' when 'data.use_bfs' is true.")
    return config


def get_max_level_from_pkl(graph_file: str) -> int:
    """
    Efficiently loads graphs from a pickle file and finds the maximum level.

    Args:
        graph_file: Path to the pickle file containing NetworkX DiGraph objects

    Returns:
        int: The maximum level found across all graphs
    """
    import os
    import pickle
    import networkx as nx

    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Dataset file not found: {graph_file}")

    max_level = 0
    loaded_count = 0

    with open(graph_file, 'rb') as f:
        raw_graphs = pickle.load(f)

    for i, g in enumerate(raw_graphs):
        if not isinstance(g, nx.Graph):
            continue

        # Skip empty graphs
        if g.number_of_nodes() == 0:
            continue

        # Ensure it's a directed graph
        if not g.is_directed():
            g = g.to_directed()

        # Skip non-DAGs
        if not nx.is_directed_acyclic_graph(g):
            continue

        # Calculate levels for this graph
        try:
            # Find source nodes (in-degree 0)
            source_nodes = [n for n in g.nodes() if g.in_degree(n) == 0]

            # Initialize levels
            levels = {node: 0 for node in source_nodes}

            # Process nodes in topological order
            for node in nx.topological_sort(g):
                if node in levels:  # Already processed (source node)
                    continue

                # Find max level of predecessors
                pred_levels = [levels.get(pred, -1) for pred in g.predecessors(node)]
                max_pred_level = max(pred_levels) if pred_levels else -1

                if max_pred_level >= 0:
                    levels[node] = max_pred_level + 1
                else:
                    levels[node] = 0

            # Find max level in this graph
            graph_max_level = max(levels.values()) if levels else 0
            max_level = max(max_level, graph_max_level)
            loaded_count += 1

        except Exception as e:
            print(f"Error calculating levels for graph {i}: {e}")
            continue

    print(f"Calculated max level {max_level} from {loaded_count} valid graphs")
    return max_level



def get_max_node_count_from_pkl(graph_file: str) -> int:
    """
    Efficiently loads raw graphs from a pickle file and finds the maximum node count.
    """
    # (Implementation remains the same)
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Dataset file not found: {graph_file}")

    max_nodes = 0
    loaded_count = 0
    try:
        with open(graph_file, 'rb') as f:
            raw_graphs = pickle.load(f)

        num_to_check = len(raw_graphs)

        for i, g in enumerate(raw_graphs):
            if i >= num_to_check:
                break
            if hasattr(g, 'number_of_nodes'):
                max_nodes = max(max_nodes, g.number_of_nodes())
                loaded_count += 1
            else:
                print(f"Warning: Item {i} in pickle file doesn't seem to be a graph. Skipping.")

        if loaded_count == 0:
             raise ValueError("No valid graph objects found in the pickle file.")

    except (pickle.UnpicklingError, EOFError) as e:
        raise IOError(f"Error reading pickle file {graph_file}: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while reading {graph_file}: {e}")

    return max_nodes
