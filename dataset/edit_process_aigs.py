#!/usr/bin/env python3
"""
AIG Graph Dataset Statistics Analyzer

This script loads a pickle file containing graph representations of AIGs and
analyzes the dataset to provide detailed statistics, including node types,
edge types, and structural properties.
"""

import os
import pickle
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import time
import argparse
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Node and edge type encodings (must match what was used to create the dataset)
NODE_TYPE_ENCODING = {
    "0": [0, 0, 0],
    "PI": [1, 0, 0],
    "AND": [0, 1, 0],
    "PO": [0, 0, 1]
}

EDGE_LABEL_ENCODING = {
    "INV": [1, 0],  # Inverted edge
    "REG": [0, 1]  # Regular edge
}

# Reverse mappings for easier identification
NODE_TYPE_MAPPING = {tuple(v): k for k, v in NODE_TYPE_ENCODING.items()}
EDGE_TYPE_MAPPING = {tuple(v): k for k, v in EDGE_LABEL_ENCODING.items()}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_graphs(pickle_file: str) -> List[nx.DiGraph]:
    """
    Load graphs from a pickle file.

    Args:
        pickle_file: Path to the pickle file containing the graphs

    Returns:
        List of NetworkX DiGraph objects
    """
    start_time = time.time()
    print(f"Loading graphs from {pickle_file}...")

    with open(pickle_file, "rb") as f:
        graphs = pickle.load(f)

    load_time = time.time() - start_time
    print(f"Loaded {len(graphs)} graphs in {load_time:.2f} seconds")

    return graphs


def get_node_type(node_attrs: Dict) -> str:
    """
    Determine node type from its attributes.

    Args:
        node_attrs: Node attributes dictionary

    Returns:
        String identifier of the node type
    """
    if "type" not in node_attrs:
        return "UNKNOWN"

    node_type = tuple(node_attrs["type"])
    return NODE_TYPE_MAPPING.get(node_type, "UNKNOWN")


def get_edge_type(edge_attrs: Dict) -> str:
    """
    Determine edge type from its attributes.

    Args:
        edge_attrs: Edge attributes dictionary

    Returns:
        String identifier of the edge type
    """
    if "type" not in edge_attrs:
        return "UNKNOWN"

    # Convert numpy array to tuple for dictionary lookup
    if hasattr(edge_attrs["type"], "tolist"):
        edge_type = tuple(edge_attrs["type"].tolist())
    else:
        edge_type = tuple(edge_attrs["type"])

    return EDGE_TYPE_MAPPING.get(edge_type, "UNKNOWN")


def collect_node_type_stats(graphs: List[nx.DiGraph]) -> Dict[str, List[int]]:
    """
    Collect statistics about node types for each graph.

    Args:
        graphs: List of NetworkX DiGraph objects

    Returns:
        Dictionary containing counts of each node type per graph
    """
    stats = {
        "0_nodes": [],
        "PI_nodes": [],
        "AND_nodes": [],
        "PO_nodes": [],
        "UNKNOWN_nodes": []
    }

    for G in graphs:
        type_counts = {"0": 0, "PI": 0, "AND": 0, "PO": 0, "UNKNOWN": 0}

        for _, attrs in G.nodes(data=True):
            node_type = get_node_type(attrs)
            type_counts[node_type] += 1

        for node_type, count in type_counts.items():
            stats[f"{node_type}_nodes"].append(count)

    return stats


def collect_edge_type_stats(graphs: List[nx.DiGraph]) -> Dict[str, List[int]]:
    """
    Collect statistics about edge types for each graph.

    Args:
        graphs: List of NetworkX DiGraph objects

    Returns:
        Dictionary containing counts of each edge type per graph
    """
    stats = {
        "INV_edges": [],
        "REG_edges": [],
        "UNKNOWN_edges": []
    }

    for G in graphs:
        type_counts = {"INV": 0, "REG": 0, "UNKNOWN": 0}

        for _, _, attrs in G.edges(data=True):
            edge_type = get_edge_type(attrs)
            type_counts[edge_type] += 1

        for edge_type, count in type_counts.items():
            stats[f"{edge_type}_edges"].append(count)

    return stats


def collect_graph_stats(graphs: List[nx.DiGraph]) -> Dict[str, Any]:
    """
    Collect comprehensive statistics about the graph dataset.

    Args:
        graphs: List of NetworkX DiGraph objects

    Returns:
        Dictionary containing various statistics about the graphs
    """
    # Basic graph statistics
    stats = {
        "num_graphs": len(graphs),
        "input_sizes": [],
        "output_sizes": [],
        "node_counts": [],
        "edge_counts": [],
        "graph_depths": [],
        "graph_density": [],
        "avg_degree": [],
        "edge_to_node_ratio": []
    }

    # Collect node type statistics
    node_type_stats = collect_node_type_stats(graphs)
    stats.update(node_type_stats)

    # Collect edge type statistics
    edge_type_stats = collect_edge_type_stats(graphs)
    stats.update(edge_type_stats)

    for G in graphs:
        # Basic properties
        stats["input_sizes"].append(G.graph.get("inputs", 0))
        stats["output_sizes"].append(G.graph.get("outputs", 0))
        stats["node_counts"].append(G.number_of_nodes())
        stats["edge_counts"].append(G.number_of_edges())

        # Graph density (ratio of actual edges to possible edges)
        n = G.number_of_nodes()
        possible_edges = n * (n - 1)  # Directed graph
        if possible_edges > 0:
            density = G.number_of_edges() / possible_edges
        else:
            density = 0
        stats["graph_density"].append(density)

        # Average degree
        if n > 0:
            avg_degree = 2 * G.number_of_edges() / n  # For directed graph
        else:
            avg_degree = 0
        stats["avg_degree"].append(avg_degree)

        # Edge to node ratio
        if n > 0:
            edge_node_ratio = G.number_of_edges() / n
        else:
            edge_node_ratio = 0
        stats["edge_to_node_ratio"].append(edge_node_ratio)

        # Try to estimate graph depth (longest path)
        try:
            # Find source nodes (those with no predecessors)
            sources = [n for n, d in G.in_degree() if d == 0]

            # Find sink nodes (those with no successors)
            sinks = [n for n, d in G.out_degree() if d == 0]

            # If we have both sources and sinks, calculate longest path
            if sources and sinks:
                # For each source-sink pair, find the longest path
                max_path_length = 0
                for source in sources:
                    for sink in sinks:
                        try:
                            # Try to find all simple paths
                            paths = list(nx.all_simple_paths(G, source, sink))
                            if paths:
                                max_path_length = max(max_path_length, max(len(p) for p in paths))
                        except nx.NetworkXNoPath:
                            continue

                # If we found any paths, record the depth
                if max_path_length > 0:
                    stats["graph_depths"].append(max_path_length - 1)  # Depth = edges, not nodes
                else:
                    stats["graph_depths"].append(0)
            else:
                stats["graph_depths"].append(0)
        except Exception:
            # If there's any issue calculating depth, just use 0
            stats["graph_depths"].append(0)

    return stats


def print_stats_summary(stats: Dict[str, Any]) -> None:
    """
    Print a summary of graph statistics.

    Args:
        stats: Dictionary of graph statistics
    """

    # Helper function to get statistics from a list
    def get_stats(values):
        if not values:
            return {"min": 0, "max": 0, "avg": 0, "median": 0, "std": 0}
        return {
            "min": min(values),
            "max": max(values),
            "avg": np.mean(values),
            "median": np.median(values),
            "std": np.std(values)
        }

    # Print basic statistics
    print("\n===== GRAPH DATASET STATISTICS =====")
    print(f"Total graphs: {stats['num_graphs']}")

    # Print node type statistics
    print("\n----- NODE TYPE STATISTICS -----")
    for node_type in ["0", "PI", "AND", "PO", "UNKNOWN"]:
        key = f"{node_type}_nodes"
        if key in stats:
            node_stats = get_stats(stats[key])
            print(f"\n{node_type} Nodes:")
            print(f"  Count: {sum(stats[key])}")
            print(f"  Min: {node_stats['min']}, Max: {node_stats['max']}")
            print(f"  Avg: {node_stats['avg']:.2f}, Median: {node_stats['median']:.1f}")
            print(f"  Std Dev: {node_stats['std']:.2f}")

    # Print edge type statistics
    print("\n----- EDGE TYPE STATISTICS -----")
    for edge_type in ["INV", "REG", "UNKNOWN"]:
        key = f"{edge_type}_edges"
        if key in stats:
            edge_stats = get_stats(stats[key])
            print(f"\n{edge_type} Edges:")
            print(f"  Count: {sum(stats[key])}")
            print(f"  Min: {edge_stats['min']}, Max: {edge_stats['max']}")
            print(f"  Avg: {edge_stats['avg']:.2f}, Median: {edge_stats['median']:.1f}")
            print(f"  Std Dev: {edge_stats['std']:.2f}")

    # Print structural statistics
    print("\n----- STRUCTURAL STATISTICS -----")

    input_stats = get_stats(stats["input_sizes"])
    print("\nInput size distribution:")
    print(f"  Min: {input_stats['min']}, Max: {input_stats['max']}")
    print(f"  Avg: {input_stats['avg']:.2f}, Median: {input_stats['median']:.1f}")
    print(f"  Std Dev: {input_stats['std']:.2f}")

    output_stats = get_stats(stats["output_sizes"])
    print("\nOutput size distribution:")
    print(f"  Min: {output_stats['min']}, Max: {output_stats['max']}")
    print(f"  Avg: {output_stats['avg']:.2f}, Median: {output_stats['median']:.1f}")
    print(f"  Std Dev: {output_stats['std']:.2f}")

    node_stats = get_stats(stats["node_counts"])
    print("\nNode count distribution:")
    print(f"  Min: {node_stats['min']}, Max: {node_stats['max']}")
    print(f"  Avg: {node_stats['avg']:.2f}, Median: {node_stats['median']:.1f}")
    print(f"  Std Dev: {node_stats['std']:.2f}")

    edge_stats = get_stats(stats["edge_counts"])
    print("\nEdge count distribution:")
    print(f"  Min: {edge_stats['min']}, Max: {edge_stats['max']}")
    print(f"  Avg: {edge_stats['avg']:.2f}, Median: {edge_stats['median']:.1f}")
    print(f"  Std Dev: {edge_stats['std']:.2f}")

    if stats["graph_depths"]:
        depth_stats = get_stats(stats["graph_depths"])
        print("\nGraph depth distribution:")
        print(f"  Min: {depth_stats['min']}, Max: {depth_stats['max']}")
        print(f"  Avg: {depth_stats['avg']:.2f}, Median: {depth_stats['median']:.1f}")
        print(f"  Std Dev: {depth_stats['std']:.2f}")

    density_stats = get_stats(stats["graph_density"])
    print("\nGraph density distribution:")
    print(f"  Min: {density_stats['min']:.4f}, Max: {density_stats['max']:.4f}")
    print(f"  Avg: {density_stats['avg']:.4f}, Median: {density_stats['median']:.4f}")
    print(f"  Std Dev: {density_stats['std']:.4f}")

    degree_stats = get_stats(stats["avg_degree"])
    print("\nAverage degree distribution:")
    print(f"  Min: {degree_stats['min']:.2f}, Max: {degree_stats['max']:.2f}")
    print(f"  Avg: {degree_stats['avg']:.2f}, Median: {degree_stats['median']:.2f}")
    print(f"  Std Dev: {degree_stats['std']:.2f}")

    ratio_stats = get_stats(stats["edge_to_node_ratio"])
    print("\nEdge-to-Node ratio distribution:")
    print(f"  Min: {ratio_stats['min']:.2f}, Max: {ratio_stats['max']:.2f}")
    print(f"  Avg: {ratio_stats['avg']:.2f}, Median: {ratio_stats['median']:.2f}")
    print(f"  Std Dev: {ratio_stats['std']:.2f}")


def visualize_statistics(stats: Dict[str, Any], output_dir: str = None) -> None:
    """
    Generate visualizations of key statistics.

    Args:
        stats: Dictionary of graph statistics
        output_dir: Directory to save visualization files (if None, just display)
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 8))

    # 1. Node Type Distribution
    node_types = ["0", "PI", "AND", "PO"]
    avg_counts = [np.mean(stats[f"{t}_nodes"]) for t in node_types]

    plt.subplot(2, 2, 1)
    plt.bar(node_types, avg_counts)
    plt.title("Average Node Type Distribution")
    plt.ylabel("Average Count per Graph")

    # 2. Edge Type Distribution
    edge_types = ["REG", "INV"]
    avg_edge_counts = [np.mean(stats[f"{t}_edges"]) for t in edge_types]

    plt.subplot(2, 2, 2)
    plt.bar(edge_types, avg_edge_counts)
    plt.title("Average Edge Type Distribution")
    plt.ylabel("Average Count per Graph")

    # 3. Node Count Histogram
    plt.subplot(2, 2, 3)
    plt.hist(stats["node_counts"], bins=30, alpha=0.7)
    plt.title("Node Count Distribution")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Frequency")

    # 4. Input/Output Size Distribution
    plt.subplot(2, 2, 4)
    plt.hist(stats["input_sizes"], bins=np.arange(max(stats["input_sizes"]) + 2) - 0.5,
             alpha=0.6, label="Inputs")
    plt.hist(stats["output_sizes"], bins=np.arange(max(stats["output_sizes"]) + 2) - 0.5,
             alpha=0.6, label="Outputs")
    plt.title("Input/Output Size Distribution")
    plt.xlabel("Number of Inputs/Outputs")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "graph_stats_overview.png"), dpi=300)
    else:
        plt.show()

    # Additional visualizations

    # 5. Edge-to-Node Ratio
    plt.figure(figsize=(10, 6))
    plt.hist(stats["edge_to_node_ratio"], bins=30, alpha=0.7)
    plt.title("Edge-to-Node Ratio Distribution")
    plt.xlabel("Edge-to-Node Ratio")
    plt.ylabel("Frequency")

    if output_dir:
        plt.savefig(os.path.join(output_dir, "edge_node_ratio.png"), dpi=300)
    else:
        plt.show()

    # 6. Graph Depth Distribution
    if stats["graph_depths"]:
        plt.figure(figsize=(10, 6))
        plt.hist(stats["graph_depths"], bins=np.arange(max(stats["graph_depths"]) + 2) - 0.5, alpha=0.7)
        plt.title("Graph Depth Distribution")
        plt.xlabel("Graph Depth (Longest Path)")
        plt.ylabel("Frequency")

        if output_dir:
            plt.savefig(os.path.join(output_dir, "graph_depth.png"), dpi=300)
        else:
            plt.show()


def main():
    """Main function to analyze an existing AIG graph dataset."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze statistics of AIG graph dataset")
    parser.add_argument("--pickle_file", type=str, default="final_data.pkl", help="Path to the pickle file containing graphs")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--output", type=str, default=None, help="Directory to save visualizations")
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.pickle_file):
        print(f"Error: Input file {args.pickle_file} does not exist.")
        return

    # Load the graphs
    graphs = load_graphs(args.pickle_file)

    if not graphs:
        print("No graphs found in the pickle file.")
        return

    print(f"Starting analysis of {len(graphs)} graphs...")
    start_time = time.time()

    # Collect comprehensive statistics
    stats = collect_graph_stats(graphs)

    # Print the statistics summary
    print_stats_summary(stats)

    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_statistics(stats, args.output)

    end_time = time.time()
    print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()