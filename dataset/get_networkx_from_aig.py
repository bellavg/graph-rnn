#!/usr/bin/env python3
"""
AIG Graph Dataset Generator

This script processes AIG files from a specified folder in the Downloads directory,
converts them to graph representations, and saves them as a pickle file.
It enforces constraints on input/output sizes and total node count.
"""

import os
import pickle
import networkx as nx
from aigverse import read_aiger_into_aig, to_edge_list, simulate, simulate_nodes
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import Counter
import time

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Path configuration
HOME_DIR = os.path.expanduser("~")
DOWNLOADS_FOLDER = os.path.join(HOME_DIR, "Downloads")
AIG_FOLDER = "rand_aigs_500k 2"  # Folder name inside Downloads
INPUT_PATH = os.path.join(DOWNLOADS_FOLDER, AIG_FOLDER)
OUTPUT_FILE = os.path.join(DOWNLOADS_FOLDER, "all_rand_aigs_data.pkl")

# Filter constraints
MAX_SIZE = 120  # Maximum number of nodes in graph
MAX_INPUTS = 8  # Maximum number of inputs
MAX_OUTPUTS = 8  # Maximum number of outputs

# Truth table configuration
MAX_TT_LENGTH = 2 ** MAX_INPUTS  # Maximum truth table length

# Node and edge type encodings
NODE_TYPE_ENCODING = {
    "0": [0, 0, 0],
    "PI": [1, 0, 0],  # [One-hot encoding]
    "AND": [0, 1, 0],
    "PO": [0, 0, 1]
}

EDGE_LABEL_ENCODING = {
    "INV": [1, 0],  # Inverted edge
    "REG": [0, 1]  # Regular edge
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_all_graphs(all_graphs: List[nx.DiGraph], output_file: str) -> None:
    """
    Save all graphs to a single pickle file.

    Args:
        all_graphs: List of DiGraph objects to save
        output_file: Path to the output file
    """
    with open(output_file, "wb") as f:
        pickle.dump(all_graphs, f)
    print(f"Saved {len(all_graphs)} graphs to {output_file}")


def generate_binary_inputs(num_inputs: int) -> List[List[int]]:
    """
    Generates all possible binary input combinations for a given number of inputs.

    Args:
        num_inputs: Number of input variables

    Returns:
        List of all possible binary input combinations
    """
    return [[(i >> bit) & 1 for bit in range(num_inputs - 1, -1, -1)]
            for i in range(2 ** num_inputs)]


def get_padded_truth_table(tt_binary: str) -> List[int]:
    """
    Convert binary truth table string to list and pad with -1 to max length.

    Args:
        tt_binary: Binary string representation of truth table

    Returns:
        Padded truth table as a list of integers
    """
    # Convert binary string to list of integers
    tt_list = [int(bit) for bit in tt_binary]

    # Pad with -1 to ensure consistent length
    padding = [-1] * (MAX_TT_LENGTH - len(tt_list))
    return tt_list + padding


def get_nodes(aig: Any, G: nx.DiGraph, pad: bool = True) -> nx.DiGraph:
    """
    Add nodes to the graph with appropriate features.

    Args:
        aig: AIG object
        G: NetworkX DiGraph
        pad: Whether to pad truth tables to max length

    Returns:
        Updated DiGraph with nodes added
    """
    input_patterns = list(zip(*generate_binary_inputs(aig.num_pis())))

    def format_tt(binary_str):
        return get_padded_truth_table(binary_str) if pad else [int(b) for b in binary_str]

    zero_tt = format_tt("0" * (2 ** aig.num_pis()))
    G.add_node(0, type=NODE_TYPE_ENCODING["0"], feature=zero_tt)

    for pi in aig.pis():
        binary_inputs = "".join(map(str, list(input_patterns[pi - 1])))
        G.add_node(pi, type=NODE_TYPE_ENCODING["PI"], feature=format_tt(binary_inputs))

    n_to_tt = simulate_nodes(aig)
    for gate in aig.gates():
        binary_truths = n_to_tt[gate].to_binary()
        G.add_node(gate, type=NODE_TYPE_ENCODING["AND"], feature=format_tt(binary_truths))

    return G


def get_edges(aig: Any, G: nx.DiGraph) -> nx.DiGraph:
    """
    Add edges to the graph with one-hot encoded edge labels.

    Args:
        aig: AIG object
        G: NetworkX DiGraph

    Returns:
        Updated DiGraph with edges added
    """
    edges = to_edge_list(aig, inverted_weight=1, regular_weight=0)
    for e in edges:
        # Assign one-hot encoded edge labels
        onehot_label = np.array(
            EDGE_LABEL_ENCODING["INV"] if e.weight == 1 else EDGE_LABEL_ENCODING["REG"],
            dtype=np.float32
        )
        G.add_edge(e.source, e.target, type=onehot_label)
    return G


def get_outs(aig: Any, G: nx.DiGraph, size: int) -> nx.DiGraph:
    """
    Add output nodes and edges to the graph with one-hot encoded output nodes.

    Args:
        aig: AIG object
        G: NetworkX DiGraph
        size: Current size of the graph (number of nodes)

    Returns:
        Updated DiGraph with output nodes and edges added
    """
    tts = simulate(aig)

    for ind, po in enumerate(aig.pos()):
        binary_truths = tts[ind].to_binary()

        # Get out node
        new_out_node_id = size + ind
        G.add_node(new_out_node_id,
                   type=NODE_TYPE_ENCODING["PO"],
                   feature=get_padded_truth_table(binary_truths))

        # Get out edge
        onehot_label = np.array(
            EDGE_LABEL_ENCODING["INV"] if aig.is_complemented(po) else EDGE_LABEL_ENCODING["REG"],
            dtype=np.float32
        )
        pre_node = aig.get_node(po)
        G.add_edge(pre_node, new_out_node_id, type=onehot_label)

    return G


def get_condition(aig: Any, graph_size: int, pad: bool = True) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Get condition lists for the graph.

    Args:
        aig: AIG object
        graph_size: Size of the graph
        pad: Whether to pad truth tables

    Returns:
        Tuple of condition list and full condition list
    """

    def format_tt(binary_str):
        return get_padded_truth_table(binary_str) if pad else [int(b) for b in binary_str]

    zero_tt = format_tt("0" * (2 ** aig.num_pis()))
    condition_list = [zero_tt]
    full_condition_list = [zero_tt]

    input_patterns = list(zip(*generate_binary_inputs(aig.num_pis())))

    for pi in aig.pis():
        binary_inputs = "".join(map(str, list(input_patterns[pi - 1])))
        tt = format_tt(binary_inputs)
        condition_list.append(tt)
        full_condition_list.append(tt)

    condition_list += [[]] * aig.num_gates()
    n_to_tt = simulate_nodes(aig)

    for gate in aig.gates():
        binary_t = n_to_tt[gate].to_binary()
        tt = format_tt(binary_t)
        full_condition_list.append(tt)

    tts = simulate(aig)
    for ind, _ in enumerate(aig.pos()):
        binary_truths = tts[ind].to_binary()
        tt = format_tt(binary_truths)
        condition_list.append(tt)
        full_condition_list.append(tt)

    assert len(condition_list) == graph_size
    return condition_list, full_condition_list


def get_graph(aig: Any, graph_size: int, pad: bool = True) -> nx.DiGraph:
    """
    Create a complete graph representation of the AIG.

    Args:
        aig: AIG object
        graph_size: Size of the graph
        pad: Whether to pad truth tables

    Returns:
        NetworkX DiGraph representing the AIG
    """
    condition, full_condition = get_condition(aig, graph_size, pad=pad)

    G = nx.DiGraph(
        inputs=aig.num_pis(),
        outputs=aig.num_pos(),
        tts=condition,
        full_tts=full_condition,
        output_tts=[[int(tt.get_bit(i)) for i in range(tt.num_bits())] for tt in simulate(aig)]
    )

    G = get_nodes(aig, G, pad=pad)
    G = get_edges(aig, G)
    G = get_outs(aig, G, aig.size())

    pos = aig.num_pos()
    assert G.number_of_nodes() == aig.size() + pos, f"Node count mismatch"

    return G


def collect_graph_stats(graphs: List[nx.DiGraph]) -> Dict[str, Any]:
    """
    Collect statistics about the graph dataset.

    Args:
        graphs: List of NetworkX DiGraph objects

    Returns:
        Dictionary containing various statistics about the graphs
    """
    stats = {
        "num_graphs": len(graphs),
        "input_sizes": [],
        "output_sizes": [],
        "node_counts": [],
        "edge_counts": [],
        "and_gate_counts": [],
        "graph_density": []
    }

    for G in graphs:
        stats["input_sizes"].append(G.graph.get("inputs", 0))
        stats["output_sizes"].append(G.graph.get("outputs", 0))
        stats["node_counts"].append(G.number_of_nodes())
        stats["edge_counts"].append(G.number_of_edges())

        # Count number of AND gates in the graph
        and_gates = sum(1 for _, data in G.nodes(data=True)
                        if "type" in data and data["type"] == NODE_TYPE_ENCODING["AND"])
        stats["and_gate_counts"].append(and_gates)

        # Calculate graph density (ratio of actual edges to possible edges)
        n = G.number_of_nodes()
        possible_edges = n * (n - 1)  # Directed graph
        if possible_edges > 0:
            density = G.number_of_edges() / possible_edges
        else:
            density = 0
        stats["graph_density"].append(density)

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
            return {"min": 0, "max": 0, "avg": 0, "median": 0}
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2]
        }

    # Get statistics for each metric
    input_stats = get_stats(stats["input_sizes"])
    output_stats = get_stats(stats["output_sizes"])
    node_stats = get_stats(stats["node_counts"])
    edge_stats = get_stats(stats["edge_counts"])
    and_gate_stats = get_stats(stats["and_gate_counts"])
    density_stats = get_stats(stats["graph_density"])

    # Print the summary
    print("\n=== GRAPH STATISTICS ===")
    print(f"Total graphs: {stats['num_graphs']}")

    print("\nInput size distribution:")
    print(f"  Min: {input_stats['min']}, Max: {input_stats['max']}")
    print(f"  Avg: {input_stats['avg']:.2f}, Median: {input_stats['median']}")

    print("\nOutput size distribution:")
    print(f"  Min: {output_stats['min']}, Max: {output_stats['max']}")
    print(f"  Avg: {output_stats['avg']:.2f}, Median: {output_stats['median']}")

    print("\nNode count distribution:")
    print(f"  Min: {node_stats['min']}, Max: {node_stats['max']}")
    print(f"  Avg: {node_stats['avg']:.2f}, Median: {node_stats['median']}")

    print("\nEdge count distribution:")
    print(f"  Min: {edge_stats['min']}, Max: {edge_stats['max']}")
    print(f"  Avg: {edge_stats['avg']:.2f}, Median: {edge_stats['median']}")

    print("\nAND gate count distribution:")
    print(f"  Min: {and_gate_stats['min']}, Max: {and_gate_stats['max']}")
    print(f"  Avg: {and_gate_stats['avg']:.2f}, Median: {and_gate_stats['median']}")

    print("\nGraph density distribution:")
    print(f"  Min: {density_stats['min']:.4f}, Max: {density_stats['max']:.4f}")
    print(f"  Avg: {density_stats['avg']:.4f}, Median: {density_stats['median']:.4f}")


def main():
    """Main function to process AIG files from a single folder and create graph dataset."""
    start_time = time.time()
    all_graphs = []
    total_processed = 0
    total_filtered = 0

    # Track rejection reasons
    rejection_reasons = {
        "too_many_inputs": 0,
        "too_many_outputs": 0,
        "too_many_nodes": 0,
        "load_error": 0
    }

    # Check if input path exists
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input path {INPUT_PATH} does not exist.")
        return

    print(f"Processing AIG files from: {INPUT_PATH}")
    print(f"Using constraints: MAX_INPUTS={MAX_INPUTS}, MAX_OUTPUTS={MAX_OUTPUTS}, MAX_SIZE={MAX_SIZE}")

    # Process all .aig files in the folder
    for filename in os.listdir(INPUT_PATH):
        if not filename.endswith('.aig'):
            continue

        # Progress update every 100 files
        if total_processed % 5000 == 0 and total_processed > 0:
            elapsed_time = time.time() - start_time
            files_per_second = total_processed / elapsed_time
            print(f"Progress: Processed {total_processed} files, accepted {total_filtered} graphs "
                  f"({files_per_second:.2f} files/sec)")

        file_path = os.path.join(INPUT_PATH, filename)
        total_processed += 1

        try:
            aig = read_aiger_into_aig(file_path)
        except Exception as e:
            if total_processed % 1000 == 0:  # Limit error output to reduce spam
                print(f"Failed to load {filename}: {e}")
            rejection_reasons["load_error"] += 1
            continue

        # Check input constraint: maximum number of inputs
        if aig.num_pis() > MAX_INPUTS:
            rejection_reasons["too_many_inputs"] += 1
            continue

        # Check output constraint: maximum number of outputs
        if aig.num_pos() > MAX_OUTPUTS:
            rejection_reasons["too_many_outputs"] += 1
            continue

        # Calculate graph size (including constant-0, PIs, gates, and POs)
        graph_size = aig.num_pis() + aig.num_pos() + aig.num_gates() + 1

        # Skip if the graph would be too large based on initial calculation
        if graph_size > MAX_SIZE:
            rejection_reasons["too_many_nodes"] += 1
            continue

        # Create the graph
        Graph = get_graph(aig, graph_size, pad=False)

        # Double-check the actual node count
        if Graph.number_of_nodes() > MAX_SIZE:
            rejection_reasons["too_many_nodes"] += 1
            continue

        # Accept the graph
        all_graphs.append(Graph)
        total_filtered += 1

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nProcessing complete in {total_time:.2f} seconds")
    print(f"Total processed: {total_processed}, Total filtered dataset size: {total_filtered}")

    # Print rejection statistics
    print("\n=== REJECTION STATISTICS ===")
    for reason, count in rejection_reasons.items():
        percentage = (count / total_processed) * 100 if total_processed > 0 else 0
        print(f"{reason}: {count} files ({percentage:.1f}%)")

    # Print overall dataset statistics
    if all_graphs:
        overall_stats = collect_graph_stats(all_graphs)
        print("\n--- Statistics for complete dataset ---")
        print_stats_summary(overall_stats)
        save_all_graphs(all_graphs, OUTPUT_FILE)
    else:
        print("No graphs were accepted. Check your filters and input path.")


if __name__ == "__main__":
    main()