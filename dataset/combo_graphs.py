#!/usr/bin/env python3
"""
Script to create a balanced graph dataset (final_data.pkl) by combining
graphs from data.pkl and aigs.pkl, aiming for an even distribution
across node size categories up to a target total count.
Saves unused graphs to remainder_graphs.pkl.
"""

import os
import pickle
import argparse
import numpy as np
import networkx as nx
import random # Needed for sampling
from collections import defaultdict # Useful for grouping
from typing import List, Dict, Any, Tuple, Set, Optional

# Set a random seed for reproducible sampling
random.seed(42)
np.random.seed(42)

# Define the node count bins used for balancing and reporting
# Adjusted slightly to match the example output ranges provided
NODE_BINS = [
    # (1, 10), # Excluding based on example having 0 here, adjust if needed
    (11, 20), (21, 30), (31, 40), (41, 50),
    (51, 60), (61, 100),
    (101, float('inf'))
]


def get_neighbours(bin_label: str, sorted_active_bins: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Find the left and right neighbours of a bin label in the sorted list."""
    try:
        index = sorted_active_bins.index(bin_label)
        left = sorted_active_bins[index - 1] if index > 0 else None
        right = sorted_active_bins[index + 1] if index < len(sorted_active_bins) - 1 else None
        return left, right
    except ValueError:
        return None, None # Should not happen if bin_label is from active_bins


def get_node_bin(node_count: int) -> str:
    """Assigns a node count to a predefined bin label."""
    for low, high in NODE_BINS:
        if low <= node_count <= high:
            if high == float('inf'):
                return f"{low}+"
            elif low == high:
                 return f"{low}"
            else:
                return f"{low}-{high}"
    # If node count is below the first bin's lower bound
    if NODE_BINS and node_count < NODE_BINS[0][0]:
         return f"<{NODE_BINS[0][0]}"
    return "Other" # Fallback

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create a graph dataset balanced across node size categories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Changed names slightly to reflect their roles more generally
    parser.add_argument('--input_file1', default='data.pkl',
                        help='First input pickle file containing graphs.')
    parser.add_argument('--input_file2', default='aigs.pkl',
                        help='Second input pickle file containing graphs.')
    parser.add_argument('--final_file', default='final_data.pkl',
                        help='Output file for the final balanced dataset.')
    parser.add_argument('--remainder_file', default='remainder_graphs.pkl',
                        help='Output file to save unused graphs.')
    parser.add_argument('--target_total_graphs', type=int, default=40000, # Updated default
                        help='Target total number of graphs in the final dataset.')
    parser.add_argument('--force', action='store_true',
                        help='Force combining graphs even if compatibility issues are detected.')
    return parser.parse_args()

# --- Keep load_graphs, print_node_distribution, check_graph_compatibility ---
# (Identical to the previous script version, including NODE_BINS in print_node_distribution)

def load_graphs(file_path: str) -> List[nx.DiGraph]:
    """
    Load graphs from a pickle file. Ensures they are nx.DiGraph.
    (Copied from previous version - no changes needed here)
    """
    print(f"Loading graphs from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
    except Exception as e:
        raise IOError(f"Error reading pickle file {file_path}: {e}")

    if not isinstance(loaded_data, list):
        raise TypeError(f"Expected a list of graphs in {file_path}, found {type(loaded_data)}")

    graphs = []
    invalid_count = 0
    for i, item in enumerate(loaded_data):
        if isinstance(item, nx.DiGraph):
            try:
                 _ = item.number_of_nodes()
                 _ = item.number_of_edges()
                 graphs.append(item)
            except Exception as e:
                print(f"Warning: Skipping potentially corrupted graph at index {i} in {file_path}: {e}")
                invalid_count += 1
        else:
            invalid_count += 1

    if not graphs:
        raise ValueError(f"No valid nx.DiGraph objects found in {file_path}")

    print(f"Loaded {len(graphs)} valid nx.DiGraph graphs from {file_path}.")
    if invalid_count > 0:
        print(f"Skipped {invalid_count} invalid or non-DiGraph objects.")
    return graphs


def print_node_distribution(graphs: List[nx.DiGraph], label: str = "") -> float:
    """
    Print node count distribution summary using defined NODE_BINS and return the mean.
    (Copied from previous version - uses global NODE_BINS)
    """
    num_graphs = len(graphs)
    if num_graphs == 0:
        print(f"\n--- {label} Node Distribution (0 graphs) ---")
        return 0.0

    try:
        node_counts = [g.number_of_nodes() for g in graphs]
    except Exception as e:
        print(f"Error calculating node counts for {label}: {e}. Check graph integrity.")
        valid_graphs = []
        node_counts = []
        for g in graphs:
            try:
                node_counts.append(g.number_of_nodes())
                valid_graphs.append(g)
            except Exception:
                pass
        graphs = valid_graphs
        num_graphs = len(graphs)
        if num_graphs == 0:
             print(f"{label} distribution: 0 graphs after filtering.")
             return 0.0
        print("Warning: Some graphs were excluded from distribution calculation due to errors.")

    mean_nodes = np.mean(node_counts) if node_counts else 0
    min_nodes = min(node_counts) if node_counts else 0
    max_nodes = max(node_counts) if node_counts else 0
    median_nodes = np.median(node_counts) if node_counts else 0
    std_dev_nodes = np.std(node_counts) if node_counts else 0

    print(f"\n--- {label} Node Distribution ({num_graphs} graphs) ---")
    print(f"  Mean: {mean_nodes:.2f}, Median: {median_nodes:.2f}, Std Dev: {std_dev_nodes:.2f}")
    print(f"  Min: {min_nodes}, Max: {max_nodes}")

    print("  Distribution by node count ranges:")
    bin_counts = defaultdict(int)
    for nc in node_counts:
        bin_label = get_node_bin(nc)
        bin_counts[bin_label] += 1

    # Define the order for printing based on NODE_BINS definition
    print_order = [get_node_bin(low) for low, high in NODE_BINS]
    if node_counts and min(node_counts) < NODE_BINS[0][0]:
        print_order.insert(0, f"<{NODE_BINS[0][0]}")
    if "Other" in bin_counts and "Other" not in print_order:
        print_order.append("Other") # Add Other if present and not already covered

    for bin_label in print_order:
         count = bin_counts.get(bin_label, 0) # Use .get for bins that might be empty
         percentage = (count / num_graphs) * 100 if num_graphs > 0 else 0
         if count > 0: # Only print bins that have graphs
              print(f"    {bin_label:>8} nodes: {count:>6} graphs ({percentage:>5.1f}%)")

    return mean_nodes


def check_graph_compatibility(graph_list1: List[nx.DiGraph], graph_list2: List[nx.DiGraph]) -> Dict[str, Tuple[Any, Any]]:
    """
    Check compatibility between two lists of graphs by comparing samples.
    (Copied from previous version - no changes needed here)
    """
    if not graph_list1 or not graph_list2:
        print("Warning: Cannot check compatibility, one or both graph lists are empty.")
        return {}

    g1 = graph_list1[0]
    g2 = graph_list2[0]

    print(f"\nComparing sample graph from list 1 (Nodes: {g1.number_of_nodes()}, File: input_file1) "
          f"with sample graph from list 2 (Nodes: {g2.number_of_nodes()}, File: input_file2)...")

    differences = {}

    if g1.is_directed() != g2.is_directed():
        differences["directed"] = (g1.is_directed(), g2.is_directed())

    def get_attribute_keys(graph, element_type='node'):
        elements = []
        if element_type == 'node': elements = list(graph.nodes(data=True))
        elif element_type == 'edge': elements = list(graph.edges(data=True))
        if not elements: return set()
        for element in elements:
            data_dict = {}
            if element_type == 'node' and len(element) > 1: data_dict = element[1]
            elif element_type == 'edge' and len(element) > 2: data_dict = element[2]
            if isinstance(data_dict, dict): return set(data_dict.keys())
        return set()

    g1_node_attrs = get_attribute_keys(g1, 'node')
    g2_node_attrs = get_attribute_keys(g2, 'node')
    if g1_node_attrs != g2_node_attrs:
        differences["node_attribute_keys"] = (g1_node_attrs, g2_node_attrs)

    g1_edge_attrs = get_attribute_keys(g1, 'edge')
    g2_edge_attrs = get_attribute_keys(g2, 'edge')
    if g1_edge_attrs != g2_edge_attrs:
        differences["edge_attribute_keys"] = (g1_edge_attrs, g2_edge_attrs)

    def get_unique_attribute_values(graph, attribute_key, element_type='node'):
        values = set()
        elements = []
        if element_type == 'node': elements = graph.nodes(data=True)
        elif element_type == 'edge': elements = graph.edges(data=True)
        for element_data in elements:
            data_dict = {}
            if element_type == 'node' and len(element_data) > 1: data_dict = element_data[1]
            elif element_type == 'edge' and len(element_data) > 2: data_dict = element_data[2]
            if isinstance(data_dict, dict) and attribute_key in data_dict:
                value = data_dict[attribute_key]
                if isinstance(value, list): values.add(tuple(value))
                elif isinstance(value, np.ndarray): values.add(tuple(value.tolist()))
                else:
                    try:
                        hash(value); values.add(value)
                    except TypeError: values.add(str(value))
        return values

    if 'type' in g1_node_attrs and 'type' in g2_node_attrs:
        g1_node_types = get_unique_attribute_values(g1, 'type', 'node')
        g2_node_types = get_unique_attribute_values(g2, 'type', 'node')
        if (g1_node_types or g2_node_types) and g1_node_types != g2_node_types:
             differences["node_type_values"] = (g1_node_types, g2_node_types)

    if 'type' in g1_edge_attrs and 'type' in g2_edge_attrs:
        g1_edge_types = get_unique_attribute_values(g1, 'type', 'edge')
        g2_edge_types = get_unique_attribute_values(g2, 'type', 'edge')
        if (g1_edge_types or g2_edge_types) and g1_edge_types != g2_edge_types:
            differences["edge_type_values"] = (g1_edge_types, g2_edge_types)

    return differences


# ============================================================
# ============================================================
# === Replace your existing function with this entire block ===
# ============================================================
def balance_graph_distribution_evenly( # Keep the function name the same
    input_file1: str,
    input_file2: str,
    final_file: str,
    remainder_file: str,
    target_total_graphs: int,
    force: bool
) -> None:
    """
    Creates a graph dataset aiming for specific initial target counts per bin,
    redistributing shortfalls downwards ("waterfall" style) based on availability.
    """
    # --- Steps 1-4: Load, Check Compatibility, Initial Analysis, Combine/Categorize ---
    # (Keep these steps exactly as in the previous version)
    try:
        graphs1 = load_graphs(input_file1)
        graphs2 = load_graphs(input_file2)
    except (FileNotFoundError, ValueError, IOError, TypeError) as e: print(f"Error loading initial graphs: {e}"); return
    print("\n--- Checking Graph Compatibility ---")
    differences = check_graph_compatibility(graphs1, graphs2)
    if differences: # Handle differences (same logic as before)
        print("\n=== COMPATIBILITY DIFFERENCES DETECTED ===")
        critical_diffs = False
        for key, (val1, val2) in differences.items():
            print(f"\n* {key}:\n  Input 1 Sample: {val1}\n  Input 2 Sample: {val2}")
            if key in ["directed", "node_type_values", "edge_type_values"]: critical_diffs = True; print("  (Considered potentially critical)")
        if critical_diffs and not force:
            if input("\nCritical compatibility differences found. Continue anyway? (y/n): ").lower() != 'y': print("Operation canceled."); return
            else: print("Proceeding despite critical differences (forced by user).")
        elif not critical_diffs: print("\nNon-critical differences detected. Proceeding.")
        elif force: print("\nProceeding despite differences (--force option used).")
    else: print("No compatibility differences detected between sample graphs.")
    print("\n--- Initial Dataset Statistics ---")
    _ = print_node_distribution(graphs1, f"Input 1 ({input_file1})"); _ = print_node_distribution(graphs2, f"Input 2 ({input_file2})")
    print("\n--- Combining and Categorizing All Graphs ---")
    all_graphs_map = {}; graphs_by_bin = defaultdict(list); available_counts = defaultdict(int)
    for g in graphs1: all_graphs_map[id(g)] = g
    for g in graphs2: all_graphs_map[id(g)] = g
    all_unique_graphs = list(all_graphs_map.values()); total_unique_available = len(all_unique_graphs)
    print(f"Total unique graphs available from both sources: {total_unique_available}")
    if total_unique_available < target_total_graphs: print(f"Warning: Available graphs ({total_unique_available}) < target ({target_total_graphs}). Adjusting target."); target_total_graphs = total_unique_available
    for g in all_unique_graphs:
        try: bin_label = get_node_bin(g.number_of_nodes()); graphs_by_bin[bin_label].append(g); available_counts[bin_label] += 1
        except Exception as e: print(f"Warning: Could not process graph {id(g)} during binning: {e}. Skipping.")
    # Ensure active_bins are sorted numerically for correct iteration order
    active_bins = sorted([label for label, graphs in graphs_by_bin.items() if graphs], key=lambda x: int(x.split('-')[0].replace('+', '').replace('<','')))
    if not active_bins: print("Error: No graphs remaining after binning."); return
    print(f"Graphs distributed across {len(active_bins)} active node bins: {active_bins}")
    print("Available counts per bin:")
    for bin_label in active_bins: print(f"  {bin_label}: {available_counts[bin_label]}")
    # --- End of Steps 1-4 ---

    # --- Step 5: Calculate Final Targets using Waterfall Redistribution ---
    print("\n--- Calculating Final Targets with Waterfall Redistribution (High to Low) ---")

    # Define INITIAL user targets (Adjusted to sum to 40k)
    initial_targets = {
        "11-20": 5000,
        "21-30": 7000,
        "31-40": 7800, # Adjusted from 8k
        "41-50": 7800, # Adjusted from 8k
        "51-60": 6400, # Adjusted from 6.5k
        "61-100": 6000,
        # "101+": 0 # Add if needed
    }
    print("Initial targets (adjusted to sum to 40k):")
    for bin_label in active_bins: print(f"  {bin_label}: {initial_targets.get(bin_label, 0)}")

    final_targets = defaultdict(int)
    carry_down = 0 # Shortfall carried from higher bin to lower bin

    print("\n--- Waterfall Calculation ---")
    # Iterate from HIGHEST bin to LOWEST bin
    for bin_label in reversed(active_bins):
        initial_target = initial_targets.get(bin_label, 0)
        effective_target = initial_target + carry_down
        available = available_counts.get(bin_label, 0)

        take = min(effective_target, available)
        final_targets[bin_label] = take

        shortfall = effective_target - take # This is passed down
        carry_down = shortfall # Update carry_down for the next (lower) bin

        print(f"  Bin {bin_label}: InitialTgt={initial_target}, CarryIn={effective_target-initial_target:.0f}, "
              f"EffectiveTgt={effective_target:.0f}, Available={available}, Took={take}, "
              f"CarryOut(Shortfall)={carry_down:.0f}")

    if carry_down > 0:
         print(f"Warning: After processing lowest bin, {carry_down:.0f} graphs could not be allocated (ran out of bins).")

    # --- Adjust final total if slightly off 40k ---
    current_total = sum(final_targets.values())
    print(f"\nTotal calculated after waterfall: {current_total}")
    difference = target_total_graphs - current_total

    if difference != 0:
        print(f"Difference from target {target_total_graphs}: {difference}. Adjusting...")
        # Try to add/remove difference from a bin with capacity, e.g., '21-30' or '11-20'
        adjust_bin = None
        if '21-30' in final_targets and available_counts['21-30'] > final_targets['21-30'] and difference > 0:
            adjust_bin = '21-30'
        elif '11-20' in final_targets and available_counts['11-20'] > final_targets['11-20'] and difference > 0:
            adjust_bin = '11-20'
        elif difference < 0: # If we need to remove graphs, try removing from largest bin first?
             if '21-30' in final_targets and final_targets['21-30'] >= abs(difference):
                 adjust_bin = '21-30'
             elif '11-20' in final_targets and final_targets['11-20'] >= abs(difference):
                  adjust_bin = '11-20'
             # Add more fallback bins if needed

        if adjust_bin:
            # Ensure adjustment doesn't exceed availability if adding
            can_add = available_counts[adjust_bin] - final_targets[adjust_bin] if difference > 0 else float('inf')
            actual_adjustment = min(difference, can_add) if difference > 0 else max(difference, -final_targets[adjust_bin])

            final_targets[adjust_bin] += actual_adjustment
            print(f"  Adjusted bin {adjust_bin} by {actual_adjustment}. New target: {final_targets[adjust_bin]}")
            current_total = sum(final_targets.values()) # Recalculate total
            if current_total != target_total_graphs:
                 print(f"Warning: Adjustment failed to reach target. Final total: {current_total}")
        else:
            print(f"  Could not find suitable bin to adjust for difference {difference}.")

    print("\nFinal calculated target counts per bin:")
    final_calculated_total = 0
    for bin_label in active_bins:
        final_count = final_targets.get(bin_label, 0)
        print(f"  {bin_label}: {final_count}")
        final_calculated_total += final_count
    print(f"Final calculated total graphs: {final_calculated_total}")


    # --- Step 6: Select Graphs Based on Calculated Final Targets ---
    print("\n--- Selecting Graphs based on Final Calculated Targets ---")
    selected_graphs_map = defaultdict(list)
    graphs_taken_count = 0
    # Remainder tracking needs to be based on the final selection vs original list
    # We don't need available_for_remainder dict during selection anymore

    # Use a fresh copy for selection sampling
    graphs_by_bin_copy = {k: list(v) for k, v in graphs_by_bin.items()}

    print("Running final selection pass...")
    for bin_label in active_bins:
        # Use the dynamically calculated final target
        target_for_this_bin = final_targets.get(bin_label, 0)

        available_in_bin_list = graphs_by_bin_copy.get(bin_label, [])
        num_available = len(available_in_bin_list)

        # Number to take is the final target, capped by actual original availability
        num_to_take = min(target_for_this_bin, num_available)

        if num_to_take != target_for_this_bin and target_for_this_bin > 0:
             print(f"  Info: Bin {bin_label}: Final target was {target_for_this_bin} but only {num_available} available originally. Taking {num_to_take}.")

        if num_to_take > 0:
            # Randomly sample the required number from the list for this bin
            selected = random.sample(available_in_bin_list, num_to_take)
            selected_graphs_map[bin_label] = selected
            graphs_taken_count += num_to_take
            # No need to manage available_for_remainder during selection

        print(f"  Bin {bin_label}: Final Target={target_for_this_bin}, Taking={num_to_take}")

    print(f"Selection pass selected {graphs_taken_count} graphs.")

    # Sanity check final count
    if graphs_taken_count != final_calculated_total:
         # This might happen if adjustment logic had issues
         print(f"Warning: Graphs selected ({graphs_taken_count}) differs from final calculated target ({final_calculated_total})!")
    if abs(graphs_taken_count - target_total_graphs) > 1: # Allow tolerance for rounding?
         print(f"Warning: Final selected count ({graphs_taken_count}) differs significantly from overall target ({target_total_graphs})!")

    # --- Step 7: Combine Final Selection ---
    final_graphs = []
    print("\n--- Combining Final Selection ---")
    for bin_label in active_bins: # Combine in sorted bin order
        if bin_label in selected_graphs_map:
            final_graphs.extend(selected_graphs_map[bin_label])
    print(f"Total graphs selected for final dataset: {len(final_graphs)}")


    # --- Step 8: Final Analysis ---
    print("\n--- Final Dataset Statistics ---")
    _ = print_node_distribution(final_graphs, f"Final Dataset ({final_file})")


    # --- Step 9: Save Final Dataset ---
    print(f"\n--- Saving Final Dataset ---")
    try:
        with open(final_file, 'wb') as f: pickle.dump(final_graphs, f)
        print(f"Successfully saved {len(final_graphs)} graphs to {final_file}")
    except Exception as e: print(f"Error saving final dataset to {final_file}: {e}"); return


    # --- Step 10: Determine and Save Remainder ---
    print(f"\n--- Determining and Saving Remainder Graphs ---")
    final_graph_ids = {id(g) for g in final_graphs}
    remainder_graphs = [g for g_id, g in all_graphs_map.items() if g_id not in final_graph_ids]

    expected_total = len(all_graphs_map)
    actual_total_after_split = len(final_graphs) + len(remainder_graphs)
    if expected_total != actual_total_after_split: print(f"Warning: Count mismatch! Initial unique {expected_total} vs Final+Remainder {actual_total_after_split}.")
    else: print(f"Count check OK: Initial Unique ({expected_total}) = Final ({len(final_graphs)}) + Remainder ({len(remainder_graphs)})")

    if remainder_graphs:
        _ = print_node_distribution(remainder_graphs, f"Remainder Graphs ({remainder_file})")
        try:
            with open(remainder_file, 'wb') as f: pickle.dump(remainder_graphs, f)
            print(f"Successfully saved {len(remainder_graphs)} unused graphs to {remainder_file}")
        except Exception as e: print(f"Error saving remainder graphs to {remainder_file}: {e}")
    else: print(f"No remainder graphs to save ({remainder_file}).")

    print("\n--- Waterfall Redistribution Process Complete ---") # Updated message
# =============================================================================
# === MODIFIED balance_graph_distribution function END                      ===
# =============================================================================


def main():
    """Main execution function."""
    args = parse_args()

    # Modified print statement to reflect the new strategy
    print("--- Starting Graph Balancing Script (Waterfall Redistribution Strategy) ---")
    print(f"Input File 1:        {args.input_file1}")
    print(f"Input File 2:        {args.input_file2}")
    print(f"Final Output File:   {args.final_file}")
    print(f"Remainder File:      {args.remainder_file}")
    print(f"Target Total Graphs: {args.target_total_graphs}")
    print(f"Balancing Bins:      {[get_node_bin(low) for low, high in NODE_BINS]}")
    print(f"Force Combine:       {args.force}")
    print("-" * 60)

    try:
        balance_graph_distribution_evenly( # Keep name or change if function renamed
            input_file1=args.input_file1,
            input_file2=args.input_file2,
            final_file=args.final_file,
            remainder_file=args.remainder_file,
            target_total_graphs=args.target_total_graphs,
            force=args.force
        )
    except Exception as e:
        print(f"\n--- SCRIPT FAILED ---")
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n--- Script Finished Successfully ---")
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

