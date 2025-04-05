#!/usr/bin/env python3
"""
AIG File Processor

This script processes all .aig files in a specified directory by:
1. Removing some output signals
2. Cleaning up dangling nodes
3. Ensuring each AIG has exactly 2 primary outputs (POs)
4. Writing the modified AIG back to the original file
"""

import os
import random
from typing import List, Dict, Optional, Set
from aigverse import Aig, read_aiger_into_aig, write_aiger


def topological_sort(aig: Aig) -> List[int]:
    """
    Perform a topological sort of the nodes in the AIG.

    Args:
        aig: The AIG network

    Returns:
        A list of node indices in topological order
    """
    visited: Set[int] = set()
    topo_order: List[int] = []

    def visit(node_idx: int) -> None:
        if node_idx in visited:
            return

        visited.add(node_idx)
        node = aig.index_to_node(node_idx)

        # Visit fanins first
        if not aig.is_constant(node) and not aig.is_pi(node):
            for fanin in aig.fanins(node):
                fanin_node = aig.get_node(fanin)
                fanin_idx = int(fanin_node)
                visit(fanin_idx)

        topo_order.append(node_idx)

    # Visit all nodes
    for node in aig.nodes():
        node_idx = int(node)
        visit(node_idx)

    return topo_order


def process_aig_file(file_path: str) -> None:
    """
    Process a single AIG file according to the requirements.

    Args:
        file_path: Path to the .aig file to process

    Returns:
        None. The original file is replaced with the modified AIG.
    """
    try:
        # Read the original AIG file
        original_aig = read_aiger_into_aig(file_path)
        print(f"Processing {file_path}: Original AIG has {original_aig.num_pos()} POs")

        # Create a new empty AIG
        new_aig = Aig()

        # Map to track correspondence between original nodes and new nodes
        node_map: Dict[int, int] = {}

        # Add constant node (typically at index 0)
        node_map[0] = 0

        # Create primary inputs in the new AIG
        pi_signals = []
        for i, pi in enumerate(original_aig.pis()):
            new_pi = new_aig.create_pi()
            pi_signals.append(new_pi)
            node_map[int(pi)] = int(new_aig.get_node(new_pi))

        # Get topologically sorted nodes to ensure we process nodes in the correct order
        topo_order = topological_sort(original_aig)

        # Process nodes in topological order
        for node_idx in topo_order:
            node = original_aig.index_to_node(node_idx)

            # Skip constants and PIs (already handled)
            if original_aig.is_constant(node) or original_aig.is_pi(node):
                continue

            # Process AND gates
            if original_aig.is_and(node):
                fanins = original_aig.fanins(node)

                if len(fanins) != 2:
                    print(f"Warning: AND gate with {len(fanins)} fanins encountered. Skipping.")
                    continue

                new_fanins = []
                all_fanins_found = True

                for fanin in fanins:
                    fanin_node = original_aig.get_node(fanin)
                    fanin_idx = int(fanin_node)

                    if fanin_idx in node_map:
                        new_node = new_aig.index_to_node(node_map[fanin_idx])
                        new_signal = new_aig.make_signal(new_node)

                        # Handle complement
                        if original_aig.is_complemented(fanin):
                            new_signal = ~new_signal

                        new_fanins.append(new_signal)
                    else:
                        print(f"Error: Fanin node {fanin_idx} not found in node_map")
                        all_fanins_found = False
                        break

                if all_fanins_found and len(new_fanins) == 2:
                    new_signal = new_aig.create_and(new_fanins[0], new_fanins[1])
                    node_map[node_idx] = int(new_aig.get_node(new_signal))

        # Handle primary outputs
        original_pos = []
        for i in range(original_aig.num_pos()):
            original_pos.append(original_aig.po_at(i))

        # Decide how many POs to add based on requirements
        pos_to_add: List[Optional[int]] = []

        if original_aig.num_pos() == 2:
            # Keep both original POs
            pos_to_add = list(range(2))
        elif original_aig.num_pos() > 2:
            # Randomly select 2 POs
            pos_to_add = random.sample(range(original_aig.num_pos()), 2)
        else:  # original_aig.num_pos() < 2
            # Add all original POs
            pos_to_add = list(range(original_aig.num_pos()))
            # Add random nodes as POs until we have 2
            while len(pos_to_add) < 2:
                pos_to_add.append(None)  # Placeholder for a random node

        # Add the selected POs to the new AIG
        added_pos = 0
        for po_idx in pos_to_add:
            if po_idx is not None:
                # Use an original PO
                original_po = original_pos[po_idx]
                original_node = original_aig.get_node(original_po)
                idx = int(original_node)

                if idx in node_map:
                    new_node = new_aig.index_to_node(node_map[idx])
                    new_signal = new_aig.make_signal(new_node)
                    if original_aig.is_complemented(original_po):
                        new_signal = ~new_signal
                    new_aig.create_po(new_signal)
                    added_pos += 1
                else:
                    # If we can't find the node, use a random PI instead
                    print(f"Warning: PO node {idx} not found in node_map, using random PI instead")
                    random_pi = random.choice(pi_signals)
                    new_aig.create_po(random_pi)
                    added_pos += 1
            else:
                # Add a random PI as PO
                random_pi = random.choice(pi_signals)
                new_aig.create_po(random_pi)
                added_pos += 1

        # Clean up dangling nodes
        new_aig.cleanup_dangling()

        # Write the modified AIG back to the original file
        write_aiger(new_aig, file_path)
        print(f"Finished processing {file_path}: New AIG has {new_aig.num_pos()} POs")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def process_all_aig_files(directory: str) -> None:
    """
    Process all .aig files in the specified directory.

    Args:
        directory: Path to the directory containing .aig files

    Returns:
        None
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return

    # Get all .aig files in the directory
    aig_files = [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.endswith('.aig') and os.path.isfile(os.path.join(directory, f))]

    if not aig_files:
        print(f"No .aig files found in {directory}")
        return

    print(f"Found {len(aig_files)} .aig files in {directory}")

    # Process each file
    for file_path in aig_files:
        process_aig_file(file_path)


def main() -> None:
    """
    Main function to parse command line arguments and process AIG files.
    """
    import argparse

    # parser = argparse.ArgumentParser(description='Process AIG files to ensure each has exactly 2 POs.')
    # parser.add_argument('directory', help='Directory containing .aig files to process')
    #
    # args = parser.parse_args()

    process_all_aig_files("./aiger")


if __name__ == "__main__":
    main()