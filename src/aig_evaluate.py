"""
Contains specific functions for evaluating the validity of generated And-Inverter Graphs (AIGs).
"""

import numpy as np
import networkx as nx
from collections import Counter
from typing import Dict, Any, List

# Attempt to import necessary constants and helpers from aig_dataset
try:
    from aig_dataset import NODE_TYPES, EDGE_TYPES, _calculate_levels
except ImportError:
    print("Warning: Could not import from aig_dataset. Assuming default values for NODE_TYPES/EDGE_TYPES.")
    # Define fallbacks if import fails
    NODE_TYPES = {"PI": 1, "AND": 2, "PO": 3, "ZERO": 0, "UNKNOWN": -1}
    EDGE_TYPES = {"NONE": 0, "REGULAR": 1, "INVERTED": 2}

def infer_node_types(g: nx.DiGraph) -> Dict[Any, str]:
    """Infers node types (PI, AND, PO, UNKNOWN, INVALID_FANIN) based on degrees."""
    types = {}
    if not isinstance(g, nx.DiGraph) or g.number_of_nodes() == 0:
        return types
    for n in g.nodes():
        in_deg = g.in_degree(n)
        out_deg = g.out_degree(n)
        if in_deg == 0:
            types[n] = "PI"
        elif in_deg == 2:
            types[n] = "AND"
        elif out_deg == 0 and in_deg >= 1: # POs are sinks with at least one input
            types[n] = "PO"
        elif in_deg == 1:
             # Could be intermediate buffer, or part of invalid structure
             types[n] = "UNKNOWN" # Nodes with fan-in 1
        else: # e.g., in_deg > 2 or other cases not covered above
             types[n] = "INVALID_FANIN" # Node with unexpected fanin
    return types

def calculate_paper_validity(g: nx.DiGraph) -> float:
    """
    Calculates validity based on a specific interpretation:
    Percentage of inferred AND gates with 2 inputs and inferred PO gates
    with 1 input.

    Returns:
        Validity score (0.0 to 1.0), or 0.0 if no relevant gates found or graph is invalid.
    """
    if not isinstance(g, nx.DiGraph) or g.number_of_nodes() == 0:
        return 0.0

    inferred_types = infer_node_types(g)
    valid_gates = 0
    total_relevant_gates = 0

    for node, node_type in inferred_types.items():
        in_deg = g.in_degree(node)
        if node_type == "AND":
            total_relevant_gates += 1
            if in_deg == 2:
                valid_gates += 1
        elif node_type == "PO":
            total_relevant_gates += 1
            # --- Interpretation: Output gates need exactly 1 input ---
            if in_deg == 1:
                valid_gates += 1

    if total_relevant_gates == 0:
        # If graph has nodes but no AND/PO, validity is 0 based on this metric
        return 0.0
    else:
        return float(valid_gates) / total_relevant_gates

def calculate_extensive_validity(g: nx.DiGraph, check_connectivity=True) -> Dict[str, Any]:
    """
    Performs a more comprehensive validity check for a generated AIG.

    Checks: DAG, self-loops, existence of PIs/ANDs/POs, correct fan-in for
            inferred types, and optionally reachability from PIs.

    Args:
        g: The NetworkX DiGraph to check.
        check_connectivity: If True, checks if all nodes are reachable from PIs.

    Returns:
        A dictionary containing boolean flags for each check and an overall validity flag.
    """
    results = {
        "is_dag": False,
        "has_pis": False,
        "has_ands": False,
        "has_pos": False,
        "correct_fanin": True, # Assume true initially
        "no_self_loops": False,
        "all_reachable_from_pi": not check_connectivity, # True if not checked, else False until verified
        "overall_valid": False,
        "node_counts": {"PI": 0, "AND": 0, "PO": 0, "UNKNOWN": 0, "INVALID_FANIN": 0},
        "error_nodes": {"fanin": [], "unreachable": [], "structure": []} # Added structure errors
    }

    if not isinstance(g, nx.DiGraph) or g.number_of_nodes() == 0:
        results["error_nodes"]["structure"].append("Empty or invalid graph object")
        return results # Basic checks fail on empty graph

    # 1. Self-loops Check
    num_self_loops = 0
    try:
        # nx.number_of_selfloops might not exist in all versions or raise error on non-simple graphs
        # Safer way:
        num_self_loops = sum(1 for u, v in g.edges() if u == v)
    except Exception as e:
         results["error_nodes"]["structure"].append(f"Self-loop check failed: {e}")
         return results # Cannot proceed reliably

    results["no_self_loops"] = (num_self_loops == 0)
    if num_self_loops > 0:
         results["error_nodes"]["structure"].append(f"Found {num_self_loops} self-loops")
         return results # Stop if self-loops exist

    # 2. DAG Check
    try:
        results["is_dag"] = nx.is_directed_acyclic_graph(g)
    except Exception as e:
         results["error_nodes"]["structure"].append(f"DAG check failed: {e}")
         return results # Cannot proceed

    if not results["is_dag"]:
         results["error_nodes"]["structure"].append("Not a DAG")
         return results # Stop further checks that rely on DAG properties

    # 3. Infer types and check fan-in/existence
    inferred_types = infer_node_types(g)
    pi_nodes = set()
    results["node_counts"] = Counter(inferred_types.values()) # Count inferred types

    results["has_pis"] = results["node_counts"]["PI"] > 0
    results["has_ands"] = results["node_counts"]["AND"] > 0
    results["has_pos"] = results["node_counts"]["PO"] > 0

    for node, node_type in inferred_types.items():
        if node_type == "PI": pi_nodes.add(node)
        in_deg = g.in_degree(node)
        correct_fanin_for_type = True
        error_msg = None

        if node_type == "PI" and in_deg != 0:
             correct_fanin_for_type = False
             error_msg = f"Node {node} (PI) has {in_deg} inputs (expected 0)"
        elif node_type == "AND" and in_deg != 2:
             correct_fanin_for_type = False
             error_msg = f"Node {node} (AND) has {in_deg} inputs (expected 2)"
        elif node_type == "PO" and in_deg < 1:
             correct_fanin_for_type = False
             error_msg = f"Node {node} (PO) has {in_deg} inputs (expected >=1)"
        elif node_type == "INVALID_FANIN":
             correct_fanin_for_type = False
             error_msg = f"Node {node} has invalid fanin ({in_deg})"
        # Allow UNKNOWN type (fanin=1) - could add check here if needed

        if not correct_fanin_for_type:
            results["correct_fanin"] = False
            if error_msg: # Only append if there's a specific error message
                results["error_nodes"]["fanin"].append(error_msg)

    # If basic type existence or fanin failed, mark overall as invalid early
    if not results["has_pis"] or not results["has_ands"] or not results["has_pos"] or not results["correct_fanin"]:
        # Don't return yet, let connectivity check run if requested for debugging info
        pass

    # 4. Connectivity Check (Optional)
    if check_connectivity:
        if not results["has_pis"]:
             # If checking connectivity but no PIs exist, it fails unless graph is just PIs/empty
             # If graph has nodes but no PIs, it's unreachable based on this check
             if g.number_of_nodes() > 0:
                 results["all_reachable_from_pi"] = False
                 results["error_nodes"]["unreachable"].append("No PIs to check reachability from")
             else: # Empty graph case handled earlier
                 results["all_reachable_from_pi"] = True # Vacuously true? Or depends on definition
        else:
            # Check reachability only if there are PIs
            all_nodes_reachable = True
            non_pi_nodes = set(g.nodes()) - pi_nodes
            if non_pi_nodes: # Only check if there are non-PI nodes
                reachable_nodes = set(pi_nodes) # Start with PIs
                for pi_node in pi_nodes:
                    try:
                        reachable_nodes.update(nx.descendants(g, pi_node))
                    except nx.NetworkXError as e:
                         print(f"Warning: NetworkX error during reachability check: {e}")
                         all_nodes_reachable = False
                         results["error_nodes"]["structure"].append("Reachability calc error")
                         break # Stop connectivity check on error

                if all_nodes_reachable: # Check subset only if no error happened
                    unreached = non_pi_nodes - reachable_nodes
                    if unreached:
                        all_nodes_reachable = False
                        results["error_nodes"]["unreachable"] = sorted(list(unreached)) # Store sorted list

            results["all_reachable_from_pi"] = all_nodes_reachable
    # else: connectivity not checked, result remains as initialized


    # 5. Overall Validity Calculation
    results["overall_valid"] = (
        results["is_dag"] and
        results["no_self_loops"] and
        results["has_pis"] and
        results["has_ands"] and
        results["has_pos"] and
        results["correct_fanin"] and
        results["all_reachable_from_pi"] # Takes value based on check_connectivity flag
    )

    return results