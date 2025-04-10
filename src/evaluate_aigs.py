# evaluate_aigs.py
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import warnings
from typing import Dict, Any, List, Optional, Set, Tuple

# --- Constants ---
# Define necessary constants here if not imported from elsewhere
EDGE_TYPES = {"NONE": 0, "REGULAR": 1, "INVERTED": 2}
# NODE_TYPES might be needed if visualization uses predefined types, but inference is preferred
NODE_TYPES_INFERRED = {"PI": 1, "AND": 2, "PO": 3, "UNKNOWN": -1, "INVALID_FANIN": -2}

# --- Logger ---
logger = logging.getLogger("evaluate_aigs")
# Basic config if run standalone, but usually configured by the main script
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- AIG Conversion ---
def aig_to_networkx(adj_conn: np.ndarray, adj_inv: np.ndarray) -> nx.DiGraph:
    """
    Converts AIG connectivity and inversion matrices into a NetworkX DiGraph.
    Edges have an 'type' attribute (1 for regular, 2 for inverted).
    """
    G = nx.DiGraph()
    if adj_conn.ndim != 2 or adj_conn.shape[0] != adj_conn.shape[1]:
        logger.error("Invalid adjacency matrix shape for AIG conversion.")
        return G
    if adj_conn.shape != adj_inv.shape:
        logger.error("Connectivity and inversion matrices shapes mismatch.")
        return G

    num_nodes = adj_conn.shape[0]
    if num_nodes == 0:
        return G # Return empty graph if 0 nodes

    G.add_nodes_from(range(num_nodes))

    sources, targets = np.where(adj_conn > 0)
    for u, v in zip(sources, targets):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
             # Determine edge type based on inversion matrix
             edge_type = EDGE_TYPES["INVERTED"] if adj_inv[u, v] > 0 else EDGE_TYPES["REGULAR"]
             G.add_edge(u, v, type=edge_type) # Store type (1 or 2)
        else:
             logger.warning(f"Edge index ({u}, {v}) out of bounds for {num_nodes} nodes during conversion.")

    return G

# --- Node Type Inference ---
def infer_node_types(g: nx.DiGraph) -> Dict[Any, str]:
    """
    Infers node types (PI, AND, PO, UNKNOWN, INVALID_FANIN) based on degrees.
    """
    types = {}
    if not isinstance(g, nx.DiGraph) or g.number_of_nodes() == 0:
        return types

    for n in g.nodes():
        in_deg = g.in_degree(n)
        out_deg = g.out_degree(n)

        if in_deg == 0:
            types[n] = "PI"
        elif in_deg > 2:
             types[n] = "INVALID_FANIN"
        elif in_deg == 2:
            # If it's a sink node (out_deg=0) with fanin 2, could be PO or AND. Prioritize AND.
            # If it's not a sink, it's an AND gate.
            types[n] = "AND" # Assume AND, PO check below might override for sinks
        elif in_deg == 1:
            # If fanin 1 and sink -> PO. If not sink -> Unknown/Buffer.
             if out_deg == 0:
                 types[n] = "PO"
             else:
                 types[n] = "UNKNOWN" # Treat non-sink fanin-1 nodes as unknown
        else: # Should not happen if graph is connected from PIs
             types[n] = "UNKNOWN"

        # Refine PO definition: any sink node (out_deg 0) with valid fan-in (1 or 2) is a PO
        if out_deg == 0:
            if in_deg == 1 or in_deg == 2:
                 types[n] = "PO"
            # If in_deg == 0 and out_deg == 0 (isolated node), it was already marked PI. Keep as PI.
            # If in_deg > 2, already marked INVALID_FANIN.

    # Ensure all nodes have a type assigned (fallback)
    for n in g.nodes():
        if n not in types:
            logger.warning(f"Node {n} had no inferred type, marking UNKNOWN.")
            types[n] = "UNKNOWN"

    return types


# --- Structural Evaluation Metrics ---
def calculate_seadag_validity(G: nx.DiGraph) -> float:
    """
    Calculates SEADAG-like validity based on fan-in of inferred AND/PO nodes.
    Returns 1.0 if all inferred AND nodes have fan-in 2 and inferred PO nodes have fan-in 1 or 2.
    Returns 1.0 for empty graphs or graphs with no inferred AND/PO nodes.
    """
    if G.number_of_nodes() == 0:
        return 1.0

    node_types = infer_node_types(G)
    relevant_gates = 0
    correct_fanin_gates = 0

    for node, node_type in node_types.items():
        fan_in = G.in_degree(node)
        if node_type == "AND":
            relevant_gates += 1
            if fan_in == 2:
                correct_fanin_gates += 1
        elif node_type == "PO":
            relevant_gates += 1
            if fan_in == 1 or fan_in == 2: # Allow fan-in 1 or 2 for POs
                correct_fanin_gates += 1
        # Ignore PI, UNKNOWN, INVALID_FANIN for this specific metric

    if relevant_gates == 0:
        return 1.0 # Vacuously valid if no relevant gates found
    else:
        validity = correct_fanin_gates / relevant_gates
        return validity

def calculate_structural_aig_validity(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculates broader structural AIG validity: DAG check, valid types ratio, type counts.
    """
    results = {
        'is_dag': False, 'validity_score': 0.0,
        'has_pi': False, 'has_po': False, 'has_and': False,
        'has_unknown': False, 'has_invalid_fanin': False,
        'num_nodes': 0, 'num_pi': 0, 'num_po': 0, 'num_and': 0,
        'num_unknown': 0, 'num_invalid_fanin': 0
    }
    num_nodes = G.number_of_nodes()
    results['num_nodes'] = num_nodes

    if num_nodes == 0:
        results['is_dag'] = True
        results['validity_score'] = 1.0
        return results

    # 1. Check DAG property
    try:
        is_dag = nx.is_directed_acyclic_graph(G)
    except Exception as e:
        logger.error(f"Error checking DAG property: {e}")
        is_dag = False # Assume not DAG if check fails
    results['is_dag'] = is_dag

    # 2. Infer Node Types
    node_types = infer_node_types(G)
    if not node_types: return results # Return defaults if inference failed

    # 3. Iterate, Count Types, and Check Validity
    valid_structural_node_count = 0
    for node, node_type in node_types.items():
        if node_type == "PI":
            results['num_pi'] += 1; results['has_pi'] = True; valid_structural_node_count += 1
        elif node_type == "AND":
            results['num_and'] += 1; results['has_and'] = True; valid_structural_node_count += 1
        elif node_type == "PO":
            results['num_po'] += 1; results['has_po'] = True; valid_structural_node_count += 1
        elif node_type == "UNKNOWN":
            results['num_unknown'] += 1; results['has_unknown'] = True
        elif node_type == "INVALID_FANIN":
            results['num_invalid_fanin'] += 1; results['has_invalid_fanin'] = True

    # 4. Calculate validity score (fraction of structurally valid types: PI, AND, PO)
    # Score is 0 if not a DAG
    if is_dag:
        if num_nodes > 0:
            results['validity_score'] = valid_structural_node_count / num_nodes
        else:
             results['validity_score'] = 1.0 # Should be caught earlier
    else:
        results['validity_score'] = 0.0 # Not a DAG, fundamental structure is invalid

    return results

def count_pi_po_paths(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Counts PIs reaching POs and POs reachable from PIs, assuming DAG structure.
    """
    results = {
        'error': None, 'is_dag': True, 'num_pis': 0, 'num_pos': 0,
        'num_pis_reaching_po': 0, 'num_pos_reachable_from_pi': 0,
        'fraction_pis_connected': 0.0, 'fraction_pos_connected': 0.0
    }
    if G.number_of_nodes() == 0: return results

    try:
        if not nx.is_directed_acyclic_graph(G):
            results['is_dag'] = False
            results['error'] = 'Graph is not a DAG.'
            # Proceed with type inference but path results are questionable
    except Exception as e:
         logger.error(f"Error checking DAG property in count_pi_po_paths: {e}")
         results['is_dag'] = False # Assume not DAG
         results['error'] = f'DAG check failed: {e}'

    node_types = infer_node_types(G)
    pis = {n for n, t in node_types.items() if t == "PI"}
    pos = {n for n, t in node_types.items() if t == "PO"}
    results['num_pis'] = len(pis)
    results['num_pos'] = len(pos)

    if not pis or not pos: return results # No paths possible

    connected_pis: Set[Any] = set()
    connected_pos: Set[Any] = set()

    # Efficiently find all nodes reachable from any PI
    all_reachable_from_pis = set()
    for pi_node in pis:
        try:
            all_reachable_from_pis.update(nx.descendants(G, pi_node))
            all_reachable_from_pis.add(pi_node) # Include the PI itself
        except nx.NodeNotFound: continue
        except Exception as e:
            logger.warning(f"Error finding descendants from PI {pi_node}: {e}")

    # Efficiently find all nodes that can reach any PO
    all_ancestors_of_pos = set()
    for po_node in pos:
        try:
            all_ancestors_of_pos.update(nx.ancestors(G, po_node))
            all_ancestors_of_pos.add(po_node) # Include the PO itself
        except nx.NodeNotFound: continue
        except Exception as e:
            logger.warning(f"Error finding ancestors for PO {po_node}: {e}")

    # Find PIs that can reach at least one PO
    connected_pis = pis.intersection(all_ancestors_of_pos)
    # Find POs that are reachable from at least one PI
    connected_pos = pos.intersection(all_reachable_from_pis)

    results['num_pis_reaching_po'] = len(connected_pis)
    results['num_pos_reachable_from_pi'] = len(connected_pos)
    if results['num_pis'] > 0:
        results['fraction_pis_connected'] = len(connected_pis) / results['num_pis']
    if results['num_pos'] > 0:
        results['fraction_pos_connected'] = len(connected_pos) / results['num_pos']

    return results


# --- Visualization ---
def visualize_aig_structure(G: nx.DiGraph, output_file='generated_aig_structure.png'):
    """ Visualize the generated AIG structure, using inferred types and edge types. """
    if G is None or not isinstance(G, nx.DiGraph) or G.number_of_nodes() == 0:
        logger.info(f"Skipping visualization for empty/invalid graph: {output_file}")
        return

    plt.figure(figsize=(16, 14)) # Slightly larger figure
    pos = None
    try:
        # Try graphviz layout first (requires pygraphviz or pydot)
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        logger.debug("Using graphviz 'dot' layout for visualization.")
    except ImportError:
        logger.warning("pygraphviz/pydot not found. Using spring_layout (less structured).")
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42) # Adjust spring layout params
    except Exception as e: # Catch other layout errors (e.g., Graphviz not installed)
        logger.warning(f"Graphviz layout failed ({e}). Using spring_layout.")
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # Get inferred types (use pre-computed if available, otherwise compute)
    node_types = G.graph.get('_inferred_types_cleaned', None)
    if not node_types:
        logger.debug("Inferring node types for visualization.")
        node_types = infer_node_types(G)
        G.graph['_inferred_types_cleaned'] = node_types # Store for potential reuse

    # Node colors and labels based on inferred types
    node_colors = []
    node_labels = {}
    color_map = {"PI": 'palegreen', "AND": 'lightskyblue', "PO": 'lightcoral',
                 "UNKNOWN": 'lightgrey', "INVALID_FANIN": 'orange', "DEFAULT": 'white'}

    for node in sorted(G.nodes()): # Sort nodes for potentially more consistent layouts
        node_type = node_types.get(node, "UNKNOWN")
        node_colors.append(color_map.get(node_type, color_map["DEFAULT"]))
        node_labels[node] = f"{node}\n({node_type})"

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.9)

    # Separate edges by type (regular/inverted)
    regular_edges = []
    inverted_edges = []
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type', EDGE_TYPES["REGULAR"]) # Default if missing
        if edge_type == EDGE_TYPES["INVERTED"]:
            inverted_edges.append((u, v))
        else: # Treat NONE or REGULAR as regular for drawing
            regular_edges.append((u, v))

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges,
                           width=1.5, edge_color='black', style='solid',
                           arrows=True, arrowsize=12, node_size=700,
                           connectionstyle='arc3,rad=0.1') # Use curved edges slightly
    nx.draw_networkx_edges(G, pos, edgelist=inverted_edges,
                           width=1.5, edge_color='red', style='dashed',
                           arrows=True, arrowsize=12, node_size=700,
                           connectionstyle='arc3,rad=0.1')

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=1.5, linestyle='solid', label=f'Regular Edge'),
        plt.Line2D([0], [0], color='red', lw=1.5, linestyle='dashed', label=f'Inverted Edge'),
        plt.scatter([], [], s=80, color=color_map["PI"], label='Inferred PI'),
        plt.scatter([], [], s=80, color=color_map["AND"], label='Inferred AND'),
        plt.scatter([], [], s=80, color=color_map["PO"], label='Inferred PO'),
        plt.scatter([], [], s=80, color=color_map["INVALID_FANIN"], label='Invalid Fan-in'),
        plt.scatter([], [], s=80, color=color_map["UNKNOWN"], label='Unknown/Other')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize='small',
               bbox_to_anchor=(1.1, 1.1), frameon=True, facecolor='white', framealpha=0.8)

    plt.title(f'Generated AIG Structure - {os.path.basename(output_file)}', fontsize=14)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving visualization {output_file}: {e}")
    finally:
        plt.close() # Close the figure


# Example of how to use if run standalone (for testing)
if __name__ == '__main__':
    print("This file contains evaluation and visualization functions. Run get_aigs.py to execute.")
    # Example: Test aig_to_networkx and visualize
    # test_conn = np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0,0,0,0]])
    # test_inv = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0,0,0,0]])
    # g_test = aig_to_networkx(test_conn, test_inv)
    # print("Test Graph Nodes:", g_test.nodes())
    # print("Test Graph Edges:", g_test.edges(data=True))
    # struct_validity = calculate_structural_aig_validity(g_test)
    # print("Structural Validity:", struct_validity)
    # visualize_aig_structure(g_test, "test_aig_viz.png")