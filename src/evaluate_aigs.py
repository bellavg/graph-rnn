# evaluate_aigs.py
import pickle
import networkx as nx
import argparse
import os
import logging
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np
import sys
from tqdm import tqdm
import json # Added for loading metadata
# Removed torch import as it's not needed after removing bin loading

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evaluate_aigs_pkl") # Changed logger name slightly

# --- Import the AIG configuration ---
# Ensure aig_config.py is in the same directory or Python path
try:
    import aig_config as aig_config
except ImportError:
    # Attempt relative import if run from src/
    try:
        from . import aig_config
        logger.info("Imported aig_config using relative path.")
    except ImportError:
        logger.error("Failed to import AIG configuration from 'aig_config.py' or '.aig_config'. Ensure it's accessible.")
        sys.exit(1)
# Derive constants from config (Ensure these match your aig_config.py)
# It's safer to access them via the imported module
VALID_AIG_NODE_TYPES = set(aig_config.NODE_TYPE_KEYS)
VALID_AIG_EDGE_TYPES = set(aig_config.EDGE_TYPE_KEYS)
NODE_CONST0 = aig_config.NODE_TYPE_KEYS[0] # Example: "NODE_CONST0"
NODE_PI = aig_config.NODE_TYPE_KEYS[1]     # Example: "NODE_PI"
NODE_AND = aig_config.NODE_TYPE_KEYS[2]    # Example: "NODE_AND"
NODE_PO = aig_config.NODE_TYPE_KEYS[3]     # Example: "NODE_PO"
MIN_AND_COUNT_CONFIG = aig_config.MIN_AND_COUNT
MIN_PO_COUNT_CONFIG = aig_config.MIN_PO_COUNT
# PAD_VALUE is likely not needed here anymore as we don't unpad bin files
# PAD_VALUE = aig_config.PAD_VALUE

# --- Structural Metrics Calculation (Unchanged) ---
def calculate_structural_aig_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculates structural AIG validity metrics based on assigned types.
    Counts violations across the graph instead of breaking early.
    Uses constants derived from aig_config.
    Returns a dictionary of detailed metrics and violation counts.
    """
    metrics = {
        'num_nodes': 0,
        'is_dag': False,
        'num_pi': 0, 'num_po': 0, 'num_and': 0, 'num_const0': 0,
        'num_unknown_nodes': 0,
        'num_unknown_edges': 0,
        'pi_indegree_violations': 0,
        'const0_indegree_violations': 0,
        'and_indegree_violations': 0,
        'po_outdegree_violations': 0,
        'po_indegree_violations': 0,
        'isolated_nodes': 0, # Still counts relevant isolates for reporting
        'is_structurally_valid': False, # The key flag indicating validity
        'constraints_failed': [] # List to store reasons for failure
    }

    num_nodes = G.number_of_nodes()
    metrics['num_nodes'] = num_nodes

    if not isinstance(G, nx.DiGraph) or num_nodes == 0:
        metrics['constraints_failed'].append("Empty or Invalid Graph Object")
        metrics['is_structurally_valid'] = False # Explicitly set invalid
        return metrics # Return early for invalid input

    # 1. Check DAG property (Critical)
    try: # Add try-except for robustness
        metrics['is_dag'] = nx.is_directed_acyclic_graph(G)
        if not metrics['is_dag']:
            metrics['constraints_failed'].append("Not a DAG")
    except Exception as e:
         logger.warning(f"DAG check failed for a graph: {e}")
         metrics['is_dag'] = False # Assume not DAG if check fails
         metrics['constraints_failed'].append(f"DAG Check Error: {e}")


    # 2. Check Node Types and Basic Degrees
    node_type_counts = Counter()
    unknown_node_indices = []
    for node, data in G.nodes(data=True):
        # Handle cases where node data might not be a dictionary
        # Assume the PKL file stores node types as strings under the 'type' key
        if isinstance(data, dict):
             node_type = data.get('type')
             # --- Type Validation ---
             # Check if the type string is one of the expected keys
             if node_type not in VALID_AIG_NODE_TYPES:
                 logger.debug(f"Node {node} has unexpected type '{node_type}'. Treating as unknown.")
                 metrics['num_unknown_nodes'] += 1
                 unknown_node_indices.append(node)
                 node_type = "UNKNOWN_NODE" # Mark internally as unknown
             # --- End Validation ---
        else:
             node_type = "Error: Node data not dict"
             logger.warning(f"Node {node} data is not a dictionary: {data}")
             metrics['num_unknown_nodes'] += 1
             unknown_node_indices.append(node)

        node_type_counts[node_type] += 1

        # Skip degree checks for unknown nodes or nodes with data errors
        if node_type in ["UNKNOWN_NODE", "Error: Node data not dict"]:
            continue

        # Check degrees - Add try-except for robustness
        try:
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
        except Exception as e:
             logger.warning(f"Could not get degree for node {node}: {e}")
             # Mark as violation? Or just skip checks? Let's add a general violation count later if invalid.
             metrics['constraints_failed'].append(f"Degree Check Error for node {node}")
             continue

        # Check degrees based on assigned type (using defined type strings)
        if node_type == NODE_CONST0:
            metrics['num_const0'] += 1
            if in_deg != 0: metrics['const0_indegree_violations'] += 1
        elif node_type == NODE_PI:
            metrics['num_pi'] += 1
            if in_deg != 0: metrics['pi_indegree_violations'] += 1
        elif node_type == NODE_AND:
            metrics['num_and'] += 1
            if in_deg != 2: metrics['and_indegree_violations'] += 1
        elif node_type == NODE_PO:
            metrics['num_po'] += 1
            if out_deg != 0: metrics['po_outdegree_violations'] += 1
            # PO must have inputs.
            if in_deg == 0: metrics['po_indegree_violations'] += 1 # Keep check for in_degree == 0


    # Add failure reasons based on type/degree checks to the list
    if metrics['num_unknown_nodes'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['num_unknown_nodes']} unknown node types")
    if metrics['const0_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['const0_indegree_violations']} CONST0 nodes with incorrect in-degree")
    if metrics['pi_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['pi_indegree_violations']} PI nodes with incorrect in-degree")
    if metrics['and_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['and_indegree_violations']} AND nodes with incorrect in-degree")
    if metrics['po_outdegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['po_outdegree_violations']} PO nodes with incorrect out-degree")
    if metrics['po_indegree_violations'] > 0:
        metrics['constraints_failed'].append(f"Found {metrics['po_indegree_violations']} PO nodes with incorrect in-degree (0)")


    # 3. Check Edge Types
    for u, v, data in G.edges(data=True):
        # Assume PKL file stores edge types as strings under the 'type' key
        if isinstance(data, dict):
             edge_type = data.get('type')
             # --- Type Validation ---
             if edge_type not in VALID_AIG_EDGE_TYPES:
                 logger.debug(f"Edge ({u}-{v}) has unexpected type '{edge_type}'. Treating as unknown.")
                 metrics['num_unknown_edges'] += 1
             # --- End Validation ---
        else:
             edge_type = "Error: Edge data not dict"
             logger.warning(f"Edge ({u},{v}) data is not a dictionary: {data}")
             metrics['num_unknown_edges'] += 1

    if metrics['num_unknown_edges'] > 0:
        # Add failure reason to the list
        metrics['constraints_failed'].append(f"Found {metrics['num_unknown_edges']} unknown edge types")

    # 4. Check Basic AIG Requirements (Using config values)
    if metrics['num_pi'] == 0 and metrics['num_const0'] == 0 :
         metrics['constraints_failed'].append("No Primary Inputs or Const0 found")
    # Use MIN_AND_COUNT_CONFIG and MIN_PO_COUNT_CONFIG safely
    min_and = MIN_AND_COUNT_CONFIG if 'MIN_AND_COUNT_CONFIG' in globals() else 1
    min_po = MIN_PO_COUNT_CONFIG if 'MIN_PO_COUNT_CONFIG' in globals() else 1
    if metrics['num_and'] < min_and :
        metrics['constraints_failed'].append(f"Insufficient AND gates ({metrics['num_and']} < {min_and})")
    if metrics['num_po'] < min_po:
        metrics['constraints_failed'].append(f"Insufficient POs ({metrics['num_po']} < {min_po})")

    # --- 5. Check isolated nodes (Keep calculation, but don't add to constraints_failed) ---
    try: # Add try-except for isolates calculation
        all_isolates = list(nx.isolates(G))
        relevant_isolates = []
        for node_idx in all_isolates:
             if node_idx not in G: continue # Check if node exists
             node_data = G.nodes[node_idx]
             isolated_node_type = node_data.get('type', None) if isinstance(node_data, dict) else None
             if isolated_node_type != NODE_CONST0: # Only count non-CONST0 nodes
                 relevant_isolates.append(node_idx)
        metrics['isolated_nodes'] = len(relevant_isolates) # Count relevant isolates for reporting
    except Exception as e:
         logger.warning(f"Isolate check failed for a graph: {e}")
         metrics['isolated_nodes'] = -1 # Indicate error


    # --- Final Validity Check ---
    # A graph is structurally valid IFF it passes ALL *critical* checks:
    is_valid = (
        metrics['is_dag'] and
        metrics['num_unknown_nodes'] == 0 and
        metrics['num_unknown_edges'] == 0 and
        metrics['const0_indegree_violations'] == 0 and
        metrics['pi_indegree_violations'] == 0 and
        metrics['and_indegree_violations'] == 0 and
        metrics['po_outdegree_violations'] == 0 and
        metrics['po_indegree_violations'] == 0 and
        (metrics['num_pi'] > 0 or metrics['num_const0'] > 0) and # At least one input source
        metrics['num_and'] >= min_and and # Use safe min_and
        metrics['num_po'] >= min_po # Use safe min_po
    )
    metrics['is_structurally_valid'] = is_valid
    # If invalid, ensure constraints_failed has at least one entry
    if not is_valid and not metrics['constraints_failed']:
         metrics['constraints_failed'].append("General Validity Check Failed")
    # --- End Final Validity Check ---

    return metrics

# --- Path Counting Function (Unchanged) ---
def count_pi_po_paths(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Counts PIs reaching POs and POs reachable from PIs based on reachability.
    Uses assigned node types (defined globally from config). Assumes graph object is valid.
    """
    results = {
        'num_pi': 0, 'num_po': 0, 'num_const0': 0,
        'num_pis_reaching_po': 0, 'num_pos_reachable_from_pi': 0,
        'fraction_pis_connected': 0.0, 'fraction_pos_connected': 0.0,
        'error': None
    }
    if G.number_of_nodes() == 0:
        return results

    try:
        # Get nodes by assigned type (using defined type strings)
        pis = set()
        pos = set()
        const0_nodes = set()
        for node, data in G.nodes(data=True):
             # Ensure data is a dictionary before using .get()
             if isinstance(data, dict):
                  node_type = data.get('type')
                  if node_type == NODE_PI: pis.add(node)
                  elif node_type == NODE_PO: pos.add(node)
                  elif node_type == NODE_CONST0: const0_nodes.add(node)
             # else: logger.warning(f"Node {node} data is not a dictionary in count_pi_po_paths.")

        # Source nodes for path checking are PIs and Const0
        source_nodes = pis.union(const0_nodes)

        results['num_pi'] = len(pis) # Report actual PIs separately
        results['num_po'] = len(pos)
        results['num_const0'] = len(const0_nodes)

        if not source_nodes or not pos: # No paths possible if no sources or no POs
            return results

        connected_sources = set()
        connected_pos = set()

        # --- Perform Reachability Checks ---
        # Optimization: Precompute reachability from all sources and to all POs if needed frequently
        # For now, calculate individually

        # Find sources that can reach at least one PO
        for source_node in source_nodes:
             try:
                 if source_node not in G: continue
                 # Check reachability to ANY PO node
                 for po_node in pos:
                      if po_node not in G: continue
                      if nx.has_path(G, source_node, po_node):
                           connected_sources.add(source_node)
                           break # Source is connected, no need to check other POs for this source
             except nx.NodeNotFound: continue # Should not happen due to 'in G' check, but safe
             except Exception as e:
                  logger.warning(f"Path check failed for source {source_node}: {e}")
                  results['error'] = "Path check error" # Flag error


        # Find POs that are reachable from at least one source
        for po_node in pos:
             try:
                 if po_node not in G: continue
                 # Check reachability from ANY source node
                 for source_node in source_nodes:
                      if source_node not in G: continue
                      if nx.has_path(G, source_node, po_node):
                           connected_pos.add(po_node)
                           break # PO is connected, no need to check other sources for this PO
             except nx.NodeNotFound: continue
             except Exception as e:
                  logger.warning(f"Path check failed for PO {po_node}: {e}")
                  results['error'] = "Path check error" # Flag error

        # --- End Reachability Checks ---


        num_sources_total = len(source_nodes)
        results['num_pis_reaching_po'] = len(connected_sources) # Count includes const0 if connected
        results['num_pos_reachable_from_pi'] = len(connected_pos)

        # Calculate fractions
        if num_sources_total > 0: results['fraction_pis_connected'] = results['num_pis_reaching_po'] / num_sources_total
        if results['num_po'] > 0: results['fraction_pos_connected'] = results['num_pos_reachable_from_pi'] / results['num_po']

    except Exception as e:
        logger.error(f"Unexpected error during count_pi_po_paths execution: {e}", exc_info=True)
        results['error'] = f"Processing error: {e}"

    return results

# --- Structure Validation Function (Unchanged) ---
def validate_aig_structures(graphs: List[nx.DiGraph]) -> float:
    """
    Validates a list of NetworkX DiGraphs based on structural AIG rules.

    Args:
        graphs: A list of NetworkX DiGraph objects representing AIGs.

    Returns:
        The fraction (0.0 to 1.0) of graphs that are structurally valid.
        Returns 0.0 if the input list is empty.
    """
    num_total = len(graphs)
    if num_total == 0:
        logger.warning("validate_aig_structures received an empty list of graphs.")
        return 0.0

    num_valid_structurally = 0
    for i, graph in enumerate(graphs):
        if not isinstance(graph, nx.DiGraph):
            logger.warning(f"Item {i} in list is not a NetworkX DiGraph, counting as invalid.")
            continue
        struct_metrics = calculate_structural_aig_metrics(graph)
        if struct_metrics.get('is_structurally_valid', False):
            num_valid_structurally += 1
        # else: logger.debug(f"Graph {i} failed validation: {struct_metrics.get('constraints_failed', [])}")

    validity_fraction = (num_valid_structurally / num_total) if num_total > 0 else 0.0
    logger.info(f"Validated {num_total} graphs. Structurally Valid: {num_valid_structurally} ({validity_fraction*100:.2f}%)")
    return validity_fraction

# --- Isomorphism Helpers (Unchanged) ---
# Match node 'type' attribute (string expected)
node_matcher = nx.isomorphism.categorical_node_match('type', 'UNKNOWN_NODE')
# Match edge 'type' attribute (string expected)
edge_matcher = nx.isomorphism.categorical_edge_match('type', 'UNKNOWN_EDGE')

def are_graphs_isomorphic(G1: nx.DiGraph, G2: nx.DiGraph) -> bool:
    """Checks isomorphism considering node/edge 'type' attributes."""
    try:
        # Ensure both graphs have the 'type' attribute on nodes/edges
        # This assumes the graphs loaded from PKL and generated graphs have consistent string types
        return nx.is_isomorphic(G1, G2, node_match=node_matcher, edge_match=edge_matcher)
    except Exception as e:
        logger.warning(f"Isomorphism check failed between two graphs: {e}")
        return False

# --- Uniqueness Calculation (Unchanged) ---
def calculate_uniqueness(valid_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    """Calculates uniqueness among valid graphs."""
    num_valid = len(valid_graphs)
    if num_valid <= 1: return (1.0, num_valid)

    unique_graph_indices = []
    logger.info(f"Calculating uniqueness for {num_valid} valid graphs...")
    for i in tqdm(range(num_valid), desc="Checking Uniqueness", leave=False):
        is_unique = True
        G1 = valid_graphs[i]
        for unique_idx in unique_graph_indices:
            G2 = valid_graphs[unique_idx]
            if are_graphs_isomorphic(G1, G2):
                is_unique = False; break
        if is_unique: unique_graph_indices.append(i)

    num_unique = len(unique_graph_indices)
    uniqueness_score = num_unique / num_valid if num_valid > 0 else 0.0
    logger.info(f"Found {num_unique} unique graphs out of {num_valid} valid graphs.")
    return uniqueness_score, num_unique

# --- Novelty Calculation (Unchanged) ---
def calculate_novelty(valid_graphs: List[nx.DiGraph], train_graphs: List[nx.DiGraph]) -> Tuple[float, int]:
    """Calculates novelty against a training set."""
    num_valid = len(valid_graphs)
    num_train = len(train_graphs)
    if num_valid == 0: return (0.0, 0)
    if num_train == 0:
        logger.warning("Training set is empty, novelty will be 100%.")
        return (1.0, num_valid)

    num_novel = 0
    logger.info(f"Calculating novelty for {num_valid} valid graphs against {num_train} training graphs...")
    for gen_graph in tqdm(valid_graphs, desc="Checking Novelty", leave=False):
        is_novel = True
        for train_graph in train_graphs:
            if are_graphs_isomorphic(gen_graph, train_graph):
                is_novel = False; break
        if is_novel: num_novel += 1

    novelty_score = num_novel / num_valid if num_valid > 0 else 0.0
    logger.info(f"Found {num_novel} novel graphs out of {num_valid} valid graphs.")
    return novelty_score, num_novel

# --- REMOVED: bin_data_to_nx function ---
# --- REMOVED: load_training_graphs_from_bin function ---

# +++ NEW: Training Graph Loader for PKL Files +++
def load_training_graphs_from_pkl(train_pkl_files: List[str]) -> Optional[List[nx.DiGraph]]:
    """
    Loads training graphs directly from a list of PKL files.

    Args:
        train_pkl_files: A list of paths to the .pkl files containing lists of NetworkX graphs.

    Returns:
        A list of NetworkX DiGraphs, or None if loading fails or no files are provided.
    """
    if not train_pkl_files:
        logger.warning("No training PKL files provided.")
        return None

    all_train_graphs = []
    logger.info(f"Attempting to load training graphs from {len(train_pkl_files)} PKL file(s)...")

    for file_path in train_pkl_files:
        if not os.path.exists(file_path):
            logger.warning(f"Training PKL file not found: {file_path}. Skipping.")
            continue

        logger.info(f" Loading training graphs from: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                graphs_in_file = pickle.load(f)
            if isinstance(graphs_in_file, list):
                # Basic check if items seem like graphs
                valid_graphs_in_file = [g for g in graphs_in_file if isinstance(g, nx.Graph)]
                num_loaded = len(graphs_in_file)
                num_valid = len(valid_graphs_in_file)
                if num_loaded != num_valid:
                    logger.warning(f"  -> Loaded {num_valid} NetworkX graphs from {file_path} ({num_loaded - num_valid} items were not graphs).")
                else:
                    logger.info(f"  -> Successfully loaded {num_valid} graphs from {file_path}.")
                all_train_graphs.extend(valid_graphs_in_file)
            else:
                logger.warning(f" Expected a list of graphs in {file_path}, got {type(graphs_in_file)}. Skipping file.")
        except (pickle.UnpicklingError, EOFError, MemoryError) as e:
            logger.error(f"Error reading pickle file {file_path}: {e}. Skipping.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading {file_path}: {e}")

    if not all_train_graphs:
        logger.error("Failed to load any valid training graphs from the specified PKL files.")
        return None

    logger.info(f"Finished loading training graphs. Total loaded: {len(all_train_graphs)}")
    return all_train_graphs
# +++ END NEW FUNCTION +++


# --- MODIFIED: Main Evaluation Logic ---
def run_standalone_evaluation(args):
    """Runs the evaluation including Validity, Uniqueness, and Novelty."""
    logger.info(f"Loading generated AIGs from: {args.input_pickle_file}")
    try:
        with open(args.input_pickle_file, 'rb') as f: generated_graphs = pickle.load(f)
        if not isinstance(generated_graphs, list):
             logger.error(f"Pickle file {args.input_pickle_file} does not contain a list."); return
        # Optional: Filter generated graphs if needed
        # generated_graphs = [g for g in generated_graphs if isinstance(g, nx.Graph)]
        logger.info(f"Loaded {len(generated_graphs)} generated graphs.")
    except FileNotFoundError: logger.error(f"Input pickle file not found: {args.input_pickle_file}"); return
    except Exception as e: logger.error(f"Error loading generated graphs pickle file: {e}"); return

    if not generated_graphs: logger.warning("No generated graphs found in the pickle file. Exiting."); return

    # --- MODIFIED: Load Training Data from PKL ---
    train_graphs = None
    if args.train_pkl_files: # Check the new argument
        train_graphs = load_training_graphs_from_pkl(args.train_pkl_files) # Call the new function
        if train_graphs is None:
             logger.warning(f"Could not load training graphs from specified PKL files. Novelty will not be calculated.")
    else:
        logger.info("No training PKL files provided via --train_pkl_files. Novelty will not be calculated.")
    # --- END MODIFIED ---

    num_total = len(generated_graphs)
    valid_graphs = [] # Store the actual valid graph objects
    aggregate_metrics = defaultdict(list)
    aggregate_path_metrics = defaultdict(list)
    failed_constraints_summary = Counter()

    logger.info("Starting evaluation (Pass 1: Validity and Metrics)...")
    for i, graph in enumerate(tqdm(generated_graphs, desc="Evaluating Validity")):
        if not isinstance(graph, nx.DiGraph):
            logger.warning(f"Item {i} is not a NetworkX DiGraph, skipping.")
            failed_constraints_summary["Invalid Graph Object"] += 1; continue

        # --- Run Structural Checks ---
        # This function now expects string types ('NODE_PI', etc.)
        struct_metrics = calculate_structural_aig_metrics(graph)
        # --- End Checks ---

        # Aggregate structural metrics for all graphs
        for key, value in struct_metrics.items():
             # Ensure value is serializable and correct type before appending
             if isinstance(value, (int, float, bool)):
                aggregate_metrics[key].append(float(value))
             elif isinstance(value, list) and key == 'constraints_failed':
                 # Don't aggregate the list itself, handle summary later
                 pass

        # Store valid graphs and calculate path metrics for them
        if struct_metrics.get('is_structurally_valid', False): # Use .get for safety
            valid_graphs.append(graph) # Store the valid graph object
            try: # Add try-except for path metrics
                 path_metrics = count_pi_po_paths(graph)
                 if path_metrics.get('error') is None:
                    for key, value in path_metrics.items():
                        if isinstance(value, (int, float)): aggregate_path_metrics[key].append(value)
                 else: logger.warning(f"Skipping path metrics for valid graph {i} due to error: {path_metrics['error']}")
            except Exception as e:
                 logger.error(f"Error calculating path metrics for valid graph {i}: {e}")
        else:
            # Collect failure reasons only for invalid graphs
            for reason in struct_metrics.get('constraints_failed', ["Unknown Failure"]):
                failed_constraints_summary[reason] += 1

    logger.info("Evaluation (Pass 1) finished.")

    num_valid_structurally = len(valid_graphs)

    # --- Calculate Uniqueness ---
    uniqueness_score, num_unique = calculate_uniqueness(valid_graphs)
    # --- End Uniqueness ---

    # --- Calculate Novelty (if training data loaded) ---
    novelty_score, num_novel = (-1.0, -1) # Default values if not calculated
    if train_graphs is not None:
        novelty_score, num_novel = calculate_novelty(valid_graphs, train_graphs)
    # --- End Novelty ---

    # --- Reporting (Unchanged) ---
    validity_fraction = (num_valid_structurally / num_total) if num_total > 0 else 0.0
    validity_percentage = validity_fraction * 100

    print("\n--- AIG Evaluation Summary (PKL Loaded) ---") # Updated Title
    print(f"Total Generated Graphs Loaded   : {num_total}")
    print(f"Structurally Valid AIGs (V)     : {num_valid_structurally} ({validity_percentage:.2f}%)")
    if num_valid_structurally > 0:
         print(f"Unique Valid AIGs             : {num_unique}")
         print(f"Uniqueness (U) among valid    : {uniqueness_score:.4f} ({uniqueness_score*100:.2f}%)")
         if train_graphs is not None:
             print(f"Novel Valid AIGs vs Train Set : {num_novel}")
             print(f"Novelty (N) among valid       : {novelty_score:.4f} ({novelty_score*100:.2f}%)")
         else:
             print(f"Novelty (N) among valid       : Not calculated (no training PKL files provided)")
    else:
         print(f"Uniqueness (U) among valid    : N/A (0 valid graphs)")
         print(f"Novelty (N) among valid       : N/A (0 valid graphs)")

    print("\n--- Average Structural Metrics (All Generated Graphs) ---")
    for key, values in sorted(aggregate_metrics.items()):
        if key == 'is_structurally_valid': continue
        if not values: continue
        avg_value = np.mean(values)
        if key == 'is_dag': print(f"  - Avg {key:<27}: {avg_value*100:.2f}%")
        else: print(f"  - Avg {key:<27}: {avg_value:.3f}")

    print("\n--- Constraint Violation Summary (Across Invalid Graphs) ---")
    num_invalid_graphs = num_total - num_valid_structurally
    if num_invalid_graphs == 0: print("  No structural violations detected.")
    else:
        sorted_reasons = sorted(failed_constraints_summary.items(), key=lambda item: item[1], reverse=True)
        print(f"  (Violations summarized across {num_invalid_graphs} invalid graphs)")
        total_violation_instances = sum(failed_constraints_summary.values())
        print(f"  (Total violation instances logged: {total_violation_instances})")
        for reason, count in sorted_reasons:
            reason_percentage_of_invalid = (count / num_invalid_graphs) * 100 if num_invalid_graphs > 0 else 0
            print(f"  - {reason:<45}: {count:<6} graphs ({reason_percentage_of_invalid:.1f}% of invalid)")

    print("\n--- Average Path Connectivity Metrics (Valid Graphs Only) ---")
    num_graphs_for_path_metrics = len(aggregate_path_metrics.get('num_pi', []))
    if num_graphs_for_path_metrics == 0: print("  No structurally valid graphs to calculate path metrics for.")
    else:
        print(f"  (Based on {num_graphs_for_path_metrics} structurally valid graphs)")
        for key, values in sorted(aggregate_path_metrics.items()):
             if key == 'error' or not values: continue
             avg_value = np.mean(values)
             print(f"  - Avg {key:<27}: {avg_value:.3f}")

    print("------------------------------------")


# --- MODIFIED: Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate generated AIGs for Validity, Uniqueness, and Novelty.')
    parser.add_argument('input_pickle_file', type=str,
                        help='Path to the pickle file containing the list of generated NetworkX DiGraphs (e.g., generated_aigs.pkl).')
    # --- MODIFIED ARGUMENT ---
    parser.add_argument('--train_pkl_files',
                        type=str,
                        default=["aigs/real_aigs_part_1_of_6.pkl, aigs/real_aigs_part_2_of_6.pkl,aigs/real_aigs_part_3_of_6.pkl, aigs/real_aigs_part_4_of_6.pkl"], nargs='+', # Accept one or more paths
                        help='(Optional) Path(s) to the training dataset PKL file(s) for Novelty calculation.')
    # --- END MODIFIED ARGUMENT ---

    parsed_args = parser.parse_args()
    run_standalone_evaluation(parsed_args)
