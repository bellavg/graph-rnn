#!/usr/bin/env python3
"""
Improved and debugged version of the AIG generation and evaluation script.
Uses fixed temperature list, better logging, and robust module loading.
"""

import argparse
import os
import glob
import sys
import torch
import networkx as nx
import re
import numpy as np
import pandas as pd
import time
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib.util
import logging
from aig_evaluate import (calculate_paper_validity,
                          calculate_extensive_validity, infer_node_types,
                          calculate_pi_po_connectivity)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to stdout
)
logger = logging.getLogger("aig_evaluator")


# --- Path detection and module import helper ---
# ... (include find_and_import_module function here - same as previous response) ...
def find_and_import_module(module_name, search_dirs=None):
    """
    Dynamically find and import a module by searching in various potential locations.
    Returns the imported module or None if not found.
    """
    if search_dirs is None:
        search_dirs = [".", "./src", "..", "../src", os.path.dirname(os.path.abspath(__file__))]
    for search_dir in search_dirs:
        module_path = os.path.join(search_dir, f"{module_name}.py")
        if os.path.exists(module_path):
            logger.info(f"Found module {module_name} at {module_path}")
            try:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None: continue # Skip if spec cannot be created
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module # Add to sys.modules before execution
                spec.loader.exec_module(module)
                return module
            except ImportError as e:
                 logger.warning(f"ImportError for {module_name} at {module_path}: {e}")
            except Exception as e:
                 logger.warning(f"Failed to load {module_name} from {module_path}: {e}")
    logger.error(f"Could not find or load module {module_name} in search paths: {search_dirs}")
    return None

# --- Dynamically import project modules ---
# ... (include import_project_modules function here - same as previous response) ...
def import_project_modules():
    """Import all required project modules dynamically."""
    modules = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Add script's directory and potential 'src' sibling directory
    potential_paths = [script_dir, os.path.join(os.path.dirname(script_dir), "src")]
    for p in potential_paths:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)
            logger.info(f"Added {p} to sys.path")

    # Define required modules and attempt import
    required = ["model", "utils", "aig_dataset", "aig_generate", "aig_evaluate"]
    fallbacks = {"aig_dataset": None, "aig_evaluate": None} # Placeholders for potential fallbacks

    for mod_name in required:
        module = find_and_import_module(mod_name)
        if module is None:
            if mod_name in fallbacks:
                 logger.warning(f"{mod_name} not found, attempting fallback...")
                 # Add specific fallback logic here if needed (like in previous response)
                 # For now, just mark as None if primary import fails
                 modules[mod_name] = None
            else:
                 logger.error(f"Required module '{mod_name}' could not be imported. Exiting.")
                 return None
        else:
             modules[mod_name] = module

    # Basic check if essential modules loaded
    if modules.get("model") is None or modules.get("utils") is None or \
       modules.get("aig_generate") is None:
        logger.error("One or more critical modules (model, utils, aig_generate) failed to load.")
        return None

    # Handle potential fallbacks for dataset/evaluate if needed (like before)
    if modules.get("aig_dataset") is None:
        logger.warning("aig_dataset module missing, using hardcoded fallback constants.")
        class FallbackAIGDataset: # Simplified Fallback
            EDGE_TYPES = {"NONE": 0, "REGULAR": 1, "INVERTED": 2}
            NUM_EDGE_FEATURES = 3
            @staticmethod
            def _calculate_levels(g):
                if not g or not isinstance(g, nx.DiGraph): return {}, -1
                if not nx.is_directed_acyclic_graph(g): return {n:0 for n in g.nodes()}, -2
                levels = {}
                for node in nx.topological_sort(g):
                    levels[node] = max([-1] + [levels.get(p, -1) for p in g.predecessors(node)]) + 1
                return levels, max(levels.values()) if levels else -1
        modules["aig_dataset"] = FallbackAIGDataset

    if modules.get("aig_evaluate") is None:
         logger.warning("aig_evaluate module missing. Validity checks will be basic.")
         class FallbackAIGEvaluate: # Simplified Fallback
             @staticmethod
             def infer_node_types(g): return {n:"UNKNOWN" for n in g.nodes()}
             @staticmethod
             def calculate_paper_validity(g): return 0.0
             @staticmethod
             def calculate_extensive_validity(g, check_connectivity=True): return {"overall_valid": False}
         modules["aig_evaluate"] = FallbackAIGEvaluate


    return modules


# --- Import modules and extract functions ---
imported_modules = import_project_modules()
if imported_modules is None:
    sys.exit(1)

model_module = imported_modules["model"]
utils_module = imported_modules["utils"]
aig_dataset = imported_modules["aig_dataset"]
aig_generate = imported_modules["aig_generate"]
aig_evaluate = imported_modules["aig_evaluate"]

setup_models = utils_module.setup_models
generate_aig = aig_generate.generate_aig
load_model_and_config = aig_generate.load_model_and_config
mlp_edge_gen_aig = aig_generate.mlp_edge_gen_aig
rnn_edge_gen_aig = aig_generate.rnn_edge_gen_aig
_calculate_levels = aig_dataset._calculate_levels
EDGE_TYPES = aig_dataset.EDGE_TYPES
NUM_EDGE_FEATURES = aig_dataset.NUM_EDGE_FEATURES
calculate_paper_validity = aig_evaluate.calculate_paper_validity
calculate_extensive_validity = aig_evaluate.calculate_extensive_validity
infer_node_types = aig_evaluate.infer_node_types


# --- Visualization Function (ensure it's present) ---
# ... (include visualize_aig_structure function here - same as previous response) ...
def visualize_aig_structure(G, output_file='generated_aig_structure.png'):
    """Visualize the generated AIG structure, handling edge types."""
    if G is None or G.number_of_nodes() == 0:
        logger.info(f"Skipping visualization for empty or None graph: {output_file}")
        return

    plt.figure(figsize=(14, 12))
    pos = None
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except ImportError:
        logger.warning("pygraphviz not found. Using spring_layout.")
        pos = nx.spring_layout(G, seed=42)
    except Exception as e:
        logger.warning(f"graphviz layout failed ({e}). Using spring_layout.")
        pos = nx.spring_layout(G, seed=42)

    inferred_types = G.graph.get('_inferred_types_cleaned', infer_node_types(G))
    node_colors = []
    node_labels = {}
    color_map = {"PI": 'lightgreen', "AND": 'lightblue', "PO": 'salmon', "UNKNOWN": 'lightgrey', "INVALID_FANIN": 'orange'}
    for node in G.nodes():
        node_type = inferred_types.get(node, "UNKNOWN")
        node_colors.append(color_map.get(node_type, 'lightgrey'))
        node_labels[node] = f"{node}\n({node_type})"

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.9)
    regular_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == EDGE_TYPES["REGULAR"]]
    inverted_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == EDGE_TYPES["INVERTED"]]
    other_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') not in [EDGE_TYPES["REGULAR"], EDGE_TYPES["INVERTED"]]]
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=1.5, edge_color='black', style='solid', arrowsize=20, node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=inverted_edges, width=1.5, edge_color='red', style='dashed', arrowsize=20, node_size=700)
    if other_edges: logger.warning(f"Found other edge types in {output_file}")

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=1.5, linestyle='solid', label=f'Regular Edge ({EDGE_TYPES["REGULAR"]})'),
        plt.Line2D([0], [0], color='red', lw=1.5, linestyle='dashed', label=f'Inverted Edge ({EDGE_TYPES["INVERTED"]})'),
        plt.scatter([], [], s=100, color='lightgreen', label='Inferred PI'),
        plt.scatter([], [], s=100, color='lightblue', label='Inferred AND'),
        plt.scatter([], [], s=100, color='salmon', label='Inferred PO'),
        plt.scatter([], [], s=100, color='orange', label='Invalid Fan-in'),
        plt.scatter([], [], s=100, color='lightgrey', label='Unknown/Other')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize='small')
    plt.title(f'Generated AIG Structure ({os.path.basename(output_file)})')
    plt.axis('off')
    plt.tight_layout()
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        # logger.info(f"Visualization saved to {output_file}") # Reduce verbosity
    except Exception as e:
        logger.error(f"Error saving visualization {output_file}: {e}")
    plt.close()


# --- get_checkpoint_step function (ensure it's present) ---
# ... (include get_checkpoint_step function here - same as previous response) ...
def get_checkpoint_step(filename):
    """Extracts the step number from a checkpoint filename."""
    match = re.search(r'checkpoint-(\d+)\.pth$', filename)
    if match: return int(match.group(1))
    match_alt = re.search(r'checkpoint_(\d+)\.pth$', filename) # Alternative pattern
    if match_alt: return int(match_alt.group(1))
    return -1


# --- Main Execution Logic ---
# src/all_gen_eval.py

# --- Ensure these imports are present at the top of the file ---
import argparse
import os
import glob
import sys
import torch
import networkx as nx
import re
import numpy as np
import pandas as pd
import time
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib.util
import logging

# Assuming the dynamic import functions and module loading happened before this main function
# Ensure the necessary functions are imported correctly, including the new one:
# from aig_evaluate import (calculate_paper_validity,
#                           calculate_extensive_validity, infer_node_types,
#                           calculate_pi_po_connectivity) # <-- Ensure this is imported
# from aig_generate import generate_aig, load_model_and_config, mlp_edge_gen_aig, rnn_edge_gen_aig
# from aig_dataset import _calculate_levels, EDGE_TYPES, NUM_EDGE_FEATURES
# from utils import setup_models
# from model import EdgeLevelMLP, EdgeLevelRNN, EdgeLevelAttentionRNN, EdgeLevelLSTM, EdgeLevelAttentionLSTM # Add specific model imports if needed by isinstance checks


# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate and Evaluate AIGs from Trained Models")
    # --- Input/Output ---
    parser.add_argument('model_dir', type=str, help='Directory containing model checkpoints (.pth files) or a single checkpoint file.')
    parser.add_argument('--output_csv', type=str, default='aig_evaluation_results.csv', help='Output CSV file name')
    parser.add_argument('--plot_dir', type=str, default='./aig_plots', help='Directory to save visualizations')
    # --- Generation Parameters ---
    parser.add_argument('-n', '--num_graphs', type=int, default=100, help='Number of AIGs to generate per model')
    parser.add_argument('--nodes_target', type=int, default=64, help='Target number of nodes for generated AIGs')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.8, 1.0, 1.2], help='List of temperatures to try for generation')
    parser.add_argument('--max_gen_steps', type=int, default=None, help='Optional max generation steps per graph')
    parser.add_argument('--patience', type=int, default=10, help='Patience for stopping generation if no edges are added')
    # --- Checkpoint Handling ---
    parser.add_argument('--num_checkpoints', type=int, default=None, help='Evaluate only the latest N checkpoints')
    parser.add_argument('--checkpoint_pattern', type=str, default="checkpoint-*.pth", help='Glob pattern to find checkpoint files')
    parser.add_argument('--find_checkpoints', action='store_true', help='Search subdirectories for checkpoints')
    # --- Model/Device Config ---
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--force_max_nodes_train', type=int, default=None, help='Override max_node_count_train if not in config')
    parser.add_argument('--force_max_level_train', type=int, default=None, help='Override max_level_train if not in config')
    # --- Visualization ---
    parser.add_argument('--save_plots', action='store_true', help='Save visualizations of the best valid generated graphs')
    parser.add_argument('--num_plots', type=int, default=5, help='Maximum number of best plots to save per model')
    parser.add_argument('--plot_sort_by', type=str, default='nodes', choices=['nodes', 'level'], help='Sort criteria for best plots')
    # --- Debugging ---
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug output')
    parser.add_argument('--debug_script', type=str, default=__file__, help=argparse.SUPPRESS) # Keep if needed

    args = parser.parse_args()

    # --- Configure Logger Level ---
    if args.debug: logger.setLevel(logging.DEBUG)
    else: logger.setLevel(logging.INFO)
    logger.debug("Debug logging enabled")

    # --- Device Setup ---
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # --- Plot Directory ---
    if args.save_plots:
        os.makedirs(args.plot_dir, exist_ok=True)
        logger.info(f"Plots will be saved to {args.plot_dir}")

    # --- Find Checkpoints ---
    checkpoint_paths_to_evaluate = []
    if os.path.isfile(args.model_dir) and args.model_dir.endswith('.pth'):
        checkpoint_paths_to_evaluate = [args.model_dir]
        logger.info(f"Evaluating single checkpoint file: {args.model_dir}")
    elif os.path.isdir(args.model_dir):
        search_pattern = os.path.join(args.model_dir, args.checkpoint_pattern)
        all_checkpoint_paths = sorted(glob.glob(search_pattern, recursive=args.find_checkpoints))

        if not all_checkpoint_paths and args.find_checkpoints:
            logger.info(f"No checkpoints matching '{args.checkpoint_pattern}' in {args.model_dir}. Searching subdirs recursively...")
            all_checkpoint_paths = sorted(glob.glob(os.path.join(args.model_dir, '**', args.checkpoint_pattern), recursive=True))

        if not all_checkpoint_paths:
            logger.error(f"Error: No checkpoint files matching '{args.checkpoint_pattern}' found in {args.model_dir} (searched recursively: {args.find_checkpoints})")
            return 1

        # Filter by num_checkpoints if specified
        if args.num_checkpoints is not None and args.num_checkpoints > 0:
            logger.info(f"Found {len(all_checkpoint_paths)} total checkpoints. Selecting latest {args.num_checkpoints}.")
            all_checkpoint_paths.sort(key=get_checkpoint_step, reverse=True) # Make sure get_checkpoint_step is defined
            checkpoint_paths_to_evaluate = all_checkpoint_paths[:args.num_checkpoints]
        else:
            checkpoint_paths_to_evaluate = all_checkpoint_paths
    else:
        logger.error(f"Error: model_dir '{args.model_dir}' is not a valid file or directory.")
        return 1

    if not checkpoint_paths_to_evaluate:
        logger.error("No checkpoints selected for evaluation.")
        return 1

    logger.info(f"Selected {len(checkpoint_paths_to_evaluate)} models/checkpoints to evaluate.")

    results_list = []

    # --- Model Loop ---
    for model_idx, model_path in enumerate(checkpoint_paths_to_evaluate):
        model_basename = os.path.basename(model_path).replace(".pth", "")
        logger.info(f"\n--- Evaluating Model [{model_idx + 1}/{len(checkpoint_paths_to_evaluate)}]: {model_basename} ---")
        start_time_model = time.time()

        try:
            # --- Load Model and Config ---
            model_state, loaded_config = load_model_and_config(model_path, device)
            logger.info(f"  Config loaded successfully from checkpoint.")
            if args.debug:
                logger.debug(f"    Config model: {loaded_config.get('model', {})}")
                logger.debug(f"    Config data: {loaded_config.get('data', {})}")

            # --- Determine Training Params (max_n, max_l) ---
            max_n_train = loaded_config.get('data', {}).get('max_node_count_train', None)
            max_l_train = loaded_config.get('data', {}).get('max_level_train', None)

            config_source_msg = "from config"
            if max_n_train is None and args.force_max_nodes_train is not None:
                max_n_train = args.force_max_nodes_train
                config_source_msg += ", max_nodes overridden"
                logger.info(f"  Using override max_nodes_train: {max_n_train}")
            if max_l_train is None and args.force_max_level_train is not None:
                max_l_train = args.force_max_level_train
                config_source_msg += ", max_level overridden"
                logger.info(f"  Using override max_level_train: {max_l_train}")

            if max_n_train is None or max_l_train is None:
                raise ValueError(f"Could not determine max_node_count_train or max_level_train ({config_source_msg})")

            logger.info(f"  Using training params: max_nodes={max_n_train}, max_level={max_l_train} ({config_source_msg})")

            # --- Setup Models ---
            # Make sure 'model_module' refers to your imported model module
            node_model_gen, edge_model_gen, _ = setup_models(loaded_config, device, max_n_train, max_l_train)
            logger.info(f"  Node model type: {type(node_model_gen).__name__}")
            logger.info(f"  Edge model type: {type(edge_model_gen).__name__}")

            # --- Load Weights ---
            node_model_gen.load_state_dict(model_state['node_model'])
            edge_model_gen.load_state_dict(model_state['edge_model'])
            logger.info(f"  Model weights loaded successfully.")

            # --- Select Edge Function ---
            edge_func = None
            # Ensure model_module is defined and refers to your imported model definitions
            if isinstance(edge_model_gen, model_module.EdgeLevelMLP):
                edge_func = mlp_edge_gen_aig
                logger.info("  Using MLP edge generation function")
            elif any(base_name in type(edge_model_gen).__name__.lower() for base_name in ['rnn', 'lstm', 'gru']):
                 edge_func = rnn_edge_gen_aig
                 logger.info("  Using RNN/LSTM edge generation function")
            else:
                 raise ValueError(f"Could not determine edge function for model type '{type(edge_model_gen).__name__}'")


            effective_m_gen = max(1, max_n_train - 1)
            logger.info(f"  Effective M for generation: {effective_m_gen}")

            # --- Generation and Evaluation Loop for this Model ---
            all_graphs_data = []
            temps_to_try = args.temperatures # Use the list from args
            logger.info(
                f"  Generating {args.num_graphs} graphs (target N={args.nodes_target}, temps={temps_to_try}, patience={args.patience})...")

            with tqdm(total=args.num_graphs, desc=f"  Generating {model_basename}", leave=False, disable=None) as pbar:
                for i in range(args.num_graphs):
                    # Generate graph using generate_aig (which tries multiple temperatures)
                    gen_graph, gen_max_level = generate_aig( # generate_aig returns the best graph found across temps
                        num_nodes_target=args.nodes_target,
                        node_model=node_model_gen,
                        edge_model=edge_model_gen,
                        effective_m=effective_m_gen,
                        max_level_model=max_l_train,
                        edge_gen_fn=edge_func,
                        device=device,
                        temperatures=temps_to_try, # Pass the list of temperatures
                        max_steps=args.max_gen_steps,
                        eos_patience=args.patience,
                        debug=args.debug
                    )

                    # --- Analyze BEFORE cleaning isolates ---
                    initial_analysis_results = {}
                    paper_val_initial = float('nan') # Default to NaN
                    is_structurally_valid_initial = False
                    if gen_graph is not None and gen_graph.number_of_nodes() > 0:
                        try:
                            # Perform structural checks first on the raw generated graph
                            is_dag_initial = nx.is_directed_acyclic_graph(gen_graph)
                            inferred_types_initial = infer_node_types(gen_graph)
                            # Run extensive check focusing on structure (connectivity maybe later)
                            extensive_val_details_initial = calculate_extensive_validity(gen_graph, check_connectivity=False) # Check structure first
                            paper_val_initial = calculate_paper_validity(gen_graph) # Paper validity on raw graph
                            # Structural validity depends on DAG and the structure-related checks in extensive_validity
                            is_structurally_valid_initial = is_dag_initial and \
                                                            extensive_val_details_initial.get("no_self_loops", False) and \
                                                            extensive_val_details_initial.get("has_pis", False) and \
                                                            extensive_val_details_initial.get("has_ands", False) and \
                                                            extensive_val_details_initial.get("has_pos", False) and \
                                                            extensive_val_details_initial.get("correct_fanin", False)


                            # Store results
                            initial_analysis_results['is_structurally_valid'] = is_structurally_valid_initial
                            initial_analysis_results['paper_validity_initial'] = paper_val_initial
                            initial_analysis_results['extensive_details_initial'] = extensive_val_details_initial # Store details if needed

                        except Exception as e_analyze_initial:
                            logger.error(f"Error during initial analysis for graph {i}: {e_analyze_initial}")
                            initial_analysis_results['is_structurally_valid'] = False
                            initial_analysis_results['paper_validity_initial'] = float('nan')


                    # --- Clean: Remove Isolated Nodes ---
                    g_cleaned = gen_graph
                    num_nodes_before_cleaning = 0
                    if gen_graph is not None: # Check if graph generation succeeded
                        num_nodes_before_cleaning = gen_graph.number_of_nodes()
                        isolates = list(nx.isolates(gen_graph))
                        if isolates:
                            logger.debug(f"Graph {i}: Removing {len(isolates)} isolated nodes: {isolates}")
                            nodes_to_keep = list(set(gen_graph.nodes()) - set(isolates))
                            g_cleaned = gen_graph.subgraph(nodes_to_keep).copy() if nodes_to_keep else nx.DiGraph()
                        else:
                             g_cleaned = gen_graph # No isolates to remove, use original (still copy if mutable operations planned)
                             if g_cleaned is not None: g_cleaned = g_cleaned.copy() # Ensure copy if further modifications happen


                    # --- Analyze Cleaned Graph ---
                    cleaned_node_count = g_cleaned.number_of_nodes() if g_cleaned else 0
                    cleaned_edge_count = g_cleaned.number_of_edges() if g_cleaned else 0
                    final_max_level_cleaned, paper_val_cleaned, is_extensively_valid_cleaned = -1, float('nan'), False
                    pi_po_conn_results = {'percent_pis_to_pos': float('nan'), 'percent_pos_from_pis': float('nan')} # Default NaN

                    if cleaned_node_count > 0:
                        try:
                            is_dag_cleaned = nx.is_directed_acyclic_graph(g_cleaned)
                            if is_dag_cleaned:
                                _, final_max_level_cleaned = _calculate_levels(g_cleaned)
                            else:
                                final_max_level_cleaned = -2 # Not a DAG after cleaning

                            inferred_types_cleaned = infer_node_types(g_cleaned)
                            # Store inferred types on the graph object itself if needed for visualization
                            g_cleaned.graph['_inferred_types_cleaned'] = inferred_types_cleaned
                            paper_val_cleaned = calculate_paper_validity(g_cleaned)

                            # Run extensive validity on cleaned graph, NOW check connectivity
                            extensive_val_details_cleaned = calculate_extensive_validity(g_cleaned, check_connectivity=True)
                            # Overall validity now includes DAG check on cleaned graph and connectivity
                            is_extensively_valid_cleaned = is_dag_cleaned and extensive_val_details_cleaned.get("overall_valid", False)


                            # Calculate PI-PO connectivity on the cleaned graph
                            pi_po_conn_results = calculate_pi_po_connectivity(g_cleaned)

                        except Exception as e_analyze_cleaned:
                            logger.error(f"Error during cleaned graph analysis for graph {i}: {e_analyze_cleaned}")
                            final_max_level_cleaned = -3 # Analysis error
                            paper_val_cleaned = float('nan')
                            is_extensively_valid_cleaned = False
                            # pi_po_conn_results remains NaN default

                    # Store all desired metrics
                    all_graphs_data.append({
                        "graph": g_cleaned, # Store the potentially cleaned graph object for plotting
                        "analysis_index": i,
                        "is_initially_structurally_valid": initial_analysis_results.get('is_structurally_valid', False),
                        "is_extensively_valid_cleaned": is_extensively_valid_cleaned,
                        "paper_validity_initial": initial_analysis_results.get('paper_validity_initial', float('nan')),
                        "paper_validity_cleaned": paper_val_cleaned,
                        "node_count_initial": num_nodes_before_cleaning,
                        "node_count_cleaned": cleaned_node_count,
                        "edge_count_cleaned": cleaned_edge_count,
                        "max_level_cleaned": final_max_level_cleaned,
                        "percent_pis_to_pos": pi_po_conn_results.get('percent_pis_to_pos', float('nan')),
                        "percent_pos_from_pis": pi_po_conn_results.get('percent_pos_from_pis', float('nan')),
                    })
                    pbar.update(1) # End of loop for one graph generation attempt

            # --- Aggregate and Report Results for Model ---
            num_analyzed = len(all_graphs_data)
            logger.info(f"  Finished generation for {model_basename}. Analyzing {num_analyzed} results...")

            if num_analyzed > 0:
                # Calculate averages, handling potential NaNs using nanmean
                avg_nodes_init = np.nanmean([d["node_count_initial"] for d in all_graphs_data])
                avg_nodes_cleaned = np.nanmean([d["node_count_cleaned"] for d in all_graphs_data])
                avg_edges_cleaned = np.nanmean([d["edge_count_cleaned"] for d in all_graphs_data])
                valid_levels = [d["max_level_cleaned"] for d in all_graphs_data if d["max_level_cleaned"] >= 0]
                avg_level_cleaned = np.nanmean(valid_levels) if valid_levels else float('nan') # Use nanmean here too
                max_level_cleaned = np.nanmax(valid_levels) if valid_levels else -1 # Use nanmax

                percent_init_struct_valid = np.nanmean([d["is_initially_structurally_valid"] for d in all_graphs_data]) * 100.0
                percent_cleaned_ext_valid = np.nanmean([d["is_extensively_valid_cleaned"] for d in all_graphs_data]) * 100.0
                avg_paper_validity_init = np.nanmean([d["paper_validity_initial"] for d in all_graphs_data])
                avg_paper_validity_cleaned = np.nanmean([d["paper_validity_cleaned"] for d in all_graphs_data])

                avg_pis_to_pos = np.nanmean([d["percent_pis_to_pos"] for d in all_graphs_data])
                avg_pos_from_pis = np.nanmean([d["percent_pos_from_pis"] for d in all_graphs_data])

                # Store aggregated results
                results_list.append({
                    "model": model_basename, "num_generated": args.num_graphs, "num_analyzed": num_analyzed,
                    "target_nodes": args.nodes_target,
                    "avg_nodes_initial": f"{avg_nodes_init:.2f}",
                    "avg_nodes_cleaned": f"{avg_nodes_cleaned:.2f}",
                    "avg_edges_cleaned": f"{avg_edges_cleaned:.2f}",
                    "percent_init_struct_valid": f"{percent_init_struct_valid:.2f}",
                    "percent_cleaned_ext_valid": f"{percent_cleaned_ext_valid:.2f}",
                    "avg_paper_validity_init": f"{avg_paper_validity_init:.4f}",
                    "avg_paper_validity_cleaned": f"{avg_paper_validity_cleaned:.4f}",
                    "avg_percent_pis_to_pos": f"{avg_pis_to_pos:.2f}",
                    "avg_percent_pos_from_pis": f"{avg_pos_from_pis:.2f}",
                    "avg_max_level_cleaned": f"{avg_level_cleaned:.2f}",
                    "max_max_level_cleaned": f"{max_level_cleaned:.0f}",
                    "temperatures": ','.join(map(str, args.temperatures)),
                    "patience": args.patience,
                })

                # Log aggregated results
                logger.info(f"    Avg Nodes (Initial): {avg_nodes_init:.2f}")
                logger.info(f"    Avg Nodes (Cleaned): {avg_nodes_cleaned:.2f}")
                logger.info(f"    Avg Edges (Cleaned): {avg_edges_cleaned:.2f}")
                logger.info(f"    Initial Struct Valid: {percent_init_struct_valid:.2f}%")
                logger.info(f"    Cleaned Extens Valid: {percent_cleaned_ext_valid:.2f}%")
                logger.info(f"    Avg Paper Valid (Init): {avg_paper_validity_init:.4f}")
                logger.info(f"    Avg Paper Valid (Clean): {avg_paper_validity_cleaned:.4f}")
                logger.info(f"    Avg % PIs->POs (Clean): {avg_pis_to_pos:.2f}%")
                logger.info(f"    Avg % POs<-PIs (Clean): {avg_pos_from_pis:.2f}%")
                logger.info(f"    Avg Max Level (Clean, DAGs): {avg_level_cleaned:.2f}")
                logger.info(f"    Max Max Level (Clean, DAGs): {max_level_cleaned}")


                # --- Plotting (operates on cleaned valid graphs) ---
                if args.save_plots:
                    # Select graphs that passed the *final* extensive validity check on the cleaned graph
                    valid_graphs_data = [d for d in all_graphs_data if d["is_extensively_valid_cleaned"]]
                    logger.info(f"  Found {len(valid_graphs_data)} extensively valid cleaned graphs for plotting.")
                    if valid_graphs_data:
                        # Sort based on cleaned metrics
                        sort_key = "node_count_cleaned" if args.plot_sort_by == "nodes" else "max_level_cleaned"
                        # Handle potential NaN or None values in sorting key
                        valid_graphs_data_sorted = sorted(
                            valid_graphs_data,
                            key=lambda x: x.get(sort_key, 0) if pd.notna(x.get(sort_key)) else 0, # Default to 0 if key missing/NaN
                            reverse=True
                        )
                        num_plots_to_save = min(len(valid_graphs_data_sorted), args.num_plots)
                        logger.info(f"  Saving plots for top {num_plots_to_save} valid cleaned graphs (sorted by {args.plot_sort_by})...")
                        for rank, graph_data in enumerate(valid_graphs_data_sorted[:num_plots_to_save]):
                            # Use cleaned graph for plotting
                            plot_filename = os.path.join(args.plot_dir, f"{model_basename}_valid_cleaned_{args.plot_sort_by}_rank{rank + 1}_idx{graph_data['analysis_index']}.png")
                            # Make sure visualize_aig_structure function is defined and available
                            visualize_aig_structure(graph_data["graph"], plot_filename) # Pass the graph object stored in dict
                        logger.info(f"  Finished saving plots.")
                    else:
                        logger.info("  No extensively valid cleaned graphs found to plot.")
            else:
                logger.warning("    No graphs were successfully generated or analyzed for this model.")
                # Add empty/error result row with new columns
                results_list.append({
                    "model": model_basename, "num_generated": args.num_graphs, "num_analyzed": 0,
                    **{k:"N/A" for k in [
                        "target_nodes", "avg_nodes_initial", "avg_nodes_cleaned", "avg_edges_cleaned",
                        "percent_init_struct_valid", "percent_cleaned_ext_valid",
                        "avg_paper_validity_init", "avg_paper_validity_cleaned",
                        "avg_percent_pis_to_pos", "avg_percent_pos_from_pis",
                        "avg_max_level_cleaned", "max_max_level_cleaned",
                        "temperatures", "patience"]}
                    })

        except Exception as model_e:
            logger.error(f"  FATAL Error processing model {model_path}: {model_e}")
            logger.error(traceback.format_exc())
            # Add error row with new columns marked N/A
            results_list.append({
                "model": model_basename, "num_generated": 0, "num_analyzed": 0, "error": str(model_e),
                 **{k:"N/A" for k in [
                        "target_nodes", "avg_nodes_initial", "avg_nodes_cleaned", "avg_edges_cleaned",
                        "percent_init_struct_valid", "percent_cleaned_ext_valid",
                        "avg_paper_validity_init", "avg_paper_validity_cleaned",
                        "avg_percent_pis_to_pos", "avg_percent_pos_from_pis",
                        "avg_max_level_cleaned", "max_max_level_cleaned",
                        "temperatures", "patience"]}
                })
            continue # Skip to the next model

        finally:
            elapsed_time_model = time.time() - start_time_model
            logger.info(f"  Model evaluation took {elapsed_time_model:.2f} seconds.")
            # --- End of loop for one model ---

    # --- Save Final Results ---
    if results_list:
        results_df = pd.DataFrame(results_list)
        # Define desired column order including new metrics
        cols_order = [
            "model", "num_generated", "num_analyzed", "target_nodes", "temperatures", "patience",
            "avg_nodes_initial", "avg_nodes_cleaned", "avg_edges_cleaned",
            "percent_init_struct_valid", "percent_cleaned_ext_valid",
            "avg_paper_validity_init", "avg_paper_validity_cleaned",
            "avg_percent_pis_to_pos", "avg_percent_pos_from_pis",
            "avg_max_level_cleaned", "max_max_level_cleaned", "error"
        ]
        final_cols = [col for col in cols_order if col in results_df.columns] # Ensure only existing columns are selected
        results_df = results_df[final_cols]
        try:
            results_df.to_csv(args.output_csv, index=False)
            logger.info(f"\nEvaluation results saved to {args.output_csv}")
        except Exception as e:
            logger.error(f"\nError saving results to CSV: {e}")
            logger.error("\nResults DataFrame:")
            # Use pandas to_string for potentially better formatting if logger truncates
            logger.error(results_df.to_string())
    else:
        logger.error("\nNo models were successfully evaluated.")

    return 0

if __name__ == "__main__":
    # Optional: Increase recursion depth if needed for deep graph processing
    # sys.setrecursionlimit(5000)
    try:
         exit_code = main()
         sys.exit(exit_code)
    except Exception as main_e:
         logger.critical(f"Unhandled exception in main: {main_e}")
         logger.critical(traceback.format_exc())
         sys.exit(1)