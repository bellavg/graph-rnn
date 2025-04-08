#!/usr/bin/env python3
"""
Improved and debugged version of the AIG generation and evaluation script.
Adds better error handling, path detection, and configuration management.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("aig_evaluator")


# --- Path detection and module import helper ---
def find_and_import_module(module_name, search_dirs=None):
    """
    Dynamically find and import a module by searching in various potential locations.
    Returns the imported module or None if not found.
    """
    if search_dirs is None:
        # Try various potential locations:
        search_dirs = [
            ".",  # Current directory
            "./src",  # Common src directory
            "..",  # Parent directory
            "../src",  # Parent's src directory
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Script's parent dir
        ]

    for search_dir in search_dirs:
        module_path = os.path.join(search_dir, f"{module_name}.py")
        if os.path.exists(module_path):
            logger.info(f"Found module {module_name} at {module_path}")
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    logger.warning(f"Could not find module {module_name} in search paths")
    return None


# --- Dynamically import project modules ---
def import_project_modules():
    """Import all required project modules dynamically."""
    modules = {}

    # Try to add the current script's directory to sys.path to help imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        logger.info(f"Added {script_dir} to sys.path")

    # Also add potential src directory
    src_dir = os.path.join(os.path.dirname(script_dir), "src")
    if os.path.exists(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        logger.info(f"Added {src_dir} to sys.path")

    # Import model module
    try:
        modules["model"] = find_and_import_module("model")
        if modules["model"] is None:
            raise ImportError("model module not found")
    except ImportError as e:
        logger.error(f"Error importing model module: {e}")
        logger.error("This module is required.")
        return None

    # Import utils module (for setup_models)
    try:
        modules["utils"] = find_and_import_module("utils")
        if modules["utils"] is None:
            raise ImportError("utils module not found")
    except ImportError as e:
        logger.error(f"Error importing utils module: {e}")
        logger.error("This module is required.")
        return None

    # Import aig_dataset module
    try:
        modules["aig_dataset"] = find_and_import_module("aig_dataset")
        if modules["aig_dataset"] is None:
            # Define fallbacks for constants if the module can't be found
            logger.warning("aig_dataset module not found. Using fallback constants.")

            class FallbackAIGDataset:
                NODE_TYPES = {"PI": 1, "AND": 2, "PO": 3, "ZERO": 0, "UNKNOWN": -1}
                EDGE_TYPES = {"NONE": 0, "REGULAR": 1, "INVERTED": 2}
                NUM_EDGE_FEATURES = 3

                @staticmethod
                def _calculate_levels(g):
                    """Fallback implementation for calculate_levels."""
                    if not g or g.number_of_nodes() == 0:
                        return {}, -1

                    levels = {}
                    for node in nx.topological_sort(g):
                        pred_levels = [levels.get(pred, 0) for pred in g.predecessors(node)]
                        levels[node] = max(pred_levels + [-1]) + 1

                    max_level = max(levels.values()) if levels else -1
                    return levels, max_level

            modules["aig_dataset"] = FallbackAIGDataset()
    except ImportError as e:
        logger.error(f"Error setting up aig_dataset fallback: {e}")
        return None

    # Import generate module
    try:
        modules["aig_generate"] = find_and_import_module("aig_generate")
        if modules["aig_generate"] is None:
            logger.error("aig_generate module not found. Cannot proceed without generation functions.")
            return None
    except ImportError as e:
        logger.error(f"Error importing aig_generate module: {e}")
        logger.error("This module is required.")
        return None

    # Import evaluate module
    try:
        modules["aig_evaluate"] = find_and_import_module("aig_evaluate")
        if modules["aig_evaluate"] is None:
            # Define fallbacks for evaluate functions
            logger.warning("aig_evaluate module not found. Using fallback evaluation functions.")
            from collections import Counter

            class FallbackAIGEvaluate:
                @staticmethod
                def infer_node_types(g):
                    """Fallback implementation for infer_node_types."""
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
                        elif out_deg == 0 and in_deg >= 1:
                            types[n] = "PO"
                        elif in_deg == 1:
                            types[n] = "UNKNOWN"
                        else:
                            types[n] = "INVALID_FANIN"
                    return types

                @staticmethod
                def calculate_paper_validity(g):
                    """Fallback implementation for paper validity."""
                    if not isinstance(g, nx.DiGraph) or g.number_of_nodes() == 0:
                        return 0.0

                    inferred_types = FallbackAIGEvaluate.infer_node_types(g)
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
                            if in_deg == 1:
                                valid_gates += 1

                    if total_relevant_gates == 0:
                        return 0.0
                    else:
                        return float(valid_gates) / total_relevant_gates

                @staticmethod
                def calculate_extensive_validity(g, check_connectivity=True):
                    """Fallback implementation for extensive validity check."""
                    default_results = {
                        "is_dag": False,
                        "has_pis": False,
                        "has_ands": False,
                        "has_pos": False,
                        "correct_fanin": True,
                        "no_self_loops": False,
                        "all_reachable_from_pi": not check_connectivity,
                        "overall_valid": False,
                        "node_counts": {"PI": 0, "AND": 0, "PO": 0, "UNKNOWN": 0, "INVALID_FANIN": 0},
                        "error_nodes": {"fanin": [], "unreachable": [], "structure": []}
                    }

                    if not isinstance(g, nx.DiGraph) or g.number_of_nodes() == 0:
                        default_results["error_nodes"]["structure"].append("Empty or invalid graph object")
                        return default_results

                    # 1. Self-loops Check
                    num_self_loops = sum(1 for u, v in g.edges() if u == v)
                    default_results["no_self_loops"] = (num_self_loops == 0)
                    if num_self_loops > 0:
                        default_results["error_nodes"]["structure"].append(f"Found {num_self_loops} self-loops")
                        return default_results

                    # 2. DAG Check
                    try:
                        default_results["is_dag"] = nx.is_directed_acyclic_graph(g)
                    except Exception as e:
                        default_results["error_nodes"]["structure"].append(f"DAG check failed: {e}")
                        return default_results

                    if not default_results["is_dag"]:
                        default_results["error_nodes"]["structure"].append("Not a DAG")
                        return default_results

                    # 3. Infer types and check existence
                    inferred_types = FallbackAIGEvaluate.infer_node_types(g)
                    pi_nodes = set()
                    default_results["node_counts"] = Counter(inferred_types.values())

                    default_results["has_pis"] = default_results["node_counts"].get("PI", 0) > 0
                    default_results["has_ands"] = default_results["node_counts"].get("AND", 0) > 0
                    default_results["has_pos"] = default_results["node_counts"].get("PO", 0) > 0

                    for node, node_type in inferred_types.items():
                        if node_type == "PI":
                            pi_nodes.add(node)
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

                        if not correct_fanin_for_type:
                            default_results["correct_fanin"] = False
                            if error_msg:
                                default_results["error_nodes"]["fanin"].append(error_msg)

                    # 4. Connectivity Check
                    if check_connectivity and default_results["has_pis"]:
                        all_nodes_reachable = True
                        non_pi_nodes = set(g.nodes()) - pi_nodes
                        if non_pi_nodes:
                            reachable_nodes = set(pi_nodes)
                            for pi_node in pi_nodes:
                                try:
                                    reachable_nodes.update(nx.descendants(g, pi_node))
                                except Exception:
                                    all_nodes_reachable = False
                                    break

                            if all_nodes_reachable:
                                unreached = non_pi_nodes - reachable_nodes
                                if unreached:
                                    all_nodes_reachable = False
                                    default_results["error_nodes"]["unreachable"] = sorted(list(unreached))

                        default_results["all_reachable_from_pi"] = all_nodes_reachable

                    # 5. Overall validity
                    default_results["overall_valid"] = (
                            default_results["is_dag"] and
                            default_results["no_self_loops"] and
                            default_results["has_pis"] and
                            default_results["has_ands"] and
                            default_results["has_pos"] and
                            default_results["correct_fanin"] and
                            default_results["all_reachable_from_pi"]
                    )

                    return default_results

            modules["aig_evaluate"] = FallbackAIGEvaluate()
    except ImportError as e:
        logger.error(f"Error setting up aig_evaluate fallback: {e}")
        return None

    return modules


# Import modules
imported_modules = import_project_modules()
if imported_modules is None:
    logger.error("Failed to import required modules. Exiting.")
    sys.exit(1)

# Extract imported modules and functions
model_module = imported_modules["model"]
utils_module = imported_modules["utils"]
aig_dataset = imported_modules["aig_dataset"]
aig_generate = imported_modules["aig_generate"]
aig_evaluate = imported_modules["aig_evaluate"]

# Check if these are actual modules or fallback classes
is_fallback_dataset = not hasattr(aig_dataset, "__file__")
is_fallback_evaluate = not hasattr(aig_evaluate, "__file__")

# Extract necessary functions and constants
setup_models = utils_module.setup_models
generate_aig = aig_generate.generate_aig
load_model_and_config = aig_generate.load_model_and_config
mlp_edge_gen_aig = aig_generate.mlp_edge_gen_aig
rnn_edge_gen_aig = aig_generate.rnn_edge_gen_aig

# Dataset constants
if is_fallback_dataset:
    # Use constants from fallback class
    NODE_TYPES = aig_dataset.NODE_TYPES
    EDGE_TYPES = aig_dataset.EDGE_TYPES
    NUM_EDGE_FEATURES = aig_dataset.NUM_EDGE_FEATURES
    _calculate_levels = aig_dataset._calculate_levels
else:
    # Use constants from imported module
    NODE_TYPES = aig_dataset.NODE_TYPES
    EDGE_TYPES = aig_dataset.EDGE_TYPES
    NUM_EDGE_FEATURES = aig_dataset.NUM_EDGE_FEATURES
    _calculate_levels = aig_dataset._calculate_levels

# Evaluate functions
if is_fallback_evaluate:
    # Use methods from fallback class
    calculate_paper_validity = aig_evaluate.calculate_paper_validity
    calculate_extensive_validity = aig_evaluate.calculate_extensive_validity
    infer_node_types = aig_evaluate.infer_node_types
else:
    # Use functions from imported module
    calculate_paper_validity = aig_evaluate.calculate_paper_validity
    calculate_extensive_validity = aig_evaluate.calculate_extensive_validity
    infer_node_types = aig_evaluate.infer_node_types


def visualize_aig_structure(G, output_file='generated_aig_structure.png'):
    """Visualize the generated AIG structure, handling edge types."""
    if G is None or G.number_of_nodes() == 0:
        logger.info(f"Skipping visualization for empty or None graph: {output_file}")
        return

    plt.figure(figsize=(14, 12))
    pos = None
    try:
        # Try a layout that works well for DAGs if pygraphviz is available
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except ImportError:
        logger.warning("pygraphviz not found. Using spring_layout (may be less clear).")
        pos = nx.spring_layout(G, seed=42)
    except Exception as e:
        logger.warning(f"graphviz layout failed ({e}). Using spring_layout.")
        pos = nx.spring_layout(G, seed=42)  # Fallback

    # Infer node types for coloring
    node_colors = []
    node_labels = {}
    # Use types stored during analysis if available, otherwise infer
    inferred_types = G.graph.get('_inferred_types_cleaned', infer_node_types(G))

    for node in G.nodes():
        node_type = inferred_types.get(node, "UNKNOWN")
        color_map = {"PI": 'lightgreen', "AND": 'lightblue', "PO": 'salmon', "UNKNOWN": 'lightgrey',
                     "INVALID_FANIN": 'orange'}
        node_colors.append(color_map.get(node_type, 'lightgrey'))
        node_labels[node] = f"{node}\n({node_type})"  # Label with node ID and type

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.9)

    # Draw edges with styles based on 'type' attribute
    regular_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == EDGE_TYPES["REGULAR"]]
    inverted_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == EDGE_TYPES["INVERTED"]]
    other_edges = [(u, v) for u, v, d in G.edges(data=True) if
                   d.get('type') not in [EDGE_TYPES["REGULAR"], EDGE_TYPES["INVERTED"]]]

    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=1.5, edge_color='black', style='solid', arrowsize=20,
                           node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=inverted_edges, width=1.5, edge_color='red', style='dashed', arrowsize=20,
                           node_size=700)
    if other_edges:
        logger.warning(f"Found edges with unexpected types in {output_file}")
        nx.draw_networkx_edges(G, pos, edgelist=other_edges, width=1.0, edge_color='gray', style='dotted', arrowsize=20,
                               node_size=700)

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    # Add legend for edge types and node colors
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=1.5, linestyle='solid',
                   label=f'Regular Edge (type {EDGE_TYPES["REGULAR"]})'),
        plt.Line2D([0], [0], color='red', lw=1.5, linestyle='dashed',
                   label=f'Inverted Edge (type {EDGE_TYPES["INVERTED"]})'),
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
        logger.info(f"Visualization saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving visualization {output_file}: {e}")
    plt.close()


def get_checkpoint_step(filename):
    """Extracts the step number from a checkpoint filename."""
    match = re.search(r'checkpoint-(\d+)\.pth$', filename)
    if match:
        return int(match.group(1))
    return -1  # Return -1 if pattern doesn't match


def get_model_classes_from_config(config):
    """
    Determine model class types from config dictionary.
    Returns classes for node and edge models.
    """
    node_model_type = config.get('model', {}).get('node_model', 'rnn').lower()
    edge_model_type = config.get('model', {}).get('edge_model', 'mlp').lower()

    # Node model class mapping
    node_model_class = None
    if 'lstm' in node_model_type:
        if 'attention' in node_model_type or 'attn' in node_model_type:
            node_model_class = model_module.NodeLevelAttentionLSTM
        else:
            node_model_class = model_module.NodeLevelLSTM
    elif 'attention' in node_model_type or 'attn' in node_model_type:
        node_model_class = model_module.NodeLevelAttentionRNN
    else:
        # Default to RNN
        node_model_class = model_module.NodeLevelRNN

    # Edge model class mapping
    edge_model_class = None
    if edge_model_type == 'mlp':
        edge_model_class = model_module.EdgeLevelMLP
    elif 'lstm' in edge_model_type:
        if 'attention' in edge_model_type or 'attn' in edge_model_type:
            edge_model_class = model_module.EdgeLevelAttentionLSTM
        else:
            edge_model_class = model_module.EdgeLevelLSTM
    elif 'attention' in edge_model_type or 'attn' in edge_model_type:
        edge_model_class = model_module.EdgeLevelAttentionRNN
    else:
        # Default to RNN for non-MLP
        edge_model_class = model_module.EdgeLevelRNN

    return node_model_class, edge_model_class


def setup_models_robust(config, device, max_n_train, max_l_train):
    """
    A more robust version of setup_models that can handle various model configurations
    and attempts multiple strategies if the default setup fails.
    """
    # Try the standard setup_models first
    try:
        return setup_models(config, device, max_n_train, max_l_train)
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"Standard setup_models failed: {e}")
        logger.info("Attempting alternative model setup approaches...")

    # If standard setup fails, try a more manual approach
    try:
        # Get model classes based on config
        node_model_class, edge_model_class = get_model_classes_from_config(config)

        # Extract model config
        model_config = config.get('model', {})
        node_hidden_size = model_config.get('node_hidden_size', 128)
        edge_hidden_size = model_config.get('edge_hidden_size', 128)
        num_edge_types = NUM_EDGE_FEATURES  # Usually 3 for AIGs (NONE, REGULAR, INVERTED)

        # Create node model
        use_level_embedding = model_config.get('use_level_embedding', True)
        node_model = node_model_class(
            node_hidden_size=node_hidden_size,
            edge_feature_len=num_edge_types,
            max_nodes=max_n_train,
            use_level_embedding=use_level_embedding,
            max_level=max_l_train
        )

        # Create edge model
        if edge_model_class == model_module.EdgeLevelMLP:
            edge_model = edge_model_class(
                node_hidden_size=node_hidden_size,
                edge_hidden_size=edge_hidden_size,
                output_size=num_edge_types,
                max_nodes=max_n_train
            )
        else:
            # For RNN/LSTM edge models
            edge_model = edge_model_class(
                input_size=num_edge_types,
                hidden_size=edge_hidden_size,
                output_size=num_edge_types
            )

        # Move models to device
        node_model = node_model.to(device)
        edge_model = edge_model.to(device)

        return node_model, edge_model, None  # Third element is normally the optimizer

    except Exception as e:
        logger.error(f"Alternative model setup also failed: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Could not set up models: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate and Evaluate AIGs from Trained Models")
    parser.add_argument('model_dir', type=str, help='Directory containing model checkpoints (.pth files)')
    parser.add_argument('-n', '--num_graphs', type=int, default=100, help='Number of AIGs to generate per model')
    parser.add_argument('--nodes_target', type=int, default=50, help='Target number of nodes for generated AIGs')
    parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature for generation (1.0=no change)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--output_csv', type=str, default='aig_evaluation_results.csv', help='Output CSV file name')
    parser.add_argument('--max_gen_steps', type=int, default=None,
                        help='Optional max generation steps per graph (overrides default)')
    parser.add_argument('--patience', type=int, default=10, help='Patience for stopping if no real edges are added')

    # Visualization Args
    parser.add_argument('--num_checkpoints', type=int, default=None,
                        help='Evaluate only the latest N checkpoints (default: evaluate all found)')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save visualizations of the best valid generated graphs')
    parser.add_argument('--plot_dir', type=str, default='./aig_plots', help='Directory to save visualizations')
    parser.add_argument('--num_plots', type=int, default=5, help='Maximum number of best plots to save per model')
    parser.add_argument('--plot_sort_by', type=str, default='nodes', choices=['nodes', 'level'],
                        help='Sort criteria for best plots ("nodes" or "level")')

    # Args potentially needed if info not in config (better to save in config)
    parser.add_argument('--force_max_nodes_train', type=int, default=None,
                        help='Manually override max_node_count_train if not in config')
    parser.add_argument('--force_max_level_train', type=int, default=None,
                        help='Manually override max_level_train if not in config')

    # Debug flags
    parser.add_argument('--debug', action='store_true', help='Enable debug output for diagnostics')
    parser.add_argument('--try_temps', action='store_true', help='Try different temperature values if generation fails')
    parser.add_argument('--checkpoint_pattern', type=str, default=None,
                        help='Regex pattern to filter checkpoint files (e.g., "checkpoint-[0-9]*.pth")')
    parser.add_argument('--find_checkpoints', action='store_true',
                        help='Search subdirectories for checkpoints if none found in model_dir')

    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
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
        logger.info(
            f"Will save up to {args.num_plots} best valid plots per model (sorted by {args.plot_sort_by}) to {args.plot_dir}")

    # --- Find Checkpoints ---
    checkpoint_pattern = args.checkpoint_pattern if args.checkpoint_pattern else 'checkpoint-*.pth'
    all_checkpoint_paths = sorted(glob.glob(os.path.join(args.model_dir, checkpoint_pattern)))

    # If no checkpoints found and --find_checkpoints is enabled, search subdirectories
    if not all_checkpoint_paths and args.find_checkpoints:
        logger.info(f"No checkpoints found in {args.model_dir}. Searching subdirectories...")
        for root, dirs, files in os.walk(args.model_dir):
            for file in files:
                if file.endswith('.pth') and 'checkpoint-' in file:
                    checkpoint_path = os.path.join(root, file)
                    all_checkpoint_paths.append(checkpoint_path)
                    logger.info(f"Found checkpoint: {checkpoint_path}")
        all_checkpoint_paths.sort()  # Sort by path

    if not all_checkpoint_paths:
        logger.error(f"Error: No checkpoint files matching '{checkpoint_pattern}' found in {args.model_dir}")
        return 1

    # --- Filter Checkpoints ---
    if args.num_checkpoints is not None and args.num_checkpoints > 0:
        # Sort by step number (descending) and take the top N
        logger.info(f"Found {len(all_checkpoint_paths)} total checkpoints. Selecting latest {args.num_checkpoints}.")
        all_checkpoint_paths.sort(key=get_checkpoint_step, reverse=True)
        checkpoint_paths_to_evaluate = all_checkpoint_paths[:args.num_checkpoints]
        if not checkpoint_paths_to_evaluate:
            logger.warning("Filtering resulted in zero checkpoints to evaluate.")
    else:
        # Evaluate all found checkpoints if arg not provided or <= 0
        logger.info(f"Found {len(all_checkpoint_paths)} total checkpoints. Evaluating all.")
        checkpoint_paths_to_evaluate = all_checkpoint_paths

    if not checkpoint_paths_to_evaluate:
        logger.error(f"No checkpoints selected for evaluation in {args.model_dir}")
        return 1

    logger.info(f"Selected {len(checkpoint_paths_to_evaluate)} models/checkpoints to evaluate.")

    checkpoint_paths = checkpoint_paths_to_evaluate

    results_list = []

    # --- Model Loop ---
    for model_idx, model_path in enumerate(checkpoint_paths):
        model_basename = os.path.basename(model_path).replace(".pth", "")
        logger.info(f"\n--- Evaluating Model [{model_idx + 1}/{len(checkpoint_paths)}]: {model_basename} ---")
        start_time_model = time.time()

        # --- Load Model and Config ---
        try:
            model_state, loaded_config = load_model_and_config(model_path, device)
            logger.info(f"  Config loaded successfully from checkpoint.")
            if args.debug:
                # Print key parts of the config for debugging
                logger.debug(f"  Config model section: {loaded_config.get('model', {})}")
                logger.debug(f"  Config data section: {loaded_config.get('data', {})}")
        except Exception as e:
            logger.error(f"  Error loading model/config from {model_path}: {e}")
            logger.error(traceback.format_exc())
            continue

        # --- Determine Training Params (max_n, max_l) ---
        max_n_train = loaded_config.get('data', {}).get('max_node_count_train', None)
        max_l_train = loaded_config.get('data', {}).get('max_level_train', None)

        # Apply overrides if provided and config values are missing
        config_source_msg = "from config"
        if max_n_train is None and args.force_max_nodes_train is not None:
            max_n_train = args.force_max_nodes_train
            config_source_msg += ", max_nodes overridden"
            logger.info(f"  Using override max_nodes_train: {max_n_train}")
        if max_l_train is None and args.force_max_level_train is not None:
            max_l_train = args.force_max_level_train
            config_source_msg += ", max_level overridden"
            logger.info(f"  Using override max_level_train: {max_l_train}")

        # Final check if values are determined
        if max_n_train is None or max_l_train is None:
            logger.error(f"  Error: Could not determine max_node_count_train or max_level_train ({config_source_msg}).")
            logger.error(f"  Config 'data': {loaded_config.get('data', {})}")
            logger.error(f"  Override args: nodes={args.force_max_nodes_train}, level={args.force_max_level_train}")
            logger.error("  Skipping this model.")
            continue  # Skip this model if values are still missing

        logger.info(f"  Using training params: max_nodes={max_n_train}, max_level={max_l_train} ({config_source_msg})")

        # --- Setup Models ---
        try:
            # Try the more robust setup
            node_model_gen, edge_model_gen, _ = setup_models_robust(loaded_config, device, max_n_train, max_l_train)
            logger.info(f"  Node model type: {type(node_model_gen).__name__}")
            logger.info(f"  Edge model type: {type(edge_model_gen).__name__}")
        except Exception as e:
            logger.error(f"  Error setting up models using config: {e}")
            logger.error(traceback.format_exc())
            continue

        # --- Load Weights ---
        try:
            # Check model state dict keys before loading
            expected_keys = ['node_model', 'edge_model']
            missing_keys = [key for key in expected_keys if key not in model_state]
            if missing_keys:
                logger.error(f"  Error: Checkpoint missing keys: {missing_keys}")
                # Try alternative keys if standard ones not found
                alt_keys = {
                    'node_model': ['node_net', 'node_rnn', 'node', 'model_node'],
                    'edge_model': ['edge_net', 'edge_rnn', 'edge', 'model_edge']
                }

                for expected_key in missing_keys:
                    for alt_key in alt_keys.get(expected_key, []):
                        if alt_key in model_state:
                            logger.info(f"  Using alternative key '{alt_key}' for '{expected_key}'")
                            model_state[expected_key] = model_state[alt_key]
                            break

            # Check if keys are now present
            if 'node_model' not in model_state or 'edge_model' not in model_state:
                logger.error("  Cannot find appropriate model state dict keys. Skipping.")
                continue

            node_model_gen.load_state_dict(model_state['node_model'])
            edge_model_gen.load_state_dict(model_state['edge_model'])
            logger.info(f"  Model weights loaded successfully.")
        except RuntimeError as e:
            logger.error(f"  Error loading state_dict (model mismatch?): {e}")
            logger.error(traceback.format_exc())
            continue

        # --- Select Edge Function ---
        edge_model_type_config = loaded_config.get('model', {}).get('edge_model', 'mlp').lower()
        edge_func = None

        # More robust edge function selection
        if isinstance(edge_model_gen, model_module.EdgeLevelMLP):
            edge_func = mlp_edge_gen_aig
            logger.info("  Using MLP edge generation function")
        elif any(base_name in type(edge_model_gen).__name__.lower() for base_name in ['rnn', 'lstm', 'gru']):
            edge_func = rnn_edge_gen_aig
            logger.info("  Using RNN/LSTM edge generation function")
        else:
            logger.error(f"  Error: Could not determine edge function for model type '{type(edge_model_gen).__name__}'")
            continue

        # --- Calculate effective_m ---
        # Ensure effective_m is at least 1
        effective_m_gen = max(1, max_n_train - 1)
        logger.info(f"  Effective M for generation (max_n_train - 1): {effective_m_gen}")

        # --- Set up temperature list for testing if enabled ---
        temps_to_try = [args.temp]
        if args.try_temps:
            # If try_temps is enabled and base temp is 1.0, try a range
            if abs(args.temp - 1.0) < 0.01:  # If very close to 1.0
                temps_to_try = [0.5, 0.8, 1.0, 1.2, 1.5]
            else:
                # If a specific temp was given, try that plus a higher and lower
                temps_to_try = [args.temp * 0.5, args.temp, args.temp * 1.5]

            logger.info(f"  Will try these temperatures: {temps_to_try}")

        # --- Generation and Evaluation Loop for this Model ---
        all_graphs_data = []  # Stores dicts with graph data and results
        logger.info(
            f"  Generating {args.num_graphs} graphs (target N={args.nodes_target}, temps={temps_to_try}, patience={args.patience})...")
        graphs_generated_count = 0

        with tqdm(total=args.num_graphs, desc=f"  Generating {model_basename}", leave=False) as pbar:
            for i in range(args.num_graphs):
                gen_graph = None
                gen_max_level = -1

                # Try different temperatures if enabled
                for temp_idx, temperature in enumerate(temps_to_try):
                    try:
                        gen_graph, gen_max_level = generate_aig(
                            num_nodes_target=args.nodes_target,
                            node_model=node_model_gen,
                            edge_model=edge_model_gen,
                            effective_m=effective_m_gen,
                            max_level_model=max_l_train,  # Pass max level model trained with
                            edge_gen_fn=edge_func,
                            device=device,
                            temperature=temperature,
                            max_steps=args.max_gen_steps,
                            eos_patience=args.patience,
                            debug=args.debug
                        )
                        graphs_generated_count += 1

                        # If the graph has nodes and we're trying temps, we can stop
                        if args.try_temps and gen_graph.number_of_nodes() > 0:
                            if temp_idx > 0:  # If not the first temperature
                                logger.info(f"  Temperature {temperature} succeeded for graph {i}!")
                            break

                    except Exception as e:
                        logger.error(f"  Error during generation for graph {i} (temp={temperature}): {e}")
                        logger.error(traceback.format_exc())
                        # Continue to next temperature or skip if last one
                        continue

                # If we couldn't generate a graph with any temperature, skip this iteration
                if gen_graph is None:
                    pbar.update(1)
                    continue

                # --- Clean: Remove Isolated Nodes ---
                g_cleaned = gen_graph
                isolates = list(nx.isolates(gen_graph))
                if isolates:
                    nodes_to_keep = list(set(gen_graph.nodes()) - set(isolates))
                    g_cleaned = gen_graph.subgraph(nodes_to_keep).copy() if nodes_to_keep else nx.DiGraph()

                # --- Analyze Cleaned Graph ---
                cleaned_node_count = g_cleaned.number_of_nodes()
                cleaned_edge_count = g_cleaned.number_of_edges()

                # Calculate level, infer types, calculate validity on cleaned graph
                final_max_level_cleaned = -1
                inferred_types_cleaned = {}
                paper_val = 0.0
                extensive_val_details = {}
                is_extensively_valid = False

                if cleaned_node_count > 0:
                    try:
                        if nx.is_directed_acyclic_graph(g_cleaned):
                            _, final_max_level_cleaned = _calculate_levels(g_cleaned)
                        else:
                            final_max_level_cleaned = -2  # Non-DAG

                        inferred_types_cleaned = infer_node_types(g_cleaned)
                        g_cleaned.graph['_inferred_types_cleaned'] = inferred_types_cleaned  # Store for viz

                        # Calculate validity metrics using functions from aig_evaluate
                        paper_val = calculate_paper_validity(g_cleaned)
                        extensive_val_details = calculate_extensive_validity(g_cleaned, check_connectivity=True)
                        is_extensively_valid = extensive_val_details.get("overall_valid", False)

                    except Exception as e:
                        logger.error(f"  Error during analysis for graph {i}: {e}")
                        logger.error(traceback.format_exc())
                        final_max_level_cleaned = -3  # Analysis error
                        paper_val = np.nan  # Mark metrics as NaN on error
                        is_extensively_valid = False

                all_graphs_data.append({
                    "graph": g_cleaned,  # Store the cleaned graph object
                    "analysis_index": i,
                    "is_extensively_valid": is_extensively_valid,
                    "paper_validity": paper_val,
                    "node_count": cleaned_node_count,
                    "edge_count": cleaned_edge_count,
                    "max_level": final_max_level_cleaned,  # Level of cleaned graph
                })
                pbar.update(1)  # End of generation loop for one graph

        # --- Aggregate and Report Results for the Current Model ---
        num_analyzed = len(all_graphs_data)
        logger.info(f"  Finished generation. Analyzing {num_analyzed} results...")

        if num_analyzed > 0:
            # Calculate overall averages (handle potential NaNs in paper validity)
            all_node_counts = [d["node_count"] for d in all_graphs_data]
            all_edge_counts = [d["edge_count"] for d in all_graphs_data]
            all_valid_levels = [d["max_level"] for d in all_graphs_data if d["max_level"] >= 0]  # Levels for valid DAGs
            all_paper_validities = [d["paper_validity"] for d in all_graphs_data]
            all_extensive_valid = [d["is_extensively_valid"] for d in all_graphs_data]

            avg_nodes = np.mean(all_node_counts)
            avg_edges = np.mean(all_edge_counts)
            avg_max_level = np.mean(all_valid_levels) if all_valid_levels else 0.0
            max_max_level = np.max(all_valid_levels) if all_valid_levels else 0.0
            avg_paper_validity = np.nanmean(all_paper_validities)  # Use nanmean
            percent_extensive_valid = np.mean(all_extensive_valid) * 100.0

            # Store aggregated results
            results_list.append({
                "model": model_basename,
                "num_generated": graphs_generated_count,
                "num_analyzed": num_analyzed,
                "target_nodes": args.nodes_target,
                "avg_nodes": f"{avg_nodes:.2f}",
                "avg_edges": f"{avg_edges:.2f}",
                "avg_paper_validity": f"{avg_paper_validity:.4f}",
                "percent_extensive_valid": f"{percent_extensive_valid:.2f}",
                "avg_max_level": f"{avg_max_level:.2f}",
                "max_max_level": f"{max_max_level:.0f}",
                "temperature": args.temp,
                "patience": args.patience,
            })

            # Print aggregated results
            logger.info(f"    Avg Nodes (cleaned): {avg_nodes:.2f}")
            logger.info(f"    Avg Edges (cleaned): {avg_edges:.2f}")
            logger.info(f"    Avg Paper Validity: {avg_paper_validity:.4f}")
            logger.info(
                f"    Extensively Valid: {percent_extensive_valid:.2f}% ({sum(all_extensive_valid)}/{num_analyzed})")
            logger.info(f"    Avg Max Level (valid DAGs): {avg_max_level:.2f}")
            logger.info(f"    Max Max Level (valid DAGs): {max_max_level}")

            # --- Select and Visualize Best Valid Graphs ---
            if args.save_plots and num_analyzed > 0:
                valid_graphs_data = [d for d in all_graphs_data if d["is_extensively_valid"]]
                logger.info(f"  Found {len(valid_graphs_data)} extensively valid graphs for plotting.")

                if valid_graphs_data:
                    sort_key = "node_count" if args.plot_sort_by == "nodes" else "max_level"
                    valid_graphs_data_sorted = sorted(
                        valid_graphs_data,
                        key=lambda x: x.get(sort_key, 0),
                        reverse=True  # Sort descending
                    )

                    num_plots_to_save = min(len(valid_graphs_data_sorted), args.num_plots)
                    logger.info(
                        f"  Saving plots for top {num_plots_to_save} valid graphs (sorted by {args.plot_sort_by})...")

                    for rank, graph_data in enumerate(valid_graphs_data_sorted[:num_plots_to_save]):
                        plot_filename = os.path.join(
                            args.plot_dir,
                            f"{model_basename}_valid_{args.plot_sort_by}_rank{rank + 1}_idx{graph_data['analysis_index']}.png"
                        )
                        # Call the visualization function defined in this script
                        visualize_aig_structure(graph_data["graph"], plot_filename)
                    logger.info(f"  Finished saving plots.")
                else:
                    logger.info("  No extensively valid graphs found to plot.")

        else:  # No graphs analyzed
            logger.warning("    No graphs were successfully generated or analyzed for this model.")
            # Add empty/error result row
            results_list.append({
                "model": model_basename, "num_generated": graphs_generated_count, "num_analyzed": 0,
                "target_nodes": args.nodes_target, "avg_nodes": "N/A", "avg_edges": "N/A",
                "avg_paper_validity": "N/A", "percent_extensive_valid": "N/A",
                "avg_max_level": "N/A", "max_max_level": "N/A",
                "temperature": args.temp, "patience": args.patience,
            })

        elapsed_time_model = time.time() - start_time_model
        logger.info(f"  Model evaluation took {elapsed_time_model:.2f} seconds.")
        # --- End of loop for one model ---

    # --- Save Final Results ---
    if results_list:
        results_df = pd.DataFrame(results_list)
        # Define column order
        cols_order = [
            "model", "num_generated", "num_analyzed", "target_nodes", "temperature", "patience",
            "avg_nodes", "avg_edges", "avg_paper_validity", "percent_extensive_valid",
            "avg_max_level", "max_max_level"
        ]
        # Reorder/select columns, handling potential missing ones if errors occurred
        final_cols = [col for col in cols_order if col in results_df.columns]
        results_df = results_df[final_cols]

        try:
            results_df.to_csv(args.output_csv, index=False)
            logger.info(f"\nEvaluation results saved to {args.output_csv}")
        except Exception as e:
            logger.error(f"\nError saving results to CSV: {e}")
            logger.error("\nResults DataFrame:")
            logger.error(results_df.to_string())
    else:
        logger.error("\nNo models were successfully evaluated.")

    return 0


if __name__ == "__main__":
    import sys

    # Set a higher recursion depth if deep graphs cause issues with NetworkX/layout (use with caution)
    # sys.setrecursionlimit(5000)
    sys.exit(main())