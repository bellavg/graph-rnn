# get_aigs.py
import sys
import logging
import os
import time
import json
import argparse
import numpy as np
import torch
import networkx as nx
import random
import inspect
from typing import Dict, Any, List, Tuple, Optional, Callable # Added Callable
import statistics # <<< ADDED IMPORT

# --- Import from project files ---
# Model imports (keep as is)
try:
    from model import (GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP,
                       GraphLevelAttentionRNN, EdgeLevelAttentionRNN,
                       GraphLevelLSTM, EdgeLevelLSTM,
                       GraphLevelAttentionLSTM, EdgeLevelAttentionLSTM)
    MODEL_CLASSES_LOADED = True
except ImportError as e:
    # ... error handling ...
    sys.exit(1)

# Generation imports (keep as is)
try:
    from generate_aigs import generate, rnn_edge_gen, mlp_edge_gen, EDGE_TYPES, NUM_EDGE_FEATURES
except ImportError as e:
    # ... error handling ...
    sys.exit(1)

# Evaluation imports (Refined)
try:
    # AIG-specific structural evaluation
    from evaluate_aigs import (aig_to_networkx, infer_node_types,
                               calculate_structural_aig_validity, calculate_seadag_validity,
                               count_pi_po_paths, visualize_aig_structure)
    # Comparison metrics and helpers from evaluate.py
    from evaluate import (compare_graphs_mmd_degree, compare_graphs_mmd_clustering_coeff,
                          compare_graphs_mmd_orbit_stats, get_orbit_stats)
                          # Add other specific compare_graphs_mmd_* if needed later
    # Metric calculation helpers
    from graph_metrics import (average_degree, get_histogram_of_clustering_coeffs,
                               average_degree_centrality, average_betweenness_centrality,
                               average_closeness_centrality) # Import necessary functions
    # MMD implementation
    from mmd_stanford_impl import compute_mmd, gaussian_emd, gaussian
    # Dataset class
    from aig_dataset import AIGDataset
except ImportError as e:
    print(f"Error importing from evaluation/utility files: {e}. Ensure all files exist and are accessible.")
    sys.exit(1)

# --- Constants for MMD functions (keep as is) ---
mmd_stanford_fn_no_hist = lambda x, y: compute_mmd(x, y, kernel=gaussian_emd, is_hist=False, sigma=1.0)
mmd_stanford_fn_is_hist = lambda x, y: compute_mmd(x, y, kernel=gaussian_emd, is_hist=True, sigma=1.0)
mmd_stanford_fn_is_hist_clustering_settings = lambda x, y: compute_mmd(x, y, kernel=gaussian_emd, is_hist=True, sigma=1.0/10, distance_scaling=100.0)
mmd_stanford_fn_orbit_settings = lambda x, y: compute_mmd(x, y, kernel=gaussian, is_hist=False, sigma=30.0)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, # Set default level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to stdout
)
# Get logger for this main script
logger = logging.getLogger("get_aigs_main")

# --- Model Loading Function ---
def load_model_from_config(model_path):
    """Loads model based on config stored in checkpoint."""
    logger.info(f"Attempting to load state from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    try:
        state = torch.load(model_path, map_location=device)
    except Exception as e:
        logger.error(f"Failed to load checkpoint file '{model_path}': {e}", exc_info=True)
        raise

    config = state.get('config')
    if config is None:
         raise ValueError(f"Checkpoint {model_path} does not contain a 'config' dictionary.")
    logger.debug(f"Loaded config: {config}")

    # --- Extract Config ---
    try:
        # Default values or structure checks might be needed depending on config format
        data_config = config.get('data', {})
        model_config = config.get('model', {})

        if data_config.get('m') is None:  # Check if 'm' is missing or None
            known_m_value = 88  # <<<--- SET YOUR KNOWN CORRECT M VALUE HERE
            data_config['m'] = known_m_value
            logger.warning(f"Manually injected 'm={known_m_value}' into loaded config for {model_path}.")

        input_size = data_config.get('m') # History length 'm'
        if input_size is None: raise KeyError("Missing 'm' in data config.")

        # Node model type and attention
        node_model_type = model_config.get('node_model', 'gru').lower()
        use_node_attention = model_config.get('use_attention', False) # General flag
        # Specific node model config dict (GraphRNN or GraphLSTM)
        node_config_dict = model_config.get('GraphRNN', model_config.get('GraphLSTM', {}))
        max_level_from_config = node_config_dict.get('max_level')
        if max_level_from_config is None:
            # Option 1: Raise an error if max_level MUST be in config
            # raise ValueError(f"'max_level' not found in 'model.{node_config_section}' config.")
            # Option 2: Add a manual injection/default (similar to 'm', but riskier)
            known_max_level = 18  # <<<--- SET YOUR KNOWN MAX_LEVEL HERE (from logs)
            node_config_dict['max_level'] = known_max_level
            logger.warning(f"Manually injected 'max_level={known_max_level}' into node_config_dict.")
        else:
            logger.info(f"Found max_level={max_level_from_config} in node config.")
        if not use_node_attention: # Check specific config if global is false
            use_node_attention = node_config_dict.get('use_attention', False)

        # Edge model type and attention
        edge_model_type = model_config.get('edge_model', 'rnn').lower()
        # Specific edge model config dict (EdgeRNN, EdgeLSTM, or EdgeMLP)
        edge_config_dict = model_config.get('EdgeRNN', model_config.get('EdgeLSTM', model_config.get('EdgeMLP', {})))
        use_edge_attention = edge_config_dict.get('use_attention', False) # Check edge specific config

        edge_feature_len = model_config.get('edge_feature_len', NUM_EDGE_FEATURES)

        logger.info(f"Model config suggests: node_model={node_model_type}, edge_model={edge_model_type}, "
                    f"node_attn={use_node_attention}, edge_attn={use_edge_attention}, "
                    f"m={input_size}, edge_features={edge_feature_len}")

        # Ensure edge_feature_len is passed down to constructors via config dicts
        node_config_dict['edge_feature_len'] = edge_feature_len
        edge_config_dict['edge_feature_len'] = edge_feature_len

    except KeyError as e:
         logger.error(f"Missing key in loaded config: {e}. Check checkpoint structure.", exc_info=True)
         raise ValueError(f"Missing key in config: {e}")
    except Exception as e:
        logger.error(f"Error parsing config: {e}", exc_info=True)
        raise

    # --- Instantiate Models ---
    if not MODEL_CLASSES_LOADED:
        raise RuntimeError("Model classes could not be imported. Cannot instantiate model.")

    node_model = None
    edge_model = None
    edge_gen_function = None # Will be set based on edge_model_type

    # --- Node Model ---
    # Determine node model output size if needed by edge RNN/LSTM
    node_output_size_for_edge = None
    if edge_model_type == 'rnn' or edge_model_type == 'lstm': # Corrected logic
        node_output_size_for_edge = edge_config_dict.get('hidden_size')
        if node_output_size_for_edge is None:
            raise ValueError("EdgeRNN/LSTM config missing 'hidden_size', needed for NodeModel output size.")
    node_config_dict['output_size'] = node_output_size_for_edge # Set output size for node model

    # Select Node Model Class
    NodeModelClass = None
    if node_model_type == 'lstm':
        NodeModelClass = GraphLevelAttentionLSTM if use_node_attention else GraphLevelLSTM
    elif node_model_type == 'gru':
        NodeModelClass = GraphLevelAttentionRNN if use_node_attention else GraphLevelRNN
    else:
        raise ValueError(f"Unsupported node_model type in config: {node_model_type}")
    logger.info(f"Using Node Model Class: {NodeModelClass.__name__}")

    # Filter config keys to match constructor signature
    sig_node = inspect.signature(NodeModelClass.__init__)
    valid_keys_node = {p for p in sig_node.parameters if p != 'self'}
    node_config_dict['input_size'] = input_size # Ensure m is passed
    filtered_node_config = {k: v for k, v in node_config_dict.items() if k in valid_keys_node}
    logger.debug(f"Node Model filtered config: {filtered_node_config}")

    try:
        node_model = NodeModelClass(**filtered_node_config).to(device)
    except TypeError as e:
        logger.error(f"TypeError instantiating {NodeModelClass.__name__}: {e}. Config: {filtered_node_config}", exc_info=True)
        raise

    # --- Edge Model ---
    EdgeModelClass = None
    if edge_model_type == 'rnn' or edge_model_type == 'lstm': # Corrected logic
        edge_gen_function = rnn_edge_gen
        # Select Edge RNN/LSTM Class based on node type and attention
        if node_model_type == 'lstm': # Assume EdgeLSTM if NodeLSTM
            EdgeModelClass = EdgeLevelAttentionLSTM if use_edge_attention else EdgeLevelLSTM
        else: # Assume EdgeRNN (GRU) otherwise
            EdgeModelClass = EdgeLevelAttentionRNN if use_edge_attention else EdgeLevelRNN
        logger.info(f"Using Edge Model Class: {EdgeModelClass.__name__}")

        # Filter config keys
        sig_edge = inspect.signature(EdgeModelClass.__init__)
        valid_keys_edge = {p for p in sig_edge.parameters if p != 'self'}
        filtered_edge_config = {k: v for k, v in edge_config_dict.items() if k in valid_keys_edge}
        logger.debug(f"Edge RNN/LSTM Model filtered config: {filtered_edge_config}")

        try:
            edge_model = EdgeModelClass(**filtered_edge_config).to(device)
        except TypeError as e:
            logger.error(f"TypeError instantiating {EdgeModelClass.__name__}: {e}. Config: {filtered_edge_config}", exc_info=True)
            raise

    elif edge_model_type == 'mlp':
        edge_gen_function = mlp_edge_gen
        EdgeModelClass = EdgeLevelMLP
        logger.info(f"Using Edge Model Class: {EdgeModelClass.__name__}")

        # Setup required MLP config keys
        try:
             # MLP input size is the hidden size of the node model output sequence
             mlp_input_dim = node_config_dict.get('hidden_size')
             if mlp_input_dim is None: raise KeyError("'hidden_size' not found in node config for MLP input")
             edge_config_dict['input_size'] = mlp_input_dim
             # MLP output size 'm' (history length for edge predictions)
             edge_config_dict['output_size'] = input_size
        except KeyError as e:
             raise ValueError(f"Missing key needed for EdgeMLP config setup: {e}")

        # Filter config keys
        sig_edge = inspect.signature(EdgeModelClass.__init__)
        valid_keys_edge = {p for p in sig_edge.parameters if p != 'self'}
        filtered_edge_config = {k: v for k, v in edge_config_dict.items() if k in valid_keys_edge}
        logger.debug(f"Edge MLP Model filtered config: {filtered_edge_config}")

        try:
            edge_model = EdgeModelClass(**filtered_edge_config).to(device)
        except TypeError as e:
            logger.error(f"TypeError instantiating {EdgeModelClass.__name__}: {e}. Config: {filtered_edge_config}", exc_info=True)
            raise
    else:
        raise ValueError(f"Unsupported edge_model type in config: {edge_model_type}")

    # --- Load State Dicts ---
    try:
        node_state_dict = state.get('node_model')
        edge_state_dict = state.get('edge_model')
        if node_state_dict is None or edge_state_dict is None:
            raise KeyError("Missing 'node_model' or 'edge_model' state_dict in checkpoint")
        node_model.load_state_dict(node_state_dict)
        edge_model.load_state_dict(edge_state_dict)
        logger.info("Model state dictionaries loaded successfully.")
    except KeyError as e:
        logger.error(f"State dict loading error: {e}")
        raise
    except RuntimeError as e:
         logger.error(f"Error loading state_dict (likely mismatched model architecture or keys): {e}", exc_info=True)
         raise

    mode = model_config.get('mode', 'aig') # Get generation mode, default 'aig'

    return node_model, edge_model, input_size, edge_gen_function, mode, edge_feature_len


# --- Helper Function: Generation ---
def get_generation(
    num_graphs_to_generate: int,
    models_tuple: Tuple[torch.nn.Module, torch.nn.Module, int, Callable, str, int],
    gen_params: Dict[str, Any]
    ) -> Tuple[List[nx.DiGraph], int]:
    """Generates graphs using the loaded models."""

    (node_model, edge_model, input_size, edge_gen_function,
     mode, edge_feature_len) = models_tuple

    logger.debug(f"Generating {num_graphs_to_generate} AIGs...")
    raw_generated_graphs: List[Optional[nx.DiGraph]] = []

    current_gen_params = {
        **gen_params,
        'temperature': gen_params.get('temperature', 1.0),
        'top_k': gen_params.get('top_k', 0),
        'top_p': gen_params.get('top_p', 0.0),
        'edge_sample_attempts': gen_params.get('edge_sample_attempts', 1)
    }

    generation_successful_count = 0
    for i in range(num_graphs_to_generate):
        # gen_start_time = time.time() # Optional: time each generation
        logger.debug(f"Generating graph {i+1}/{num_graphs_to_generate}...")
        try:
            # --- Call generate function from generate_aigs.py ---
            adj_conn, adj_inv = generate(
                node_model=node_model, edge_model=edge_model,
                input_size=input_size, edge_gen_function=edge_gen_function,
                mode=mode, edge_feature_len=edge_feature_len,
                # Pass generation parameters from current_gen_params
                **current_gen_params
            )

            graph = None
            if adj_conn is not None and adj_inv is not None and adj_conn.shape[0] > 0:
                # --- Convert to NetworkX ---
                graph = aig_to_networkx(adj_conn, adj_inv) # From evaluate_aigs.py
                if graph.number_of_nodes() == 0:
                     logger.warning(f"Generation {i+1}: aig_to_networkx resulted in empty graph.")
                     graph = None # Treat as failed generation
                else:
                     generation_successful_count += 1
                     # --- Pre-calculate inferred types ---
                     try:
                        graph.graph['_inferred_types_cleaned'] = infer_node_types(graph) # From evaluate_aigs.py
                     except Exception as infer_e:
                         logger.warning(f"Could not pre-calculate node types for graph {i+1}: {infer_e}")
            else:
                logger.warning(f"Generation {i+1} resulted in empty or None adjacency matrix.")

            raw_generated_graphs.append(graph) # Append graph or None

        except Exception as e:
            logger.error(f"Error during generation or conversion of graph {i+1}: {e}", exc_info=True)
            raw_generated_graphs.append(None) # Mark as failed

    # Filter out None values / truly empty graphs
    valid_generated_graphs = [g for g in raw_generated_graphs if isinstance(g, nx.DiGraph) and g.number_of_nodes() > 0]
    num_successfully_generated = len(valid_generated_graphs) # This should match generation_successful_count

    logger.debug(f"Finished generation. Generated {num_successfully_generated}/{num_graphs_to_generate} non-empty graphs.")

    # Return the list of valid graphs and the count
    return valid_generated_graphs, num_successfully_generated



def get_visualization(
    generated_graphs: List[nx.DiGraph],
    num_graphs_to_visualize: int,
    viz_dir: str,
    num_successfully_generated: int
    ) -> int:
    """Selects and visualizes the 'best' generated graphs."""

    num_graphs_visualized = 0
    if num_graphs_to_visualize <= 0 or num_successfully_generated == 0:
        logger.info("Visualization skipped (zero requested or no graphs generated).")
        return num_graphs_visualized

    num_to_viz = min(num_graphs_to_visualize, num_successfully_generated)
    logger.info(f"Selecting up to {num_to_viz} graphs for visualization (best structural score, then size)...")

    # Use evaluation results stored on graphs if available
    graph_scores = []
    for i, g in enumerate(generated_graphs):
        struct_info = g.graph.get('_structural_validity')
        if struct_info and struct_info.get('is_dag', False):
             score = struct_info.get('validity_score', 0.0)
             nodes = g.number_of_nodes()
             graph_scores.append((score, nodes, i)) # Store score, size, original index
        elif struct_info and not struct_info.get('is_dag', False):
             # Optionally include non-DAGs with low score if desired
             pass
        else:
             # If no eval results, fallback to just size or index
             logger.debug(f"No structural info for graph {i} during viz selection.")
             graph_scores.append((0.0, g.number_of_nodes(), i)) # Assign score 0

    if not graph_scores:
         logger.warning("No graphs found to select for visualization.")
         return num_graphs_visualized

    # Sort: higher score first, then larger size first (or smaller size: x[1])
    graph_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    graphs_to_visualize_indices = [idx for score, nodes, idx in graph_scores[:num_to_viz]]
    logger.info(f"Selected graph indices for visualization: {graphs_to_visualize_indices}")

    # Visualize the selected graphs
    logger.info(f"Visualizing {len(graphs_to_visualize_indices)} selected graphs...")
    for rank, graph_index in enumerate(graphs_to_visualize_indices):
         if graph_index >= len(generated_graphs): # Safety check
              logger.warning(f"Graph index {graph_index} out of bounds for visualization. Skipping.")
              continue
         g_to_viz = generated_graphs[graph_index]
         # Find score/nodes for filename (more robust lookup)
         score, nodes = 0.0, 0
         for s, n, idx in graph_scores:
              if idx == graph_index:
                  score, nodes = s, n
                  break

         fname = f"rank_{rank+1:02d}_score_{score:.3f}_nodes_{nodes}_idx_{graph_index}.png"
         output_path = os.path.join(viz_dir, fname)
         try:
             # Call visualize function from evaluate_aigs.py
             visualize_aig_structure(g_to_viz, output_file=output_path)
             num_graphs_visualized += 1
         except Exception as e:
             logger.error(f"Failed to visualize graph index {graph_index} ({output_path}): {e}", exc_info=True)

    logger.info(f"Visualizations saved in {viz_dir}")
    return num_graphs_visualized


def get_evaluation(
    generated_graphs: List[nx.DiGraph],
        test_graphs: List[nx.DiGraph],
    num_graphs_to_evaluate: int, # Max number to evaluate from the list
    num_successfully_generated: int # Actual number of valid graphs generated
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates structural properties of the generated graphs."""

    aggregated_evaluation_results: Dict[str, Any] = {}
    evaluation_results_list: List[Dict[str, Any]] = []

    if num_successfully_generated == 0:
        logger.warning("No graphs to evaluate.")
        return aggregated_evaluation_results, evaluation_results_list

    # Decide how many graphs to actually evaluate
    num_to_evaluate = num_successfully_generated
    graphs_for_eval = generated_graphs
    logger.info(f"Evaluating structural properties of {num_to_evaluate} generated graphs...")
    eval_start_time = time.time()

    for i, g in enumerate(graphs_for_eval):
        # Use the original index if needed (though less relevant now)
        graph_eval_results = {'graph_index_in_list': i}
        try:
            logger.debug(f"Evaluating graph {i+1}/{num_to_evaluate}...")
            # Ensure node types are inferred if not already done (safety check)
            if '_inferred_types_cleaned' not in g.graph:
                 g.graph['_inferred_types_cleaned'] = infer_node_types(g)

            # --- Call evaluation functions from evaluate_aigs.py ---
            struct_info = calculate_structural_aig_validity(g)
            seadag_v = calculate_seadag_validity(g)
            path_info = count_pi_po_paths(g)

            # Store results on graph object (optional)
            g.graph['_structural_validity'] = struct_info
            g.graph['_seadag_validity'] = seadag_v
            g.graph['_path_info'] = path_info

            # Store results for aggregation
            graph_eval_results.update(struct_info)
            graph_eval_results['seadag_validity'] = seadag_v
            graph_eval_results.update(path_info)

        except Exception as eval_e:
            logger.error(f"Error evaluating graph {i+1}: {eval_e}", exc_info=True)
            graph_eval_results["error"] = str(eval_e) # Mark error

        evaluation_results_list.append(graph_eval_results)

    eval_time = time.time() - eval_start_time
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds.")

    # --- Aggregate Evaluation Results ---
    aggregated_evaluation_results["num_graphs_evaluated"] = num_to_evaluate
    if num_to_evaluate > 0:
        valid_evals = [r for r in evaluation_results_list if 'error' not in r]
        num_valid_evals = len(valid_evals)
        aggregated_evaluation_results["num_valid_evaluations"] = num_valid_evals

        if num_valid_evals > 0:
            aggregated_evaluation_results["evaluated_dag_rate"] = sum(r.get('is_dag', False) for r in valid_evals) / num_valid_evals
            aggregated_evaluation_results["avg_structural_validity_score"] = round(np.mean([r.get('validity_score', 0.0) for r in valid_evals]), 4)
            aggregated_evaluation_results["avg_seadag_validity"] = round(np.mean([r.get('seadag_validity', 0.0) for r in valid_evals]), 4)
            aggregated_evaluation_results["num_perfect_structural_validity"] = sum(1 for r in valid_evals if r.get('validity_score') == 1.0)
            aggregated_evaluation_results["num_perfect_seadag_validity"] = sum(1 for r in valid_evals if r.get('seadag_validity') == 1.0)
            aggregated_evaluation_results["avg_nodes_eval"] = round(np.mean([r.get('num_nodes', 0) for r in valid_evals]), 2)
            aggregated_evaluation_results["avg_pi_eval"] = round(np.mean([r.get('num_pi', 0) for r in valid_evals]), 2) # Renamed key
            aggregated_evaluation_results["avg_po_eval"] = round(np.mean([r.get('num_po', 0) for r in valid_evals]), 2) # Renamed key
            aggregated_evaluation_results["avg_and_eval"] = round(np.mean([r.get('num_and', 0) for r in valid_evals]), 2) # Renamed key
            aggregated_evaluation_results["total_invalid_fanin_nodes_eval"] = sum(r.get('num_invalid_fanin', 0) for r in valid_evals)
            aggregated_evaluation_results["total_unknown_type_nodes_eval"] = sum(r.get('num_unknown', 0) for r in valid_evals)

            path_evals = [r for r in valid_evals if r.get('is_dag', False)]
            if path_evals:
                 aggregated_evaluation_results["avg_fraction_pis_connected"] = round(np.mean([r.get('fraction_pis_connected', 0.0) for r in path_evals]), 4)
                 aggregated_evaluation_results["avg_fraction_pos_connected"] = round(np.mean([r.get('fraction_pos_connected', 0.0) for r in path_evals]), 4)
                 aggregated_evaluation_results["avg_pis_reaching_po"] = round(np.mean([r.get('num_pis_reaching_po', 0) for r in path_evals]), 2)
                 aggregated_evaluation_results["avg_pos_reachable_from_pi"] = round(np.mean([r.get('num_pos_reachable_from_pi', 0) for r in path_evals]), 2)

    if not test_graphs:
        logger.warning("Test dataset is empty or failed to load. Skipping comparison metrics.")
    elif not generated_graphs:
        logger.warning("Generated graph list is empty. Skipping comparison metrics.")
    else:
        logger.info(
            f"Calculating comparison metrics between {len(generated_graphs)} generated and {len(test_graphs)} test graphs...")
        comp_metric_start_time = time.time()
        try:
            # --- MMD Metrics ---
            # Wrap MMD calls in try-except blocks as they can be sensitive
            try:
                mmd_deg = compare_graphs_mmd_degree(generated_graphs, test_graphs, mmd_stanford_fn_is_hist)
                aggregated_evaluation_results["mmd_degree_distribution"] = round(mmd_deg, 6)
            except Exception as e:
                logger.error(f"Error calculating MMD Degree: {e}", exc_info=False)

            try:
                mmd_clus = compare_graphs_mmd_clustering_coeff(generated_graphs, test_graphs,
                                                               mmd_stanford_fn_is_hist_clustering_settings)
                aggregated_evaluation_results["mmd_clustering_coeff"] = round(mmd_clus, 6)
            except Exception as e:
                logger.error(f"Error calculating MMD Clustering Coeff: {e}", exc_info=False)

            try:
                # Ensure orca executable is available and permissions are set if running orbit stats
                mmd_orbit = compare_graphs_mmd_orbit_stats(generated_graphs, test_graphs,
                                                           mmd_stanford_fn_orbit_settings)
                aggregated_evaluation_results["mmd_orbit_stats"] = round(mmd_orbit, 6)
            except FileNotFoundError:
                logger.error("Orca executable not found. Skipping MMD Orbit Stats.")
            except Exception as e:
                logger.error(f"Error calculating MMD Orbit Stats: {e}", exc_info=False)

            # --- Average Metrics (Compare averages of sets) ---
            # Define a helper to calculate average metric for a list of graphs
            def calculate_avg_metric(graph_list, metric_func, metric_name):
                values = []
                for g in graph_list:
                    if g.number_of_nodes() > 0:  # Avoid errors on empty graphs
                        try:
                            # Handle functions that return dicts (centralities) vs single values
                            metric_val = metric_func(g)
                            if isinstance(metric_val, dict):
                                values.extend(list(metric_val.values()))  # Use all node values
                                # Or: values.append(statistics.mean(metric_val.values())) # Use avg per graph
                            else:
                                values.append(metric_val)
                        except Exception as e:
                            logger.warning(f"Could not calculate {metric_name} for a graph: {e}")
                return round(statistics.mean(values), 4) if values else 0.0

            # Calculate averages for generated graphs
            aggregated_evaluation_results["avg_degree_gen"] = calculate_avg_metric(generated_graphs, average_degree,
                                                                                   "Avg Degree")
            aggregated_evaluation_results["avg_clustering_coeff_gen"] = calculate_avg_metric(generated_graphs,
                                                                                             nx.average_clustering,
                                                                                             "Avg Clustering")  # Use nx.average_clustering
            aggregated_evaluation_results["avg_density_gen"] = calculate_avg_metric(generated_graphs, nx.density,
                                                                                    "Density")
            aggregated_evaluation_results["avg_transitivity_gen"] = calculate_avg_metric(generated_graphs,
                                                                                         nx.transitivity,
                                                                                         "Transitivity")

            # Calculate averages for test graphs
            aggregated_evaluation_results["avg_degree_test"] = calculate_avg_metric(test_graphs, average_degree,
                                                                                    "Avg Degree")
            aggregated_evaluation_results["avg_clustering_coeff_test"] = calculate_avg_metric(test_graphs,
                                                                                              nx.average_clustering,
                                                                                              "Avg Clustering")
            aggregated_evaluation_results["avg_density_test"] = calculate_avg_metric(test_graphs, nx.density, "Density")
            aggregated_evaluation_results["avg_transitivity_test"] = calculate_avg_metric(test_graphs, nx.transitivity,
                                                                                          "Transitivity")

            comp_metric_time = time.time() - comp_metric_start_time
            logger.info(f"Comparison metrics calculated in {comp_metric_time:.2f} seconds.")

        except Exception as e:
            logger.error(f"Error occurred during comparison metric calculation: {e}", exc_info=True)
        # --- >>> END: Comparison Metrics <<< ---

        # Log summary (Combined)
    logger.info("--- Aggregated Evaluation Summary (Structural & Comparison) ---")
    for key, value in aggregated_evaluation_results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("---------------------------------------------------------------")

    return aggregated_evaluation_results, evaluation_results_list




def aig_control(
    model_checkpoint_path: str,
    output_dir: str,
    gen_params: Dict[str, Any],
    graph_file: str, # <<< Correct placement
    num_graphs_to_generate: int = 50,
    num_graphs_to_evaluate: int = 50,
    evaluate: bool = True,
    visualize: bool = True,
    num_graphs_to_visualize: int = 5,
num_test_graphs: Optional[int] = None # <<< ADD THIS LINE
) -> Dict[str, Any]:
    """
    Main control function to load model, generate, evaluate, and visualize AIGs.
    Orchestrates calls to helper functions.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    if visualize:
        os.makedirs(viz_dir, exist_ok=True)

    logger.info(f"Starting AIG control process. Output dir: {output_dir}")
    logger.info(f"Graph file: {graph_file}") # Log the graph file being used
    logger.info(f"Requested Generations: {num_graphs_to_generate}, Evaluate: {'Yes' if evaluate else 'No'}, "
                f"Visualize: {'Yes' if visualize else 'No'}, Num Visualize: {num_graphs_to_visualize if visualize else 'N/A'}")
    logger.info(f"Base Generation Parameters: {gen_params}")

    # --- Initialize Aggregated Results ---
    aggregated_results: Dict[str, Any] = {
        "model_checkpoint": model_checkpoint_path,
        "generation_params_base": gen_params,
        "output_directory": output_dir,
        "requested_generations": num_graphs_to_generate,
        "evaluation_enabled": evaluate,
        "visualization_enabled": visualize,
        "requested_visualizations": num_graphs_to_visualize if visualize else 0,
        "graph_file_used": graph_file,
    }

    # 1. Load Model
    logger.info(f"Loading model from {model_checkpoint_path}...")
    config = {} # Initialize config dict
    try:
        models_tuple = load_model_from_config(model_checkpoint_path)
        logger.info("Model loaded successfully.")
        # Try to get config from loaded state for dataset params
        try:
             state = torch.load(model_checkpoint_path, map_location='cpu')
             config = state.get('config', {})
             state = None # Free memory
        except Exception as config_e:
             logger.warning(f"Could not load config from checkpoint for dataset parameters: {config_e}")
             config = {} # Default if config loading fails

    except FileNotFoundError:
         logger.error(f"Model checkpoint not found: {model_checkpoint_path}")
         aggregated_results["error"] = "Model checkpoint not found."
         # Optionally save partial results before returning
         # save_results_json(aggregated_results, output_dir, "aig_summary.json") # Example helper
         return aggregated_results # Exit early
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        aggregated_results["error"] = f"Model loading failed: {e}"
        # save_results_json(aggregated_results, output_dir, "aig_summary.json") # Example helper
        return aggregated_results # Exit early


    # 2. Load Test Dataset (if evaluating)
    test_graphs = []
    if evaluate:
        try:
            logger.info(f"Loading test dataset from: {graph_file}")
            # Use train_split from the loaded config if possible, else default
            train_split_from_config = config.get('data', {}).get('train_split', 0.9) # Default 0.9

            test_dataset = AIGDataset(
                graph_file=graph_file,
                training=False, # Load the test split
                train_split=train_split_from_config,
                # Add other relevant AIGDataset params if needed (e.g., max_graphs from config?)
                # max_graphs=config.get('data', {}).get('max_graphs'), # Example
            )
            # Retrieve the actual graph objects for the test split
            if hasattr(test_dataset, 'graphs') and test_dataset.graphs is not None:
                 test_graphs = [g for g in test_dataset.graphs if isinstance(g, nx.DiGraph)] # Ensure they are graphs
                 if num_test_graphs is not None and 0 < num_test_graphs < len(test_graphs):
                     logger.info(f"Randomly sampling {num_test_graphs} graphs from the test set for evaluation...")
                     try:
                         test_graphs = random.sample(test_graphs, num_test_graphs)
                         logger.info(f"Using {len(test_graphs)} sampled test graphs.")
                     except ValueError as e:
                         logger.error(f"Error during random sampling: {e}. Using full test set.")
                 elif num_test_graphs is not None:
                     logger.warning(
                         f"--num-test-graphs value ({num_test_graphs}) is invalid or >= total test graphs. Using all {len(test_graphs)} test graphs.")
                 else:
                     logger.info(f"Using all {len(test_graphs)} loaded test graphs for evaluation.")
            else:
                 test_graphs = []

            logger.info(f"Loaded {len(test_graphs)} graphs for test set.")
            if not test_graphs:
                 logger.warning("Test dataset is empty. Comparison metrics will be skipped.")
                 # Decide if this should be a fatal error or just skip comparisons
                 # evaluate = False # Option: Disable evaluation if test set empty

        except FileNotFoundError:
            logger.error(f"Test dataset file not found: {graph_file}. Comparison metrics will be skipped.")
            test_graphs = []
        except Exception as e:
            logger.error(f"Error loading test dataset: {e}. Comparison metrics will be skipped.", exc_info=True)
            test_graphs = []


    # 3. Generate Graphs
    generated_graphs, num_successfully_generated = get_generation(
        num_graphs_to_generate=num_graphs_to_generate,
        models_tuple=models_tuple,
        gen_params=gen_params
    )
    aggregated_results["num_graphs_generated"] = num_successfully_generated

    # --- Early Exit if No Graphs Generated ---
    if num_successfully_generated == 0:
        logger.warning("No graphs were generated successfully. Skipping evaluation and visualization.")
        aggregated_results["warning"] = "No graphs generated successfully."
        # Save partial summary and exit
        results_file = os.path.join(output_dir, "aig_summary.json")
        try:
            # Use a helper or inline the save logic
            def convert_numpy(obj):
                if isinstance(obj, np.integer): return int(obj)
                elif isinstance(obj, np.floating): return float(obj)
                elif isinstance(obj, np.ndarray): return obj.tolist()
                return obj
            serializable_results = json.loads(json.dumps(aggregated_results, default=convert_numpy))
            with open(results_file, "w") as f: json.dump(serializable_results, f, indent=2)
            logger.info(f"Partial results saved to {results_file}")
        except Exception as e: logger.error(f"Failed to save partial results file: {e}")
        return aggregated_results
    # --- End Early Exit ---

    # 4. Evaluate Generated Graphs (if requested)
    aggregated_evaluation_results = {}
    evaluation_details_list = [] # Keep this if you want detailed per-graph results later
    if evaluate:
        # Call get_evaluation, passing the loaded test_graphs
        aggregated_evaluation_results, evaluation_details_list = get_evaluation(
            generated_graphs=generated_graphs,
            test_graphs=test_graphs,  # Pass the loaded test graphs
            num_graphs_to_evaluate=num_graphs_to_evaluate,
            num_successfully_generated=num_successfully_generated
        )
        # Merge evaluation results into main results dict
        aggregated_results.update(aggregated_evaluation_results)
        # Optionally add the detailed list if saving it
        # aggregated_results["evaluation_details_per_graph"] = evaluation_details_list
    else:
         logger.info("Evaluation step skipped as per request.")


    # 5. Visualize Generated Graphs (if requested)
    num_visualized = 0
    if visualize:
        num_visualized = get_visualization(
            generated_graphs=generated_graphs,
            num_graphs_to_visualize=num_graphs_to_visualize,
            viz_dir=viz_dir,
            num_successfully_generated=num_successfully_generated
        )
        aggregated_results["num_graphs_visualized"] = num_visualized
    else:
         logger.info("Visualization step skipped as per request.")
         aggregated_results["num_graphs_visualized"] = 0


    # 6. Save Final Aggregated Results
    results_file = os.path.join(output_dir, "aig_summary.json")
    try:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        serializable_results = json.loads(json.dumps(aggregated_results, default=convert_numpy))

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Aggregated results saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results JSON file: {e}")

    # 7. Return Results
    total_time = time.time() - start_time
    logger.info(f"AIG control process finished in {total_time:.2f} seconds.")
    aggregated_results['total_time_seconds'] = round(total_time, 2)

    return aggregated_results

# --- Main Execution Guard (Update to use new evaluate argument) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate, Evaluate, and Visualize AIGs using a trained model.")

    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save generated graphs, evaluations, and visualizations.")
    # --- ENSURE graph-file argument is correctly defined ---
    # It was graph_file in the previous response, ensure consistency or fix here
    parser.add_argument("--graph-file", type=str, default="./dataset/final_data.pkl", # Changed from graph_file
                        help="Path to the dataset pickle file (e.g., ./dataset/final_data.pkl)")

    # Generation and Evaluation Control
    parser.add_argument("--num-generate", type=int, default=1000, # Renamed from num-graphs
                        help="Number of AIGs to attempt generating.")
    parser.add_argument("--num-test-graphs", type=int, default=1000,
                        help="Number of test graphs to randomly sample for evaluation (default: use all).")
    parser.add_argument("--evaluate", action='store_true', # Added flag
                        help="Evaluate structural and comparison metrics for generated graphs.") # Updated help text

    # Visualization Control
    parser.add_argument("--visualize", action='store_true',
                        help="Enable visualization of generated AIGs.")
    parser.add_argument("--num-visualize", type=int, default=5,
                        help="Maximum number of 'best' AIGs to visualize (if --visualize is enabled).")

    args = parser.parse_args()

    # --- Base Generation Config Dictionary ---
    GENERATION_CONFIG = {
        'max_nodes': 100, # Example, adjust as needed
        'min_nodes': 8,   # Example
        'patience': 30,   # Example
        'temperature': 1.2,
        'top_k': 0,
        'top_p': 0.0,
        'edge_sample_attempts': 3,
    }
    # You could override GENERATION_CONFIG with args here if needed

    logger.info("Starting AIG generation/evaluation script with command line arguments...")

    # --- CORRECTED Call to aig_control ---
    final_results = aig_control(
        model_checkpoint_path=args.model_path,
        output_dir=args.output_dir,
        gen_params=GENERATION_CONFIG,
        graph_file=args.graph_file,           # <<< Pass args.graph_file here
        num_graphs_to_generate=args.num_generate,
        evaluate=args.evaluate,                  # Pass the evaluate flag
        visualize=args.visualize,
        num_graphs_to_visualize=args.num_visualize,
        num_test_graphs=args.num_test_graphs
    )
    # --- End Correction ---

    logger.info("--- Final Control Function Summary ---")
    # Use default=str to handle potential non-serializable types like numpy numbers
    print(json.dumps(final_results, indent=2, default=str))
    logger.info("--- End of Script ---")