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
# In src/get_aigs.py

import inspect # Ensure inspect is imported at the top

# ... other imports ...

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
        data_config = config.get('data', {})
        model_config = config.get('model', {})

        # Manually inject 'm' if missing (keep your existing logic)
        if data_config.get('m') is None:
            known_m_value = 88
            data_config['m'] = known_m_value
            logger.warning(f"Manually injected 'm={known_m_value}' into loaded config for {model_path}.")

        input_size = data_config.get('m')
        if input_size is None: raise KeyError("Missing 'm' in data config.")

        # Node model type and attention
        node_model_type = model_config.get('node_model', 'gru').lower() # Default 'gru'
        use_node_attention = model_config.get('use_attention', False) # General flag

        # <<< START MODIFICATION >>>
        # Determine the correct config section for the node model
        node_config_section_name = None
        if node_model_type == 'lstm':
             node_config_section_name = 'GraphAttentionLSTM' if use_node_attention else 'GraphLSTM'
        elif node_model_type == 'gru':
             node_config_section_name = 'GraphAttentionRNN' if use_node_attention else 'GraphRNN'
        else:
             raise ValueError(f"Unsupported node_model type in config: {node_model_type}")

        logger.info(f"Attempting to load node parameters from config section: 'model.{node_config_section_name}'")

        # Get the specific config dictionary for the node model
        node_config_dict = model_config.get(node_config_section_name)
        if node_config_dict is None:
             # Fallback for attention models if specific section is missing but base exists
             if use_node_attention and node_model_type == 'gru' and 'GraphRNN' in model_config:
                 logger.warning(f"Config section 'model.{node_config_section_name}' not found, using 'GraphRNN'.")
                 node_config_dict = model_config.get('GraphRNN')
             elif use_node_attention and node_model_type == 'lstm' and 'GraphLSTM' in model_config:
                 logger.warning(f"Config section 'model.{node_config_section_name}' not found, using 'GraphLSTM'.")
                 node_config_dict = model_config.get('GraphLSTM')
             else:
                 raise ValueError(f"Required config section 'model.{node_config_section_name}' not found in checkpoint.")

        # Ensure node_config_dict is a mutable dictionary
        node_config_dict = dict(node_config_dict) if node_config_dict else {}

        # --- Explicitly check for required parameters ---
        required_node_params = ['embedding_size', 'hidden_size', 'num_layers']
        missing_params = [p for p in required_node_params if p not in node_config_dict]
        if missing_params:
            raise KeyError(f"Missing required node parameters in config section '{node_config_section_name}': {missing_params}")
        # --- End Check ---

        # Manually inject max_level if needed (keep your existing logic)
        max_level_from_config = node_config_dict.get('max_level')
        if max_level_from_config is None:
            known_max_level = 18
            node_config_dict['max_level'] = known_max_level
            logger.warning(f"Manually injected 'max_level={known_max_level}' into node_config_dict.")
        else:
            logger.info(f"Found max_level={max_level_from_config} in node config.")
        # <<< END MODIFICATION >>>

        # --- Continue with edge model logic (mostly unchanged) ---
        edge_model_type = model_config.get('edge_model', 'rnn').lower()
        edge_config_dict_base = {} # Find the base config for the edge model
        if edge_model_type == 'attention_rnn':
             edge_config_dict_base = model_config.get('EdgeAttentionRNN', model_config.get('EdgeRNN', {})) # Fallback to EdgeRNN
        elif edge_model_type == 'rnn':
             edge_config_dict_base = model_config.get('EdgeRNN', {})
        elif edge_model_type == 'mlp':
             edge_config_dict_base = model_config.get('EdgeMLP', {})
        elif edge_model_type == 'attention_lstm':
             edge_config_dict_base = model_config.get('EdgeAttentionLSTM', model_config.get('EdgeLSTM', {}))
        elif edge_model_type == 'lstm':
             edge_config_dict_base = model_config.get('EdgeLSTM', {})
        else:
            raise ValueError(f"Unsupported edge_model type in config: {edge_model_type}")
        edge_config_dict = dict(edge_config_dict_base) # Make mutable

        use_edge_attention = edge_config_dict.get('use_attention', edge_model_type.startswith('attention'))

        edge_feature_len = node_config_dict.get('edge_feature_len', NUM_EDGE_FEATURES) # <<< Get edge_feature_len from node_config_dict


        logger.info(f"Model config suggests: node_model={node_model_type}, edge_model={edge_model_type}, "
                    f"node_attn={use_node_attention}, edge_attn={use_edge_attention}, "
                    f"m={input_size}, edge_features={edge_feature_len}")

        # Ensure edge_feature_len is passed down
        node_config_dict['edge_feature_len'] = edge_feature_len
        edge_config_dict['edge_feature_len'] = edge_feature_len


    except KeyError as e:
         logger.error(f"Missing key in loaded config: {e}. Check checkpoint structure.", exc_info=True)
         raise ValueError(f"Missing key in config: {e}")
    except Exception as e:
        logger.error(f"Error parsing config: {e}", exc_info=True)
        raise

    # --- Instantiate Models ---
    if not MODEL_CLASSES_LOADED: # Assuming this global flag exists
        raise RuntimeError("Model classes could not be imported. Cannot instantiate model.")

    node_model = None
    edge_model = None
    edge_gen_function = None

    # --- Node Model ---
    node_output_size_for_edge = None # Determine if edge model needs node output
    if edge_model_type in ['rnn', 'lstm', 'attention_rnn', 'attention_lstm']:
        # Ensure edge config has hidden size
        if 'hidden_size' not in edge_config_dict:
             raise ValueError(f"Edge model config missing 'hidden_size', needed for node output size.")
        node_output_size_for_edge = edge_config_dict.get('hidden_size')
        node_config_dict['output_size'] = node_output_size_for_edge # Set output size for node model

    # Select Node Model Class based on parsed type and attention
    NodeModelClass = None
    if node_model_type == 'lstm':
        NodeModelClass = GraphLevelAttentionLSTM if use_node_attention else GraphLevelLSTM
    elif node_model_type == 'gru':
        NodeModelClass = GraphLevelAttentionRNN if use_node_attention else GraphLevelRNN
    # Ensure NodeModelClass is assigned
    if NodeModelClass is None:
        raise ValueError(f"Could not determine NodeModelClass for type {node_model_type}")

    logger.info(f"Using Node Model Class: {NodeModelClass.__name__}")

    # Filter config keys to match constructor signature
    sig_node = inspect.signature(NodeModelClass.__init__)
    valid_keys_node = {p for p in sig_node.parameters if p != 'self'}
    # Add required params explicitly if not already in node_config_dict (should be checked above now)
    node_config_dict['input_size'] = input_size
    # node_config_dict['embedding_size'] = node_config_dict['embedding_size'] # Already checked
    # node_config_dict['hidden_size'] = node_config_dict['hidden_size'] # Already checked
    # node_config_dict['num_layers'] = node_config_dict['num_layers'] # Already checked
    if use_node_attention: # Add attention params if needed by class
         node_config_dict['attention_heads'] = node_config_dict.get('attention_heads', 4) # Default if missing
         node_config_dict['attention_dropout'] = node_config_dict.get('attention_dropout', 0.1) # Default if missing


    filtered_node_config = {k: v for k, v in node_config_dict.items() if k in valid_keys_node}
    logger.debug(f"Node Model filtered config: {filtered_node_config}")

    try:
        # Instantiate the node model with the filtered config
        node_model = NodeModelClass(**filtered_node_config).to(device)
    except TypeError as e:
        logger.error(f"TypeError instantiating {NodeModelClass.__name__}: {e}. Config: {filtered_node_config}", exc_info=True)
        raise

    # --- Edge Model ---
    # (Edge model instantiation logic - keep similar to your existing code,
    # ensuring edge_config_dict is populated correctly based on edge_model_type
    # and necessary parameters like input_size (from node hidden) and output_size (m)
    # are set correctly, especially for EdgeLevelMLP)

    # Example for EdgeLevelAttentionRNN:
    EdgeModelClass = None
    if edge_model_type == 'attention_rnn':
         edge_gen_function = rnn_edge_gen # Check if this function exists and handles attention models
         EdgeModelClass = EdgeLevelAttentionRNN
         logger.info(f"Using Edge Model Class: {EdgeModelClass.__name__}")
         # Filter config keys for EdgeLevelAttentionRNN
         sig_edge = inspect.signature(EdgeModelClass.__init__)
         valid_keys_edge = {p for p in sig_edge.parameters if p != 'self'}
         # Add default attention params if missing in config
         edge_config_dict['attention_heads'] = edge_config_dict.get('attention_heads', 4)
         edge_config_dict['attention_dropout'] = edge_config_dict.get('attention_dropout', 0.1)
         filtered_edge_config = {k: v for k, v in edge_config_dict.items() if k in valid_keys_edge}
         logger.debug(f"Edge Attention RNN Model filtered config: {filtered_edge_config}")
         try:
             edge_model = EdgeModelClass(**filtered_edge_config).to(device)
         except TypeError as e:
             logger.error(f"TypeError instantiating {EdgeModelClass.__name__}: {e}. Config: {filtered_edge_config}", exc_info=True)
             raise
    elif edge_model_type == 'mlp':
         edge_gen_function = mlp_edge_gen
         EdgeModelClass = EdgeLevelMLP
         # ... (ensure input/output sizes are set correctly for MLP based on node_hidden_size and input_size (m)) ...
         # Example setting required MLP params (ensure these exist in node/edge config)
         if 'hidden_size' not in node_config_dict: raise ValueError("Node config missing hidden_size for MLP input")
         edge_config_dict['input_size'] = node_config_dict['hidden_size']
         edge_config_dict['output_size'] = input_size # m
         # Filter and instantiate MLP
         sig_edge = inspect.signature(EdgeModelClass.__init__)
         valid_keys_edge = {p for p in sig_edge.parameters if p != 'self'}
         filtered_edge_config = {k: v for k, v in edge_config_dict.items() if k in valid_keys_edge}
         logger.debug(f"Edge MLP Model filtered config: {filtered_edge_config}")
         try:
             edge_model = EdgeModelClass(**filtered_edge_config).to(device)
         except TypeError as e:
              logger.error(f"TypeError instantiating {EdgeModelClass.__name__}: {e}. Config: {filtered_edge_config}", exc_info=True)
              raise
    # ... Add cases for other edge models (RNN, LSTM, AttentionLSTM) ...
    else:
        # Handle other edge model types or raise error
        raise ValueError(f"Unsupported edge_model type '{edge_model_type}' needs instantiation logic.")


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

    mode = model_config.get('mode', 'aig') # Get generation mode

    # Ensure all models were assigned
    if node_model is None or edge_model is None or edge_gen_function is None:
        raise RuntimeError("Failed to instantiate all required model components.")

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




# --- UPDATED get_evaluation FUNCTION ---
def get_evaluation(
    generated_graphs: List[nx.DiGraph],
    test_graphs: List[nx.DiGraph],
    num_graphs_to_evaluate: int, # Max number to evaluate from the list
    num_successfully_generated: int # Actual number of valid graphs generated
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates structural and comparison properties of the generated graphs.
    Includes enhanced error logging and more nuanced aggregation.
    """
    aggregated_evaluation_results: Dict[str, Any] = {}
    evaluation_results_list: List[Dict[str, Any]] = []

    if num_successfully_generated == 0:
        logger.warning("No graphs to evaluate.")
        return aggregated_evaluation_results, evaluation_results_list

    num_to_struct_evaluate = min(num_successfully_generated, num_graphs_to_evaluate)
    graphs_for_struct_eval = generated_graphs[:num_to_struct_evaluate]

    logger.info(f"Evaluating structural properties of {len(graphs_for_struct_eval)} generated graphs...")
    eval_start_time = time.time()
    num_exceptions = 0 # Count only exceptions caught

    for i, g in enumerate(graphs_for_struct_eval):
        graph_eval_results = {'graph_index_in_list': i}
        try:
            graph_size_info = "N/A"
            if isinstance(g, nx.DiGraph):
                 graph_size_info = f"(Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()})"
            logger.debug(f"Evaluating graph {i+1}/{len(graphs_for_struct_eval)} {graph_size_info}")

            if isinstance(g, nx.DiGraph) and '_inferred_types_cleaned' not in g.graph:
                 g.graph['_inferred_types_cleaned'] = infer_node_types(g)

            struct_info = calculate_structural_aig_validity(g)
            seadag_v = calculate_seadag_validity(g)
            path_info = count_pi_po_paths(g) # This returns 'error' key if not DAG

            if isinstance(g, nx.DiGraph):
                g.graph['_structural_validity'] = struct_info
                g.graph['_seadag_validity'] = seadag_v
                g.graph['_path_info'] = path_info

            # Store results for aggregation - include everything returned
            graph_eval_results.update(struct_info)
            graph_eval_results['seadag_validity'] = seadag_v
            graph_eval_results.update(path_info) # Includes potential 'error' key from path_info

            # Log warning if path_info reported an error (e.g., not DAG)
            if path_info.get('error'):
                 logger.warning(f"Path evaluation for graph {i+1} reported error: {path_info['error']}")

        except Exception as eval_e:
            # Log exceptions and mark the result dict with a different error key maybe?
            logger.error(f"Exception during structural evaluation of graph {i+1}: {eval_e}", exc_info=True)
            graph_eval_results["evaluation_exception"] = str(eval_e) # Use a different key
            num_exceptions += 1

        # Log the full dictionary before appending
        logger.debug(f"Graph {i+1} evaluation result dict: {graph_eval_results}")
        evaluation_results_list.append(graph_eval_results)

    eval_time = time.time() - eval_start_time
    logger.info(f"Structural evaluation loop completed in {eval_time:.2f} seconds with {num_exceptions} exceptions caught.")

    # --- Aggregate Structural Evaluation Results (Revised Logic) ---
    aggregated_evaluation_results["num_graphs_structurally_evaluated"] = len(graphs_for_struct_eval)
    aggregated_evaluation_results["num_structural_evaluation_exceptions"] = num_exceptions

    # --- Calculate stats based on results WITHOUT exceptions ---
    evals_without_exception = [r for r in evaluation_results_list if 'evaluation_exception' not in r]
    num_evals_without_exception = len(evals_without_exception)
    aggregated_evaluation_results["num_evals_without_exception"] = num_evals_without_exception

    if num_evals_without_exception > 0:
        # Calculate basic counts and SEADAG validity based on all non-exception results
        # These don't strictly require the graph to be a DAG
        aggregated_evaluation_results["avg_nodes_eval"] = round(np.mean([r.get('num_nodes', 0) for r in evals_without_exception]), 2)
        aggregated_evaluation_results["avg_pi_eval"] = round(np.mean([r.get('num_pi', 0) for r in evals_without_exception]), 2)
        aggregated_evaluation_results["avg_po_eval"] = round(np.mean([r.get('num_po', 0) for r in evals_without_exception]), 2)
        aggregated_evaluation_results["avg_and_eval"] = round(np.mean([r.get('num_and', 0) for r in evals_without_exception]), 2)
        aggregated_evaluation_results["avg_unknown_eval"] = round(np.mean([r.get('num_unknown', 0) for r in evals_without_exception]), 2)
        aggregated_evaluation_results["avg_invalid_fanin_eval"] = round(np.mean([r.get('num_invalid_fanin', 0) for r in evals_without_exception]), 2)
        aggregated_evaluation_results["avg_seadag_validity"] = round(np.mean([r.get('seadag_validity', 0.0) for r in evals_without_exception]), 4) # SEADAG validity doesn't require DAG check itself

        # --- Calculate DAG-dependent stats based ONLY on non-exception AND DAG results ---
        valid_dag_evals = [r for r in evals_without_exception if r.get('is_dag', False)]
        num_valid_dag_evals = len(valid_dag_evals)
        aggregated_evaluation_results["num_valid_dag_evals"] = num_valid_dag_evals

        if num_evals_without_exception > 0: # Calculate rate based on attempts without exceptions
             aggregated_evaluation_results["evaluated_dag_rate"] = num_valid_dag_evals / num_evals_without_exception
        else:
             aggregated_evaluation_results["evaluated_dag_rate"] = 0.0

        if num_valid_dag_evals > 0:
            # Structural validity score relies on being a DAG
            aggregated_evaluation_results["avg_structural_validity_score"] = round(np.mean([r.get('validity_score', 0.0) for r in valid_dag_evals]), 4)
            aggregated_evaluation_results["num_perfect_structural_validity"] = sum(1 for r in valid_dag_evals if r.get('validity_score') == 1.0)
            # Path statistics also rely on being a DAG
            aggregated_evaluation_results["avg_fraction_pis_connected"] = round(np.mean([r.get('fraction_pis_connected', 0.0) for r in valid_dag_evals]), 4)
            aggregated_evaluation_results["avg_fraction_pos_connected"] = round(np.mean([r.get('fraction_pos_connected', 0.0) for r in valid_dag_evals]), 4)
            aggregated_evaluation_results["avg_pis_reaching_po"] = round(np.mean([r.get('num_pis_reaching_po', 0) for r in valid_dag_evals]), 2)
            aggregated_evaluation_results["avg_pos_reachable_from_pi"] = round(np.mean([r.get('num_pos_reachable_from_pi', 0) for r in valid_dag_evals]), 2)
            # Perfect SEADAG might only be meaningful if also a DAG
            aggregated_evaluation_results["num_perfect_seadag_validity"] = sum(1 for r in valid_dag_evals if r.get('seadag_validity') == 1.0)
        else:
            logger.warning("No valid DAGs found among successful evaluations to calculate DAG-dependent averages.")
            # Add placeholder keys if desired
            aggregated_evaluation_results["avg_structural_validity_score"] = 0.0
            # ... add placeholders for other DAG-dependent metrics ...

    else:
         logger.warning("No evaluations completed without exceptions.")
         # Add placeholder keys if desired
         aggregated_evaluation_results["evaluated_dag_rate"] = 0.0
         aggregated_evaluation_results["avg_structural_validity_score"] = 0.0
         # ... add placeholders for other metrics ...


    # --- Comparison Metrics (MMD, Averages vs Test Set) ---
    # This part should remain unchanged as it seemed to be working
    if not test_graphs:
        logger.warning("Test dataset is empty or failed to load. Skipping comparison metrics.")
    elif not generated_graphs:
        logger.warning("Generated graph list is empty. Skipping comparison metrics.")
    else:
        logger.info(
            f"Calculating comparison metrics between {len(generated_graphs)} generated and {len(test_graphs)} test graphs...")
        comp_metric_start_time = time.time()
        try:
            # Define MMD functions
            mmd_stanford_fn_is_hist = lambda x, y: compute_mmd(x, y, kernel=gaussian_emd, is_hist=True, sigma=1.0)
            mmd_stanford_fn_is_hist_clustering_settings = lambda x, y: compute_mmd(x, y, kernel=gaussian_emd, is_hist=True, sigma=1.0/10, distance_scaling=100.0)
            mmd_stanford_fn_orbit_settings = lambda x, y: compute_mmd(x, y, kernel=gaussian, is_hist=False, sigma=30.0) # Requires Orca

            # Calculate MMD Metrics (keep try-except blocks)
            try:
                mmd_deg = compare_graphs_mmd_degree(generated_graphs, test_graphs, mmd_stanford_fn_is_hist)
                aggregated_evaluation_results["mmd_degree_distribution"] = round(mmd_deg, 6)
            except Exception as e: logger.error(f"Error calculating MMD Degree: {e}", exc_info=False)
            try:
                mmd_clus = compare_graphs_mmd_clustering_coeff(generated_graphs, test_graphs, mmd_stanford_fn_is_hist_clustering_settings)
                aggregated_evaluation_results["mmd_clustering_coeff"] = round(mmd_clus, 6)
            except Exception as e: logger.error(f"Error calculating MMD Clustering Coeff: {e}", exc_info=False)
            try:
                mmd_orbit = compare_graphs_mmd_orbit_stats(generated_graphs, test_graphs, mmd_stanford_fn_orbit_settings)
                aggregated_evaluation_results["mmd_orbit_stats"] = round(mmd_orbit, 6)
            except FileNotFoundError: logger.error("Orca executable not found. Skipping MMD Orbit Stats.")
            except Exception as e: logger.error(f"Error calculating MMD Orbit Stats: {e}", exc_info=False)

            # Calculate Average Metrics (keep helper and calls)
            def calculate_avg_metric(graph_list, metric_func, metric_name):
                # ... (keep implementation from previous versions) ...
                values = []
                for g_idx, g in enumerate(graph_list):
                     if not isinstance(g, nx.Graph) or g.number_of_nodes() == 0: continue
                     try:
                         metric_val = metric_func(g)
                         if isinstance(metric_val, dict):
                             dict_vals = list(metric_val.values())
                             if dict_vals: values.append(statistics.mean(dict_vals))
                         else: values.append(metric_val)
                     except Exception as e: logger.warning(f"Could not calculate {metric_name} for graph index {g_idx}: {e}")
                return round(statistics.mean(values), 4) if values else 0.0

            from graph_metrics import average_degree # Ensure import
            aggregated_evaluation_results["avg_degree_gen"] = calculate_avg_metric(generated_graphs, average_degree, "Avg Degree (Gen)")
            aggregated_evaluation_results["avg_clustering_coeff_gen"] = calculate_avg_metric(generated_graphs, nx.average_clustering, "Avg Clustering (Gen)")
            aggregated_evaluation_results["avg_density_gen"] = calculate_avg_metric(generated_graphs, nx.density, "Density (Gen)")
            aggregated_evaluation_results["avg_transitivity_gen"] = calculate_avg_metric(generated_graphs, nx.transitivity, "Transitivity (Gen)")
            aggregated_evaluation_results["avg_degree_test"] = calculate_avg_metric(test_graphs, average_degree, "Avg Degree (Test)")
            aggregated_evaluation_results["avg_clustering_coeff_test"] = calculate_avg_metric(test_graphs, nx.average_clustering, "Avg Clustering (Test)")
            aggregated_evaluation_results["avg_density_test"] = calculate_avg_metric(test_graphs, nx.density, "Density (Test)")
            aggregated_evaluation_results["avg_transitivity_test"] = calculate_avg_metric(test_graphs, nx.transitivity, "Transitivity (Test)")

            comp_metric_time = time.time() - comp_metric_start_time
            logger.info(f"Comparison metrics calculated in {comp_metric_time:.2f} seconds.")

        except Exception as e:
            logger.error(f"General error during comparison metric calculation: {e}", exc_info=True)
    # --- END: Comparison Metrics ---

    # Log summary (Combined)
    logger.info("--- Aggregated Evaluation Summary (Structural & Comparison) ---")
    for key in sorted(aggregated_evaluation_results.keys()): # Sort for consistency
        value = aggregated_evaluation_results[key]
        if isinstance(value, float): logger.info(f"  {key}: {value:.4f}")
        else: logger.info(f"  {key}: {value}")
    logger.info("---------------------------------------------------------------")

    # Return the aggregated dictionary and the list of detailed per-graph results
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