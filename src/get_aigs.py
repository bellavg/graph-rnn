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

        # Manually inject 'm' if missing
        if data_config.get('m') is None:
            known_m_value = 88
            data_config['m'] = known_m_value
            logger.warning(f"Manually injected 'm={known_m_value}' into loaded config for {model_path}.")

        input_size = data_config.get('m')
        if input_size is None: raise KeyError("Missing 'm' in data config.")

        # --- <<< MODIFIED: Determine Node Model Type using use_lstm >>> ---
        use_lstm_flag = model_config.get('use_lstm', False) # Default False if key missing
        use_node_attention = model_config.get('use_attention', False)
        node_model_type_str = 'lstm' if use_lstm_flag else 'gru' # Determine type based on flag
        # --- <<< END MODIFICATION >>> ---

        # Determine the correct config section name for the node model
        node_config_section_name = None
        if node_model_type_str == 'lstm':
             node_config_section_name = 'GraphAttentionLSTM' if use_node_attention else 'GraphLSTM'
        elif node_model_type_str == 'gru':
             node_config_section_name = 'GraphAttentionRNN' if use_node_attention else 'GraphRNN'
        else:
             # This case should not be reached now
             raise ValueError(f"Internal logic error determining node model type.")

        logger.info(f"Attempting to load node parameters from config section: 'model.{node_config_section_name}'")

        # Get the specific config dictionary for the node model
        node_config_dict_base = model_config.get(node_config_section_name)
        if node_config_dict_base is None:
             # Add more specific fallbacks if needed, e.g., maybe attention params are in base LSTM/GRU section
             if use_node_attention and node_model_type_str == 'gru' and 'GraphRNN' in model_config:
                 logger.warning(f"Config section 'model.{node_config_section_name}' not found, falling back to 'GraphRNN'.")
                 node_config_dict_base = model_config.get('GraphRNN')
             elif use_node_attention and node_model_type_str == 'lstm' and 'GraphLSTM' in model_config:
                 logger.warning(f"Config section 'model.{node_config_section_name}' not found, falling back to 'GraphLSTM'.")
                 node_config_dict_base = model_config.get('GraphLSTM')
             else:
                  # If still not found after fallback, raise the error
                  raise ValueError(f"Required config section 'model.{node_config_section_name}' not found in checkpoint, and fallback failed.")

        # Ensure node_config_dict is a mutable dictionary
        node_config_dict = dict(node_config_dict_base) if node_config_dict_base else {}

        # Explicitly check for required parameters
        required_node_params = ['embedding_size', 'hidden_size', 'num_layers']
        missing_params = [p for p in required_node_params if p not in node_config_dict]
        if missing_params:
            raise KeyError(f"Missing required node parameters in config section '{node_config_section_name}': {missing_params}")

        # Manually inject max_level if needed
        max_level_from_config = node_config_dict.get('max_level')
        if max_level_from_config is None:
            # Try getting max_level from data_config as alternative
            max_level_from_data = data_config.get('max_level')
            if max_level_from_data is not None:
                 node_config_dict['max_level'] = max_level_from_data
                 logger.warning(f"Injected 'max_level={max_level_from_data}' from data_config.")
            else:
                 # Fallback to hardcoded value if needed
                 known_max_level = 18
                 node_config_dict['max_level'] = known_max_level
                 logger.warning(f"Manually injected 'max_level={known_max_level}' into node_config_dict.")
        else:
            logger.info(f"Found max_level={max_level_from_config} in node config.")

        # --- Continue with edge model logic ---
        edge_model_type = model_config.get('edge_model', 'rnn').lower() # e.g., 'attention_rnn' from your config

        # --- <<< MODIFIED: Determine Edge Model Section Name >>> ---
        edge_config_section_name = None
        use_edge_attention = edge_model_type.startswith('attention') # Check if name indicates attention
        # Determine if edge model should be LSTM based on the NODE model type
        is_edge_lstm = use_lstm_flag # Edge model type matches node model type (LSTM/GRU)

        if is_edge_lstm:
             edge_config_section_name = 'EdgeAttentionLSTM' if use_edge_attention else 'EdgeLSTM'
        else: # GRU-based edge model
             edge_config_section_name = 'EdgeAttentionRNN' if use_edge_attention else 'EdgeRNN'

        logger.info(f"Attempting to load edge parameters from config section: 'model.{edge_config_section_name}'")
        edge_config_dict_base = model_config.get(edge_config_section_name)

        # Fallback logic for edge models (similar to node models)
        if edge_config_dict_base is None:
             if use_edge_attention and is_edge_lstm and 'EdgeLSTM' in model_config:
                 logger.warning(f"Edge config section '{edge_config_section_name}' not found, falling back to 'EdgeLSTM'.")
                 edge_config_dict_base = model_config.get('EdgeLSTM')
             elif use_edge_attention and not is_edge_lstm and 'EdgeRNN' in model_config:
                 logger.warning(f"Edge config section '{edge_config_section_name}' not found, falling back to 'EdgeRNN'.")
                 edge_config_dict_base = model_config.get('EdgeRNN')
             elif edge_model_type == 'mlp' and 'EdgeMLP' in model_config: # Check MLP separately
                  edge_config_section_name = 'EdgeMLP'
                  edge_config_dict_base = model_config.get('EdgeMLP')
                  logger.info(f"Using edge parameters from config section: 'model.EdgeMLP'")
             else:
                 # If still None after fallbacks (and not MLP), raise error
                 if edge_model_type != 'mlp':
                      raise ValueError(f"Required edge config section 'model.{edge_config_section_name}' not found, and fallback failed.")
                 elif 'EdgeMLP' not in model_config: # If it was MLP but section missing
                      raise ValueError(f"Required edge config section 'model.EdgeMLP' not found for edge_model='mlp'.")

        edge_config_dict = dict(edge_config_dict_base) if edge_config_dict_base else {}
        # --- <<< END MODIFICATION >>> ---

        # Get edge_feature_len from node config (should be reliable now)
        edge_feature_len = node_config_dict.get('edge_feature_len', NUM_EDGE_FEATURES)
        if 'edge_feature_len' not in node_config_dict: # Check just in case
             logger.warning(f"edge_feature_len not found in node config, using default {NUM_EDGE_FEATURES}")


        logger.info(f"Model config interpreted as: node_model={node_model_type_str}, edge_model={edge_model_type}, "
                    f"node_attn={use_node_attention}, edge_attn={use_edge_attention}, "
                    f"m={input_size}, edge_features={edge_feature_len}")

        # Ensure edge_feature_len is passed down
        node_config_dict['edge_feature_len'] = edge_feature_len
        edge_config_dict['edge_feature_len'] = edge_feature_len

    except KeyError as e:
         logger.error(f"Missing key during config parsing: {e}. Check checkpoint structure.", exc_info=True)
         raise ValueError(f"Missing key in config: {e}")
    except ValueError as e: # Catch ValueErrors raised above
         logger.error(f"Configuration error: {e}")
         raise # Re-raise the ValueError
    except Exception as e:
        logger.error(f"Unexpected error parsing config: {e}", exc_info=True)
        raise

    # --- Instantiate Models ---
    if not MODEL_CLASSES_LOADED:
        raise RuntimeError("Model classes could not be imported.")

    node_model = None
    edge_model = None
    edge_gen_function = None

    # --- Node Model Instantiation ---
    NodeModelClass = None
    if node_model_type_str == 'lstm':
        NodeModelClass = GraphLevelAttentionLSTM if use_node_attention else GraphLevelLSTM
    elif node_model_type_str == 'gru':
        NodeModelClass = GraphLevelAttentionRNN if use_node_attention else GraphLevelRNN

    if NodeModelClass is None: # Safety check
        raise RuntimeError(f"Could not determine NodeModelClass for type '{node_model_type_str}'.")

    logger.info(f"Using Node Model Class: {NodeModelClass.__name__}")

    # Determine node output size required by edge model BEFORE filtering node config
    node_output_size_for_edge = None
    if edge_model_type in ['rnn', 'lstm', 'attention_rnn', 'attention_lstm']:
        if 'hidden_size' not in edge_config_dict:
             raise ValueError(f"Edge model config section '{edge_config_section_name}' missing 'hidden_size'.")
        node_output_size_for_edge = edge_config_dict.get('hidden_size')
        node_config_dict['output_size'] = node_output_size_for_edge

    # Filter node config keys AFTER potentially adding output_size
    sig_node = inspect.signature(NodeModelClass.__init__)
    valid_keys_node = {p for p in sig_node.parameters if p != 'self'}
    node_config_dict['input_size'] = input_size # Ensure m is passed
    # Add attention params if needed by class
    if use_node_attention:
         node_config_dict['attention_heads'] = node_config_dict.get('attention_heads', 4)
         node_config_dict['attention_dropout'] = node_config_dict.get('attention_dropout', 0.1)

    filtered_node_config = {k: v for k, v in node_config_dict.items() if k in valid_keys_node}
    logger.debug(f"Node Model filtered config for {NodeModelClass.__name__}: {filtered_node_config}")

    try:
        node_model = NodeModelClass(**filtered_node_config).to(device)
    except TypeError as e:
        logger.error(f"TypeError instantiating {NodeModelClass.__name__}: {e}. Config: {filtered_node_config}", exc_info=True)
        raise

    # --- Edge Model Instantiation ---
    EdgeModelClass = None
    if edge_model_type == 'attention_rnn': # From your config
         edge_gen_function = rnn_edge_gen # Assuming rnn_edge_gen works for attention models
         if is_edge_lstm: # Check if node was LSTM
              EdgeModelClass = EdgeLevelAttentionLSTM
         else: # Node was GRU
              EdgeModelClass = EdgeLevelAttentionRNN # <<< This seems correct based on your config
         logger.info(f"Using Edge Model Class: {EdgeModelClass.__name__}")
         # Add default attention params if missing
         edge_config_dict['attention_heads'] = edge_config_dict.get('attention_heads', 4)
         edge_config_dict['attention_dropout'] = edge_config_dict.get('attention_dropout', 0.1)

    elif edge_model_type == 'rnn':
         edge_gen_function = rnn_edge_gen
         EdgeModelClass = EdgeLevelLSTM if is_edge_lstm else EdgeLevelRNN
         logger.info(f"Using Edge Model Class: {EdgeModelClass.__name__}")

    elif edge_model_type == 'mlp':
         edge_gen_function = mlp_edge_gen
         EdgeModelClass = EdgeLevelMLP
         logger.info(f"Using Edge Model Class: {EdgeModelClass.__name__}")
         # Set required MLP params
         if 'hidden_size' not in node_config_dict: raise ValueError("Node config missing 'hidden_size' for MLP input")
         edge_config_dict['input_size'] = node_config_dict['hidden_size']
         edge_config_dict['output_size'] = input_size # m
         # Check if edge MLP config itself has hidden_size
         if 'hidden_size' not in edge_config_dict: raise ValueError("EdgeMLP config missing 'hidden_size'.")

    # Add cases for 'attention_lstm' and 'lstm' if needed
    elif edge_model_type == 'attention_lstm':
         edge_gen_function = rnn_edge_gen # Or specific LSTM gen func?
         if not is_edge_lstm: logger.warning("edge_model=attention_lstm but node model is GRU based on use_lstm flag.")
         EdgeModelClass = EdgeLevelAttentionLSTM
         logger.info(f"Using Edge Model Class: {EdgeModelClass.__name__}")
         edge_config_dict['attention_heads'] = edge_config_dict.get('attention_heads', 4)
         edge_config_dict['attention_dropout'] = edge_config_dict.get('attention_dropout', 0.1)

    elif edge_model_type == 'lstm':
         edge_gen_function = rnn_edge_gen # Or specific LSTM gen func?
         if not is_edge_lstm: logger.warning("edge_model=lstm but node model is GRU based on use_lstm flag.")
         EdgeModelClass = EdgeLevelLSTM
         logger.info(f"Using Edge Model Class: {EdgeModelClass.__name__}")

    else:
        raise ValueError(f"Unsupported edge_model type '{edge_model_type}' needs instantiation logic.")


    # Filter edge config and instantiate
    if EdgeModelClass:
        sig_edge = inspect.signature(EdgeModelClass.__init__)
        valid_keys_edge = {p for p in sig_edge.parameters if p != 'self'}
        # Ensure hidden_size is present if needed by RNN/LSTM models
        if edge_model_type != 'mlp' and 'hidden_size' not in edge_config_dict:
            raise ValueError(f"Edge config for {EdgeModelClass.__name__} missing 'hidden_size'.")

        filtered_edge_config = {k: v for k, v in edge_config_dict.items() if k in valid_keys_edge}
        logger.debug(f"Edge Model filtered config for {EdgeModelClass.__name__}: {filtered_edge_config}")
        try:
            edge_model = EdgeModelClass(**filtered_edge_config).to(device)
        except TypeError as e:
            logger.error(f"TypeError instantiating {EdgeModelClass.__name__}: {e}. Config: {filtered_edge_config}", exc_info=True)
            raise
    else:
        # This should not happen if logic above is complete
        raise RuntimeError(f"Could not determine EdgeModelClass for edge_model type '{edge_model_type}'.")

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
         # Provide more context on mismatch error
         logger.error(f"Error loading state_dict (likely mismatched model architecture or keys): {e}", exc_info=True)
         logger.error(f"Node model expected keys example: {list(node_model.state_dict().keys())[:5]}")
         logger.error(f"Node state_dict loaded keys example: {list(node_state_dict.keys())[:5]}")
         logger.error(f"Edge model expected keys example: {list(edge_model.state_dict().keys())[:5]}")
         logger.error(f"Edge state_dict loaded keys example: {list(edge_state_dict.keys())[:5]}")
         raise

    mode = model_config.get('mode', 'aig') # Get generation mode

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
    #test_graphs: List[nx.DiGraph],
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

    #
    # # --- Comparison Metrics (MMD, Averages vs Test Set) ---
    # # This part should remain unchanged as it seemed to be working
    # if not test_graphs:
    #     logger.warning("Test dataset is empty or failed to load. Skipping comparison metrics.")
    # elif not generated_graphs:
    #     logger.warning("Generated graph list is empty. Skipping comparison metrics.")
    # else:
    #     logger.info(
    #         f"Calculating comparison metrics between {len(generated_graphs)} generated and {len(test_graphs)} test graphs...")
    #     comp_metric_start_time = time.time()
    #     try:
    #         # Define MMD functions
    #         mmd_stanford_fn_is_hist = lambda x, y: compute_mmd(x, y, kernel=gaussian_emd, is_hist=True, sigma=1.0)
    #         mmd_stanford_fn_is_hist_clustering_settings = lambda x, y: compute_mmd(x, y, kernel=gaussian_emd, is_hist=True, sigma=1.0/10, distance_scaling=100.0)
    #         mmd_stanford_fn_orbit_settings = lambda x, y: compute_mmd(x, y, kernel=gaussian, is_hist=False, sigma=30.0) # Requires Orca
    #
    #         # Calculate MMD Metrics (keep try-except blocks)
    #         try:
    #             mmd_deg = compare_graphs_mmd_degree(generated_graphs, test_graphs, mmd_stanford_fn_is_hist)
    #             aggregated_evaluation_results["mmd_degree_distribution"] = round(mmd_deg, 6)
    #         except Exception as e: logger.error(f"Error calculating MMD Degree: {e}", exc_info=False)
    #         try:
    #             mmd_clus = compare_graphs_mmd_clustering_coeff(generated_graphs, test_graphs, mmd_stanford_fn_is_hist_clustering_settings)
    #             aggregated_evaluation_results["mmd_clustering_coeff"] = round(mmd_clus, 6)
    #         except Exception as e: logger.error(f"Error calculating MMD Clustering Coeff: {e}", exc_info=False)
    #         try:
    #             mmd_orbit = compare_graphs_mmd_orbit_stats(generated_graphs, test_graphs, mmd_stanford_fn_orbit_settings)
    #             aggregated_evaluation_results["mmd_orbit_stats"] = round(mmd_orbit, 6)
    #         except FileNotFoundError: logger.error("Orca executable not found. Skipping MMD Orbit Stats.")
    #         except Exception as e: logger.error(f"Error calculating MMD Orbit Stats: {e}", exc_info=False)
    #
    #         # Calculate Average Metrics (keep helper and calls)
    #         def calculate_avg_metric(graph_list, metric_func, metric_name):
    #             # ... (keep implementation from previous versions) ...
    #             values = []
    #             for g_idx, g in enumerate(graph_list):
    #                  if not isinstance(g, nx.Graph) or g.number_of_nodes() == 0: continue
    #                  try:
    #                      metric_val = metric_func(g)
    #                      if isinstance(metric_val, dict):
    #                          dict_vals = list(metric_val.values())
    #                          if dict_vals: values.append(statistics.mean(dict_vals))
    #                      else: values.append(metric_val)
    #                  except Exception as e: logger.warning(f"Could not calculate {metric_name} for graph index {g_idx}: {e}")
    #             return round(statistics.mean(values), 4) if values else 0.0
    #
    #         from graph_metrics import average_degree # Ensure import
    #         aggregated_evaluation_results["avg_degree_gen"] = calculate_avg_metric(generated_graphs, average_degree, "Avg Degree (Gen)")
    #         aggregated_evaluation_results["avg_clustering_coeff_gen"] = calculate_avg_metric(generated_graphs, nx.average_clustering, "Avg Clustering (Gen)")
    #         aggregated_evaluation_results["avg_density_gen"] = calculate_avg_metric(generated_graphs, nx.density, "Density (Gen)")
    #         aggregated_evaluation_results["avg_transitivity_gen"] = calculate_avg_metric(generated_graphs, nx.transitivity, "Transitivity (Gen)")
    #         aggregated_evaluation_results["avg_degree_test"] = calculate_avg_metric(test_graphs, average_degree, "Avg Degree (Test)")
    #         aggregated_evaluation_results["avg_clustering_coeff_test"] = calculate_avg_metric(test_graphs, nx.average_clustering, "Avg Clustering (Test)")
    #         aggregated_evaluation_results["avg_density_test"] = calculate_avg_metric(test_graphs, nx.density, "Density (Test)")
    #         aggregated_evaluation_results["avg_transitivity_test"] = calculate_avg_metric(test_graphs, nx.transitivity, "Transitivity (Test)")
    #
    #         comp_metric_time = time.time() - comp_metric_start_time
    #         logger.info(f"Comparison metrics calculated in {comp_metric_time:.2f} seconds.")
    #
    #     except Exception as e:
    #         logger.error(f"General error during comparison metric calculation: {e}", exc_info=True)
    # # --- END: Comparison Metrics ---

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
    num_graphs_to_generate: int = 500,
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


    # # 2. Load Test Dataset (if evaluating)
    # test_graphs = []
    # if evaluate:
    #     try:
    #         logger.info(f"Loading test dataset from: {graph_file}")
    #         # Use train_split from the loaded config if possible, else default
    #         train_split_from_config = config.get('data', {}).get('train_split', 0.9) # Default 0.9
    #
    #         test_dataset = AIGDataset(
    #             graph_file=graph_file,
    #             training=False, # Load the test split
    #             train_split=train_split_from_config,
    #             # Add other relevant AIGDataset params if needed (e.g., max_graphs from config?)
    #             # max_graphs=config.get('data', {}).get('max_graphs'), # Example
    #         )
    #         # Retrieve the actual graph objects for the test split
    #         if hasattr(test_dataset, 'graphs') and test_dataset.graphs is not None:
    #              test_graphs = [g for g in test_dataset.graphs if isinstance(g, nx.DiGraph)] # Ensure they are graphs
    #              if num_test_graphs is not None and 0 < num_test_graphs < len(test_graphs):
    #                  logger.info(f"Randomly sampling {num_test_graphs} graphs from the test set for evaluation...")
    #                  try:
    #                      test_graphs = random.sample(test_graphs, num_test_graphs)
    #                      logger.info(f"Using {len(test_graphs)} sampled test graphs.")
    #                  except ValueError as e:
    #                      logger.error(f"Error during random sampling: {e}. Using full test set.")
    #              elif num_test_graphs is not None:
    #                  logger.warning(
    #                      f"--num-test-graphs value ({num_test_graphs}) is invalid or >= total test graphs. Using all {len(test_graphs)} test graphs.")
    #              else:
    #                  logger.info(f"Using all {len(test_graphs)} loaded test graphs for evaluation.")
    #         else:
    #              test_graphs = []
    #
    #         logger.info(f"Loaded {len(test_graphs)} graphs for test set.")
    #         if not test_graphs:
    #              logger.warning("Test dataset is empty. Comparison metrics will be skipped.")
    #              # Decide if this should be a fatal error or just skip comparisons
    #              # evaluate = False # Option: Disable evaluation if test set empty
    #
    #     except FileNotFoundError:
    #         logger.error(f"Test dataset file not found: {graph_file}. Comparison metrics will be skipped.")
    #         test_graphs = []
    #     except Exception as e:
    #         logger.error(f"Error loading test dataset: {e}. Comparison metrics will be skipped.", exc_info=True)
    #         test_graphs = []


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
    aggregated_evaluation_results = {}
    evaluation_details_list = []
    if evaluate:  # Check if evaluation is enabled (via args.evaluate)
        logger.info(
            f"Evaluation flag is set, preparing to evaluate all {num_successfully_generated} generated graphs structurally.")
        # <<< CHANGE HERE: Pass num_successfully_generated as the count to evaluate >>>
        aggregated_evaluation_results, evaluation_details_list = get_evaluation(
            generated_graphs=generated_graphs,
            #test_graphs=test_graphs,
            # Use the actual number generated for structural evaluation count
            num_graphs_to_evaluate=num_successfully_generated,  # <<< Use this value
            num_successfully_generated=num_successfully_generated
        )
        # <<< END CHANGE >>>

        # Merge evaluation results into main results dict
        aggregated_results.update(aggregated_evaluation_results)
        # Optionally add the detailed list if saving it
        # aggregated_results["evaluation_details_per_graph"] = evaluation_details_list
    else:
        logger.info("Evaluation step skipped as per request.")
        # You might want to explicitly add keys indicating skipped eval to results
        aggregated_results["num_graphs_structurally_evaluated"] = 0
        aggregated_results["num_structural_evaluation_exceptions"] = 0


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
    parser.add_argument("--num-generate", type=int, default=500, # Renamed from num-graphs
                        help="Number of AIGs to attempt generating.")
    parser.add_argument("--num-test-graphs", type=int, default=500,
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
        'max_nodes': 80, # Example, adjust as needed
        'min_nodes': 8,   # Example
        'patience': 16,   # Example
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