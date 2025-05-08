# get_aigs.py (Generation Only Version)
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
from typing import Dict, Any, List, Tuple, Optional, Callable
import pickle # Added for saving graphs

# --- Import from project files ---
# Model imports
try:
    from src.model import (GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP,
                       GraphLevelAttentionRNN, EdgeLevelAttentionRNN,
                       GraphLevelLSTM, EdgeLevelLSTM,
                       GraphLevelAttentionLSTM, EdgeLevelAttentionLSTM)
    MODEL_CLASSES_LOADED = True
except ImportError as e:
    print(f"Error importing model classes: {e}. Ensure model.py is accessible.")
    MODEL_CLASSES_LOADED = False
    sys.exit(1) # Exit if models are critical

# Generation imports (Ensure generate_aigs.py is the updated version)
try:
    # Import the main generate function and necessary helpers/constants
    from src.generate_aigs import generate, rnn_edge_gen, mlp_edge_gen, EDGE_TYPES, NUM_EDGE_FEATURES
except ImportError as e:
    print(f"Error importing from generate_aigs.py: {e}. Ensure it exists and is accessible.")
    sys.exit(1)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, # Set default level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to stdout
)
# Get logger for this main script
logger = logging.getLogger("get_aigs_main")

# --- Model Loading Function ---
# (load_model_from_config function remains the same as provided previously)
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
            known_m_value = 63 # Default or determine dynamically if possible
            data_config['m'] = known_m_value
            logger.warning(f"Manually injected 'm={known_m_value}' into loaded config for {model_path}.")

        input_size = data_config.get('m')
        if input_size is None: raise KeyError("Missing 'm' in data config.")

        # Determine Node Model Type using use_lstm
        use_lstm_flag = model_config.get('use_lstm', False)
        use_node_attention = model_config.get('use_attention', False)
        node_model_type_str = 'lstm' if use_lstm_flag else 'gru'

        # Determine the correct config section name for the node model
        node_config_section_name = None
        if node_model_type_str == 'lstm':
             node_config_section_name = 'GraphAttentionLSTM' if use_node_attention else 'GraphLSTM'
        elif node_model_type_str == 'gru':
             node_config_section_name = 'GraphAttentionRNN' if use_node_attention else 'GraphRNN'
        else:
             raise ValueError(f"Internal logic error determining node model type.")

        logger.info(f"Attempting to load node parameters from config section: 'model.{node_config_section_name}'")
        node_config_dict_base = model_config.get(node_config_section_name)
        if node_config_dict_base is None:
             # Fallback logic
             if use_node_attention and node_model_type_str == 'gru' and 'GraphRNN' in model_config:
                 logger.warning(f"Config section 'model.{node_config_section_name}' not found, falling back to 'GraphRNN'.")
                 node_config_dict_base = model_config.get('GraphRNN')
             elif use_node_attention and node_model_type_str == 'lstm' and 'GraphLSTM' in model_config:
                 logger.warning(f"Config section 'model.{node_config_section_name}' not found, falling back to 'GraphLSTM'.")
                 node_config_dict_base = model_config.get('GraphLSTM')
             else:
                  raise ValueError(f"Required config section 'model.{node_config_section_name}' not found in checkpoint, and fallback failed.")

        node_config_dict = dict(node_config_dict_base) if node_config_dict_base else {}

        # Check required node params
        required_node_params = ['embedding_size', 'hidden_size', 'num_layers']
        missing_params = [p for p in required_node_params if p not in node_config_dict]
        if missing_params:
            raise KeyError(f"Missing required node parameters in config section '{node_config_section_name}': {missing_params}")

        node_config_dict['max_level'] = 22
        logger.info(f"Found max_level={node_config_dict['max_level']} in node config.")



        # Determine Edge Model Config Section
        edge_model_type = model_config.get('edge_model', 'rnn').lower()
        edge_config_section_name = None
        use_edge_attention = edge_model_type.startswith('attention')
        is_edge_lstm = use_lstm_flag # Edge model type matches node model type

        if is_edge_lstm:
             edge_config_section_name = 'EdgeAttentionLSTM' if use_edge_attention else 'EdgeLSTM'
        else: # GRU-based edge model
             edge_config_section_name = 'EdgeAttentionRNN' if use_edge_attention else 'EdgeRNN'

        logger.info(f"Attempting to load edge parameters from config section: 'model.{edge_config_section_name}'")
        edge_config_dict_base = model_config.get(edge_config_section_name)

        # Fallback logic for edge models
        if edge_config_dict_base is None:
             if use_edge_attention and is_edge_lstm and 'EdgeLSTM' in model_config:
                 logger.warning(f"Edge config section '{edge_config_section_name}' not found, falling back to 'EdgeLSTM'.")
                 edge_config_dict_base = model_config.get('EdgeLSTM')
             elif use_edge_attention and not is_edge_lstm and 'EdgeRNN' in model_config:
                 logger.warning(f"Edge config section '{edge_config_section_name}' not found, falling back to 'EdgeRNN'.")
                 edge_config_dict_base = model_config.get('EdgeRNN')
             elif edge_model_type == 'mlp' and 'EdgeMLP' in model_config:
                  edge_config_section_name = 'EdgeMLP'
                  edge_config_dict_base = model_config.get('EdgeMLP')
                  logger.info(f"Using edge parameters from config section: 'model.EdgeMLP'")
             else:
                 if edge_model_type != 'mlp':
                      raise ValueError(f"Required edge config section 'model.{edge_config_section_name}' not found, and fallback failed.")
                 elif 'EdgeMLP' not in model_config:
                      raise ValueError(f"Required edge config section 'model.EdgeMLP' not found for edge_model='mlp'.")

        edge_config_dict = dict(edge_config_dict_base) if edge_config_dict_base else {}

        # Get edge_feature_len from node config
        edge_feature_len = node_config_dict.get('edge_feature_len', NUM_EDGE_FEATURES)
        if 'edge_feature_len' not in node_config_dict:
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
    except ValueError as e:
         logger.error(f"Configuration error: {e}")
         raise
    except Exception as e:
        logger.error(f"Unexpected error parsing config: {e}", exc_info=True)
        raise

    # --- Instantiate Models ---
    if not MODEL_CLASSES_LOADED:
        raise RuntimeError("Model classes could not be imported.")

    node_model = None
    edge_model = None
    edge_gen_function = None

    # Node Model Instantiation
    NodeModelClass = None
    if node_model_type_str == 'lstm':
        NodeModelClass = GraphLevelAttentionLSTM if use_node_attention else GraphLevelLSTM
    elif node_model_type_str == 'gru':
        NodeModelClass = GraphLevelAttentionRNN if use_node_attention else GraphLevelRNN

    if NodeModelClass is None:
        raise RuntimeError(f"Could not determine NodeModelClass for type '{node_model_type_str}'.")

    logger.info(f"Using Node Model Class: {NodeModelClass.__name__}")

    node_output_size_for_edge = None
    if edge_model_type in ['rnn', 'lstm', 'attention_rnn', 'attention_lstm']:
        if 'hidden_size' not in edge_config_dict:
             raise ValueError(f"Edge model config section '{edge_config_section_name}' missing 'hidden_size'.")
        node_output_size_for_edge = edge_config_dict.get('hidden_size')
        node_config_dict['output_size'] = node_output_size_for_edge

    sig_node = inspect.signature(NodeModelClass.__init__)
    valid_keys_node = {p for p in sig_node.parameters if p != 'self'}
    node_config_dict['input_size'] = input_size
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

    # Edge Model Instantiation
    EdgeModelClass = None
    if edge_model_type == 'attention_rnn':
         edge_gen_function = rnn_edge_gen
         EdgeModelClass = EdgeLevelAttentionLSTM if is_edge_lstm else EdgeLevelAttentionRNN
         logger.info(f"Using Edge Model Class: {EdgeModelClass.__name__}")
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
         if 'hidden_size' not in node_config_dict: raise ValueError("Node config missing 'hidden_size' for MLP input")
         edge_config_dict['input_size'] = node_config_dict['hidden_size']
         edge_config_dict['output_size'] = input_size
         if 'hidden_size' not in edge_config_dict: raise ValueError("EdgeMLP config missing 'hidden_size'.")
    elif edge_model_type == 'attention_lstm':
         edge_gen_function = rnn_edge_gen
         if not is_edge_lstm: logger.warning("edge_model=attention_lstm but node model is GRU based on use_lstm flag.")
         EdgeModelClass = EdgeLevelAttentionLSTM
         logger.info(f"Using Edge Model Class: {EdgeModelClass.__name__}")
         edge_config_dict['attention_heads'] = edge_config_dict.get('attention_heads', 4)
         edge_config_dict['attention_dropout'] = edge_config_dict.get('attention_dropout', 0.1)
    elif edge_model_type == 'lstm':
         edge_gen_function = rnn_edge_gen
         if not is_edge_lstm: logger.warning("edge_model=lstm but node model is GRU based on use_lstm flag.")
         EdgeModelClass = EdgeLevelLSTM
         logger.info(f"Using Edge Model Class: {EdgeModelClass.__name__}")
    else:
        raise ValueError(f"Unsupported edge_model type '{edge_model_type}' needs instantiation logic.")

    if EdgeModelClass:
        sig_edge = inspect.signature(EdgeModelClass.__init__)
        valid_keys_edge = {p for p in sig_edge.parameters if p != 'self'}
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
        raise RuntimeError(f"Could not determine EdgeModelClass for edge_model type '{edge_model_type}'.")

    # Load State Dicts
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
         logger.error(f"Node model expected keys example: {list(node_model.state_dict().keys())[:5]}")
         logger.error(f"Node state_dict loaded keys example: {list(node_state_dict.keys())[:5]}")
         logger.error(f"Edge model expected keys example: {list(edge_model.state_dict().keys())[:5]}")
         logger.error(f"Edge state_dict loaded keys example: {list(edge_state_dict.keys())[:5]}")
         raise

    mode = model_config.get('mode', 'aig') # Get generation mode

    return node_model, edge_model, input_size, edge_gen_function, mode, edge_feature_len


# --- MODIFIED Helper Function: Generation ---
def get_generation(
    num_graphs_to_generate: int,
    models_tuple: Tuple[torch.nn.Module, torch.nn.Module, int, Callable, str, int],
    gen_params: Dict[str, Any]
    ) -> Tuple[List[Optional[nx.DiGraph]], int]: # Returns List of Graphs (or None)
    """
    Generates graphs using the loaded models.
    Returns a list of NetworkX graphs (or None for failed generations) and the count of successes.
    """
    (node_model, edge_model, input_size, edge_gen_function,
     mode, edge_feature_len) = models_tuple

    logger.debug(f"Generating {num_graphs_to_generate} AIGs...")
    generated_graphs_list: List[Optional[nx.DiGraph]] = [] # Store nx.DiGraph or None

    current_gen_params = {
        **gen_params, # Includes max_nodes, min_nodes, patience etc.
        'temperature': gen_params.get('temperature', 1.0),
        'top_k': gen_params.get('top_k', 0),
        'top_p': gen_params.get('top_p', 0.0),
        'edge_sample_attempts': gen_params.get('edge_sample_attempts', 1)
    }

    generation_successful_count = 0
    for i in range(num_graphs_to_generate):
        logger.debug(f"Generating graph {i+1}/{num_graphs_to_generate}...")
        try:
            # --- Call generate function which now returns nx.DiGraph or None ---
            graph = generate(
                node_model=node_model, edge_model=edge_model,
                input_size=input_size, edge_gen_function=edge_gen_function,
                mode=mode, edge_feature_len=edge_feature_len,
                # Pass generation parameters from current_gen_params
                **current_gen_params
            )
            # --- End Call ---

            if isinstance(graph, nx.DiGraph) and graph.number_of_nodes() > 0:
                 generation_successful_count += 1
                 # Optionally pre-calculate inferred types if needed later
                 # try:
                 #     graph.graph['_inferred_types_cleaned'] = infer_node_types(graph)
                 # except Exception as infer_e:
                 #     logger.warning(f"Could not pre-calculate node types for graph {i+1}: {infer_e}")
            else:
                logger.warning(f"Generation {i+1} resulted in None or empty graph.")
                graph = None # Ensure it's None if generation failed

            generated_graphs_list.append(graph) # Append graph or None

        except Exception as e:
            logger.error(f"Error during generation of graph {i+1}: {e}", exc_info=True)
            generated_graphs_list.append(None) # Mark as failed

    num_successfully_generated = generation_successful_count
    logger.info(f"Finished generation. Generated {num_successfully_generated}/{num_graphs_to_generate} non-empty graphs.")

    # Return the list containing graphs or None, and the success count
    return generated_graphs_list, num_successfully_generated
# --- End MODIFIED Helper Function ---

# --- REMOVED get_visualization function ---
# --- REMOVED get_evaluation function ---

# --- MODIFIED: Main Control Function ---
def aig_control(
        model_checkpoint_path: str,
        output_dir: str,
        gen_params: Dict[str, Any],
        # graph_file: str, # Removed - not needed for generation only
        num_graphs_to_generate: int = 500,
        # evaluate: bool = True, # Removed
        # visualize: bool = True, # Removed
        # num_graphs_to_visualize: int = 5, # Removed
        # num_test_graphs: Optional[int] = None, # Removed
        output_graphs_filename: str = "generated_graphs.pkl"
) -> Dict[str, Any]:
    """
    Main control function to load model, generate AIGs,
    and save the generated graphs to a PKL file.
    Removes evaluation and visualization steps.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    # viz_dir removed

    logger.info(f"Starting AIG GENERATION ONLY process. Output dir: {output_dir}")
    # logger.info(f"Graph file (for potential comparison): {graph_file}") # Removed
    logger.info(f"Requested Generations: {num_graphs_to_generate}")
    logger.info(f"Base Generation Parameters: {gen_params}")

    # --- Initialize Aggregated Results (Simplified) ---
    aggregated_results: Dict[str, Any] = {
        "model_checkpoint": model_checkpoint_path,
        "generation_params_base": gen_params,
        "output_directory": output_dir,
        "requested_generations": num_graphs_to_generate,
        # "graph_file_used": graph_file, # Removed
        "generated_graphs_file": os.path.join(output_dir, output_graphs_filename)
    }

    # 1. Load Model
    logger.info(f"Loading model from {model_checkpoint_path}...")
    try:
        models_tuple = load_model_from_config(model_checkpoint_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        aggregated_results["error"] = f"Model loading failed: {e}"
        # Save partial summary and exit
        results_file = os.path.join(output_dir, "aig_summary.json")
        try:
            with open(results_file, "w") as f: json.dump(aggregated_results, f, indent=2)
        except Exception as save_e: logger.error(f"Failed to save error summary: {save_e}")
        return aggregated_results

    # 2. Generate Graphs
    generated_graphs_list, num_successfully_generated = get_generation(
        num_graphs_to_generate=num_graphs_to_generate,
        models_tuple=models_tuple,
        gen_params=gen_params
    )
    aggregated_results["num_graphs_generated"] = num_successfully_generated

    # Filter out None values for saving
    valid_generated_graphs = [g for g in generated_graphs_list if g is not None]

    # --- Early Exit or Warning if No Graphs Generated ---
    if not valid_generated_graphs:
        logger.warning("No valid graphs were generated successfully. Cannot save PKL file.")
        aggregated_results["warning"] = "No valid graphs generated successfully."
    else:
        # --- 3. Save Generated Graphs ---
        graphs_output_path = aggregated_results["generated_graphs_file"]
        logger.info(f"Saving {len(valid_generated_graphs)} generated graphs to {graphs_output_path}...")
        try:
            with open(graphs_output_path, 'wb') as f_out:
                pickle.dump(valid_generated_graphs, f_out)
            logger.info("Successfully saved generated graphs.")
        except Exception as e:
            logger.error(f"Failed to save generated graphs pkl file: {e}")
            aggregated_results["error_saving_graphs"] = str(e)
        # --- End Save ---

    # 4. Save Final Aggregated Results (JSON Summary)
    results_file = os.path.join(output_dir, "aig_summary.json")
    try:
        def convert_numpy(obj): # Helper for JSON serialization
            if isinstance(obj, (np.integer, np.int_)): return int(obj)
            elif isinstance(obj, (np.floating, np.float_)): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, (set, frozenset)): return list(obj)
            return obj

        serializable_results = json.loads(json.dumps(aggregated_results, default=convert_numpy))
        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Aggregated results summary saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results JSON file: {e}")

    # 5. Return Results
    total_time = time.time() - start_time
    logger.info(f"AIG generation process finished in {total_time:.2f} seconds.")
    aggregated_results['total_time_seconds'] = round(total_time, 2)

    return aggregated_results
# --- End MODIFIED Main Control Function ---

# --- MODIFIED Main Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AIGs using a trained model and save them.")

    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the generated graphs PKL file and summary JSON.")

    # Generation Control
    parser.add_argument("--num-generate", type=int, default=500,
                        help="Number of AIGs to attempt generating.")
    parser.add_argument("--output-graphs-file", type=str, default="generated_graphs.pkl",
                        help="Filename for saving the generated NetworkX graphs as a pickle file within the output directory.")

    # Generation Parameters (Optional Overrides)
    parser.add_argument('--gen-max-nodes', type=int, help="Override max_nodes for generation.")
    parser.add_argument('--gen-min-nodes', type=int, help="Override min_nodes for generation.")
    parser.add_argument('--gen-patience', type=int, help="Override patience for generation.")
    parser.add_argument('--gen-temp', type=float, help="Override temperature for sampling.")
    parser.add_argument('--gen-top-k', type=int, help="Override top_k for sampling.")
    parser.add_argument('--gen-top-p', type=float, help="Override top_p for sampling.")
    parser.add_argument('--gen-edge-attempts', type=int, help="Override edge_sample_attempts.")

    # REMOVED arguments related to evaluation and visualization
    # parser.add_argument("--evaluate", action='store_true', ...)
    # parser.add_argument("--visualize", action='store_true', ...)
    # parser.add_argument("--num-visualize", type=int, ...)
    # parser.add_argument("--graph-file", type=str, ...) # Removed reference graph file

    args = parser.parse_args()

    # --- Base Generation Config Dictionary ---
    GENERATION_CONFIG = {
        'max_nodes': args.gen_max_nodes if args.gen_max_nodes is not None else 80,
        'min_nodes': args.gen_min_nodes if args.gen_min_nodes is not None else 8,
        'patience': args.gen_patience if args.gen_patience is not None else 16,
        'temperature': args.gen_temp if args.gen_temp is not None else 1.2,
        'top_k': args.gen_top_k if args.gen_top_k is not None else 0,
        'top_p': args.gen_top_p if args.gen_top_p is not None else 0.0,
        'edge_sample_attempts': args.gen_edge_attempts if args.gen_edge_attempts is not None else 3,
    }

    logger.info("Starting AIG generation script with command line arguments...")

    # --- Call to simplified aig_control ---
    final_results = aig_control(
        model_checkpoint_path=args.model_path,
        output_dir=args.output_dir,
        gen_params=GENERATION_CONFIG,
        # graph_file=args.graph_file, # Removed
        num_graphs_to_generate=args.num_generate,
        # evaluate=args.evaluate, # Removed
        # visualize=args.visualize, # Removed
        # num_graphs_to_visualize=args.num_visualize, # Removed
        # num_test_graphs=args.num_test_graphs # Removed
        output_graphs_filename=args.output_graphs_file
    )

    logger.info("--- Final Control Function Summary ---")
    # Use default=str to handle potential non-serializable types like numpy numbers
    print(json.dumps(final_results, indent=2, default=str))
    logger.info("--- End of Script ---")
