# get_aigs.py (Generation Only Version - GRU/RNN Only, Node Prediction Setup)
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
import pickle

# --- Import from project files ---
# --- MODIFIED: Import only necessary models ---
try:
    from model import GraphLevelRNN, EdgeLevelRNN
    MODEL_CLASSES_LOADED = True
except ImportError as e:
    print(f"Error importing models from model.py: {e}. Ensure GraphLevelRNN and EdgeLevelRNN are defined.")
    sys.exit(1)
# --- END MODIFIED ---

# --- MODIFIED: Import only necessary generation functions ---
try:
    from generate_aigs import generate, rnn_edge_gen # Removed mlp_edge_gen
except ImportError as e:
    print(f"Error importing from generate_aigs.py: {e}. Ensure it exists and is accessible.")
    sys.exit(1)
# --- END MODIFIED ---




NUM_EDGE_FEATURES = 3 # Default internal edge features (None, Reg, Inv)
NUM_NODE_TYPES = 4    # Default node types (CONST0, PI, AND, PO)
# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("get_aigs_main")

# --- MODIFIED: Model Loading Function ---
def load_model_from_config(model_path):
    """
    Loads GraphLevelRNN and EdgeLevelRNN models based on config stored in checkpoint.
    Handles configuration for node type prediction.
    """
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
    logger.debug(f"Loaded config: {json.dumps(config, indent=2)}")

    # --- Extract Config ---
    try:
        data_config = config.get('data', {})
        model_config = config.get('model', {})
        train_config = config.get('train', {})

        # Get 'm' (effective input size)
        input_size = data_config.get('m')
        if input_size is None:
            max_node_count = data_config.get('max_node_count', 64) # Default if missing
            input_size = max_node_count - 1
            logger.warning(f"Config 'data.m' missing. Inferred m={input_size} from max_node_count={max_node_count}.")
        data_config['m'] = input_size

        # --- Get Node Prediction Config ---
        predict_node_types = train_config.get('predict_node_types', model_config.get('predict_node_types', False))
        num_node_types = NUM_NODE_TYPES if predict_node_types else None
        logger.info(f"Node type prediction enabled: {predict_node_types}")
        if predict_node_types: logger.info(f"Number of node types: {num_node_types}")
        # --- End Node Prediction Config ---

        # --- Simplified Model Config Loading (Assuming GRU/RNN) ---
        node_config_section_name = 'GraphRNN'
        edge_config_section_name = 'EdgeRNN'

        if node_config_section_name not in model_config:
             raise ValueError(f"Required config section 'model.GraphRNN' not found.")
        if edge_config_section_name not in model_config:
             raise ValueError(f"Required config section 'model.EdgeRNN' not found.")

        logger.info(f"Loading node parameters from config section: 'model.{node_config_section_name}'")
        node_config_dict = dict(model_config.get(node_config_section_name, {}))

        logger.info(f"Loading edge parameters from config section: 'model.{edge_config_section_name}'")
        edge_config_dict = dict(model_config.get(edge_config_section_name, {}))
        # --- End Simplified Loading ---

        # Check required node params
        required_node_params = ['embedding_size', 'hidden_size', 'num_layers']
        missing_params = [p for p in required_node_params if p not in node_config_dict]
        if missing_params:
            raise KeyError(f"Missing required node parameters in 'model.GraphRNN': {missing_params}")

        # Check required edge params
        required_edge_params = ['embedding_size', 'hidden_size', 'num_layers']
        missing_params_edge = [p for p in required_edge_params if p not in edge_config_dict]
        if missing_params_edge:
            raise KeyError(f"Missing required edge parameters in 'model.EdgeRNN': {missing_params_edge}")


        # Set derived/default parameters
        node_config_dict['max_level'] = node_config_dict.get('max_level', data_config.get('max_level', 22))
        node_config_dict['predict_node_types'] = predict_node_types
        node_config_dict['num_node_types'] = num_node_types
        edge_feature_len = NUM_EDGE_FEATURES # Use internal constant
        node_config_dict['edge_feature_len'] = edge_feature_len
        edge_config_dict['edge_feature_len'] = edge_feature_len
        # Set node output size to match edge hidden size for RNN->RNN connection
        node_config_dict['output_size'] = edge_config_dict['hidden_size']

        logger.info(f"Final interpreted config: node_model=gru, edge_model=rnn, "
                    f"predict_nodes={predict_node_types}, m={input_size}")

    except (KeyError, ValueError) as e:
         logger.error(f"Configuration error during parsing: {e}", exc_info=True)
         raise
    except Exception as e:
        logger.error(f"Unexpected error parsing config: {e}", exc_info=True)
        raise

    # --- Instantiate Models ---
    if not MODEL_CLASSES_LOADED:
        raise RuntimeError("Model classes could not be imported.")

    # Node Model Instantiation (GraphLevelRNN)
    NodeModelClass = GraphLevelRNN
    sig_node = inspect.signature(NodeModelClass.__init__)
    valid_keys_node = {p for p in sig_node.parameters if p != 'self'}
    node_config_dict['input_size'] = input_size
    filtered_node_config = {k: v for k, v in node_config_dict.items() if k in valid_keys_node}
    logger.debug(f"Node Model filtered config for {NodeModelClass.__name__}: {filtered_node_config}")
    try:
        node_model = NodeModelClass(**filtered_node_config).to(device)
    except TypeError as e:
        logger.error(f"TypeError instantiating {NodeModelClass.__name__}: {e}. Config: {filtered_node_config}", exc_info=True)
        raise

    # Edge Model Instantiation (EdgeLevelRNN)
    EdgeModelClass = EdgeLevelRNN
    edge_gen_function = rnn_edge_gen # Set generation function
    sig_edge = inspect.signature(EdgeModelClass.__init__)
    valid_keys_edge = {p for p in sig_edge.parameters if p != 'self'}
    filtered_edge_config = {k: v for k, v in edge_config_dict.items() if k in valid_keys_edge}
    logger.debug(f"Edge Model filtered config for {EdgeModelClass.__name__}: {filtered_edge_config}")
    try:
        edge_model = EdgeModelClass(**filtered_edge_config).to(device)
    except TypeError as e:
        logger.error(f"TypeError instantiating {EdgeModelClass.__name__}: {e}. Config: {filtered_edge_config}", exc_info=True)
        raise

    # Load State Dicts
    try:
        node_state_dict = state.get('node_model')
        edge_state_dict = state.get('edge_model')
        if node_state_dict is None or edge_state_dict is None:
            raise KeyError("Missing 'node_model' or 'edge_model' state_dict in checkpoint")

        node_load_result = node_model.load_state_dict(node_state_dict, strict=False)
        edge_load_result = edge_model.load_state_dict(edge_state_dict, strict=False)
        # Log missing/unexpected keys
        if node_load_result.missing_keys: logger.warning(f"Node model missing keys: {node_load_result.missing_keys}")
        if node_load_result.unexpected_keys: logger.warning(f"Node model unexpected keys: {node_load_result.unexpected_keys}")
        if edge_load_result.missing_keys: logger.warning(f"Edge model missing keys: {edge_load_result.missing_keys}")
        if edge_load_result.unexpected_keys: logger.warning(f"Edge model unexpected keys: {edge_load_result.unexpected_keys}")
        logger.info("Model state dictionaries loaded (strict=False).")
    except KeyError as e:
        logger.error(f"State dict loading error: Missing key {e}")
        raise
    except Exception as e:
         logger.error(f"Error loading state_dict: {e}", exc_info=True)
         raise

    mode = model_config.get('mode', 'aig')

    # Return necessary components including node prediction info
    return (node_model, edge_model, input_size, edge_gen_function, mode,
            edge_feature_len, predict_node_types, num_node_types)
# --- END MODIFIED ---


# --- Generation Helper Function (Unchanged from previous version) ---
def get_generation(
    num_graphs_to_generate: int,
    models_tuple: Tuple[torch.nn.Module, torch.nn.Module, int, Callable, str, int, bool, Optional[int]],
    gen_params: Dict[str, Any]
    ) -> Tuple[List[Optional[nx.DiGraph]], int]:
    """Generates graphs using the loaded models."""
    (node_model, edge_model, input_size, edge_gen_function,
     mode, edge_feature_len, predict_node_types, num_node_types) = models_tuple

    logger.debug(f"Generating {num_graphs_to_generate} AIGs (predict_nodes={predict_node_types})...")
    generated_graphs_list: List[Optional[nx.DiGraph]] = []

    current_gen_params = { **gen_params } # Copy base params

    generation_successful_count = 0
    for i in range(num_graphs_to_generate):
        logger.debug(f"Generating graph {i+1}/{num_graphs_to_generate}...")
        try:
            # Call generate function (ensure it's updated)
            graph = generate(
                node_model=node_model, edge_model=edge_model,
                input_size=input_size, edge_gen_function=edge_gen_function,
                mode=mode, edge_feature_len=edge_feature_len,
                predict_node_types=predict_node_types, # Pass flag
                num_node_types=num_node_types,       # Pass count
                **current_gen_params # Pass other gen params
            )

            if isinstance(graph, nx.DiGraph) and graph.number_of_nodes() > 0:
                 generation_successful_count += 1
            else:
                logger.warning(f"Generation {i+1} resulted in None or empty graph.")
                graph = None
            generated_graphs_list.append(graph)
        except Exception as e:
            logger.error(f"Error during generation of graph {i+1}: {e}", exc_info=True)
            generated_graphs_list.append(None)

    num_successfully_generated = generation_successful_count
    logger.info(f"Finished generation. Generated {num_successfully_generated}/{num_graphs_to_generate} non-empty graphs.")
    return generated_graphs_list, num_successfully_generated


# --- Main Control Function (Unchanged from previous version) ---
def aig_control(
        model_checkpoint_path: str,
        output_dir: str,
        gen_params: Dict[str, Any],
        num_graphs_to_generate: int = 500,
        output_graphs_filename: str = "generated_graphs.pkl"
) -> Dict[str, Any]:
    """Main control function: Load model, generate graphs, save results."""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting AIG GENERATION process. Output dir: {output_dir}")
    logger.info(f"Requested Generations: {num_graphs_to_generate}")
    logger.info(f"Base Generation Parameters: {gen_params}")

    aggregated_results: Dict[str, Any] = {
        "model_checkpoint": model_checkpoint_path,
        "generation_params_base": gen_params,
        "output_directory": output_dir,
        "requested_generations": num_graphs_to_generate,
        "generated_graphs_file": os.path.join(output_dir, output_graphs_filename)
    }

    # 1. Load Model
    logger.info(f"Loading model from {model_checkpoint_path}...")
    try:
        models_tuple = load_model_from_config(model_checkpoint_path)
        logger.info(f"Model loaded. Node prediction enabled: {models_tuple[6]}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        aggregated_results["error"] = f"Model loading failed: {e}"
        results_file = os.path.join(output_dir, "aig_summary.json")
        try:
            with open(results_file, "w") as f: json.dump(aggregated_results, f, indent=2, default=str)
        except Exception as save_e: logger.error(f"Failed to save error summary: {save_e}")
        return aggregated_results

    # 2. Generate Graphs
    generated_graphs_list, num_successfully_generated = get_generation(
        num_graphs_to_generate=num_graphs_to_generate,
        models_tuple=models_tuple,
        gen_params=gen_params
    )
    aggregated_results["num_graphs_generated"] = num_successfully_generated
    valid_generated_graphs = [g for g in generated_graphs_list if g is not None]

    # 3. Save Generated Graphs
    if not valid_generated_graphs:
        logger.warning("No valid graphs generated. Cannot save PKL file.")
        aggregated_results["warning"] = "No valid graphs generated."
    else:
        graphs_output_path = aggregated_results["generated_graphs_file"]
        logger.info(f"Saving {len(valid_generated_graphs)} graphs to {graphs_output_path}...")
        try:
            with open(graphs_output_path, 'wb') as f_out:
                pickle.dump(valid_generated_graphs, f_out)
            logger.info("Successfully saved generated graphs.")
        except Exception as e:
            logger.error(f"Failed to save generated graphs pkl: {e}")
            aggregated_results["error_saving_graphs"] = str(e)

    # 4. Save Summary
    results_file = os.path.join(output_dir, "aig_summary.json")
    try:
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int_)): return int(obj)
            elif isinstance(obj, (np.floating, np.float_)): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, (set, frozenset)): return list(obj)
            elif isinstance(obj, (torch.Tensor,)): return obj.tolist()
            return str(obj) # Default to string conversion

        serializable_results = json.loads(json.dumps(aggregated_results, default=convert_numpy))
        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Aggregated results summary saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results JSON: {e}")

    # 5. Return Results
    total_time = time.time() - start_time
    logger.info(f"AIG generation process finished in {total_time:.2f} seconds.")
    aggregated_results['total_time_seconds'] = round(total_time, 2)
    return aggregated_results


# --- Main Execution Guard (Unchanged from previous version) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AIGs using a trained model and save them.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint (.pth).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--num-generate", type=int, default=500, help="Number of AIGs to generate.")
    parser.add_argument("--output-graphs-file", type=str, default="generated_graphs.pkl", help="Output PKL filename.")
    # Generation Parameters
    parser.add_argument('--gen-max-nodes', type=int, default=64, help="Max nodes for generation.")
    parser.add_argument('--gen-min-nodes', type=int, default=5, help="Min nodes for generation.")
    parser.add_argument('--gen-patience', type=int, default=16, help="Patience for stopping generation.")
    parser.add_argument('--gen-temp', type=float, default=0.6, help="Temperature for sampling.")
    parser.add_argument('--gen-top-k', type=int, default=0, help="Top_k for sampling (0 to disable).")
    parser.add_argument('--gen-top-p', type=float, default=0.0, help="Top_p for sampling (0.0 to disable).")
    parser.add_argument('--gen-edge-attempts', type=int, default=1, help="Attempts for edge sampling (for RNN).") # Changed default to 1 for RNN

    args = parser.parse_args()

    GENERATION_CONFIG = {
        'max_nodes': args.gen_max_nodes,
        'min_nodes': args.gen_min_nodes,
        'patience': args.gen_patience,
        'temperature': args.gen_temp,
        'top_k': args.gen_top_k,
        'top_p': args.gen_top_p,
        'edge_sample_attempts': args.gen_edge_attempts,
    }

    logger.info("Starting AIG generation script...")
    final_results = aig_control(
        model_checkpoint_path=args.model_path,
        output_dir=args.output_dir,
        gen_params=GENERATION_CONFIG,
        num_graphs_to_generate=args.num_generate,
        output_graphs_filename=args.output_graphs_file
    )
    logger.info("--- Final Control Function Summary ---")
    print(json.dumps(final_results, indent=2, default=str))
    logger.info("--- End of Script ---")
