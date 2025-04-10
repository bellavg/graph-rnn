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
import inspect
from typing import Dict, Any, List, Tuple, Optional

# --- Import from project files ---
# Assume model.py is in the same directory or accessible via PYTHONPATH
try:
    from model import (GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP,
                       GraphLevelAttentionRNN, EdgeLevelAttentionRNN,
                       GraphLevelLSTM, EdgeLevelLSTM,
                       GraphLevelAttentionLSTM, EdgeLevelAttentionLSTM)
    MODEL_CLASSES_LOADED = True
except ImportError as e:
    print(f"Error importing model classes: {e}. Ensure model.py is accessible.")
    MODEL_CLASSES_LOADED = False
    # Exit if models can't be loaded, as they are essential
    sys.exit(1)

# Import generation functions
try:
    from generate_aigs import generate, rnn_edge_gen, mlp_edge_gen, EDGE_TYPES, NUM_EDGE_FEATURES
except ImportError as e:
    print(f"Error importing from generate_aigs.py: {e}. Ensure the file exists and is accessible.")
    sys.exit(1)

# Import evaluation and visualization functions
try:
    from evaluate_aigs import (aig_to_networkx, infer_node_types,
                               calculate_structural_aig_validity, calculate_seadag_validity,
                               count_pi_po_paths, visualize_aig_structure)
except ImportError as e:
    print(f"Error importing from evaluate_aigs.py: {e}. Ensure the file exists and is accessible.")
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


# --- Main Control Function ---
def aig_control(
    model_checkpoint_path: str,
    output_dir: str,
    gen_params: Dict[str, Any],
    num_graphs_to_generate: int = 50,
    num_graphs_to_evaluate: int = 50,
    visualize: bool = True,
    num_graphs_to_visualize: int = 5
) -> Dict[str, Any]:
    """
    Main control function to load model, generate, evaluate, and visualize AIGs.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    if visualize:
        os.makedirs(viz_dir, exist_ok=True)

    logger.info(f"Starting AIG control process. Output dir: {output_dir}")
    logger.info(f"Requested Generations: {num_graphs_to_generate}, Evaluations: {num_graphs_to_evaluate}, Visualizations: {num_graphs_to_visualize if visualize else 0}")
    logger.info(f"Base Generation Parameters: {gen_params}")

    aggregated_results: Dict[str, Any] = {
        "model_checkpoint": model_checkpoint_path,
        "generation_params_base": gen_params,
        "output_directory": output_dir,
        "requested_generations": num_graphs_to_generate,
        "requested_evaluations": num_graphs_to_evaluate,
        "visualization_enabled": visualize,
        "requested_visualizations": num_graphs_to_visualize if visualize else 0,
    }

    # 1. Load Model
    logger.info(f"Loading model from {model_checkpoint_path}...")
    try:
        models_tuple = load_model_from_config(model_checkpoint_path)
        (node_model, edge_model, input_size, edge_gen_function,
         mode, edge_feature_len) = models_tuple
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
         logger.error(f"Model checkpoint not found: {model_checkpoint_path}")
         aggregated_results["error"] = "Model checkpoint not found."
         return aggregated_results
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        aggregated_results["error"] = f"Model loading failed: {e}"
        return aggregated_results

    # 2. Generate Graphs
    logger.info(f"Generating {num_graphs_to_generate} AIGs...")
    generated_graphs_nx: List[Optional[nx.DiGraph]] = []
    generation_times = []

    # Prepare the full generation parameters dictionary
    # Combine base params with sampling params (which might be fixed or varied later)
    current_gen_params = {
        **gen_params, # max_nodes, min_nodes, patience
        'temperature': gen_params.get('temperature', 1.0),
        'top_k': gen_params.get('top_k', 0),
        'top_p': gen_params.get('top_p', 0.0),
        'edge_sample_attempts': gen_params.get('edge_sample_attempts', 1)
    }

    for i in range(num_graphs_to_generate):
        gen_start_time = time.time()
        logger.info(f"Generating graph {i+1}/{num_graphs_to_generate}...")
        try:
            adj_conn, adj_inv = generate( # Call generate from generate_aigs.py
                node_model=node_model, edge_model=edge_model,
                input_size=input_size, edge_gen_function=edge_gen_function,
                mode=mode, edge_feature_len=edge_feature_len,
                # Pass all necessary params from current_gen_params
                **current_gen_params
            )

            graph = None
            if adj_conn is not None and adj_inv is not None and adj_conn.shape[0] > 0:
                graph = aig_to_networkx(adj_conn, adj_inv) # Call from evaluate_aigs.py
                if graph.number_of_nodes() == 0:
                     logger.warning(f"Generation {i+1}: aig_to_networkx resulted in empty graph.")
                     graph = None # Treat as failed generation
                else:
                     # Pre-calculate inferred types for efficiency in evaluation/visualization
                     try:
                        graph.graph['_inferred_types_cleaned'] = infer_node_types(graph)
                     except Exception as infer_e:
                         logger.warning(f"Could not pre-calculate node types for graph {i+1}: {infer_e}")

            else:
                logger.warning(f"Generation {i+1} resulted in empty or None adjacency matrix.")

            generated_graphs_nx.append(graph) # Append graph or None

        except Exception as e:
            logger.error(f"Error during generation or conversion of graph {i+1}: {e}", exc_info=True)
            generated_graphs_nx.append(None) # Mark as failed

        generation_times.append(time.time() - gen_start_time)

    # Filter out None values / truly empty graphs before evaluation
    valid_generated_graphs = [g for g in generated_graphs_nx if isinstance(g, nx.DiGraph) and g.number_of_nodes() > 0]
    num_successfully_generated = len(valid_generated_graphs)
    avg_gen_time = np.mean(generation_times) if generation_times else 0
    logger.info(f"Finished generation. Successfully generated {num_successfully_generated}/{num_graphs_to_generate} non-empty graphs.")
    logger.info(f"Average generation time per graph: {avg_gen_time:.3f}s")

    aggregated_results["num_graphs_successfully_generated"] = num_successfully_generated
    aggregated_results["avg_generation_time_seconds"] = round(avg_gen_time, 3)
    aggregated_results["generation_times_list"] = [round(t, 3) for t in generation_times] # Store individual times

    if num_successfully_generated == 0:
        logger.warning("No graphs were generated successfully. Skipping evaluation and visualization.")
        aggregated_results["warning"] = "No graphs generated successfully."
        # Save summary and exit
        results_file = os.path.join(output_dir, "aig_summary.json")
        try:
            with open(results_file, "w") as f: json.dump(aggregated_results, f, indent=2)
            logger.info(f"Partial results saved to {results_file}")
        except Exception as e: logger.error(f"Failed to save partial results file: {e}")
        return aggregated_results

    # 3. Evaluate Generated Graphs
    # Decide how many graphs to actually evaluate
    num_to_evaluate = min(num_graphs_to_evaluate, num_successfully_generated)
    graphs_for_eval = valid_generated_graphs[:num_to_evaluate]
    logger.info(f"Evaluating structural properties of {num_to_evaluate} generated graphs...")

    evaluation_results_list: List[Dict[str, Any]] = [] # Store detailed results per graph
    eval_start_time = time.time()

    for i, g in enumerate(graphs_for_eval):
        graph_eval_results = {}
        try:
            logger.debug(f"Evaluating graph {i+1}/{num_to_evaluate}...")
            # Call evaluation functions from evaluate_aigs.py
            struct_info = calculate_structural_aig_validity(g)
            seadag_v = calculate_seadag_validity(g)
            path_info = count_pi_po_paths(g)

            # Store results on graph object and in list
            g.graph['_structural_validity'] = struct_info
            g.graph['_seadag_validity'] = seadag_v
            g.graph['_path_info'] = path_info

            graph_eval_results.update(struct_info)
            graph_eval_results['seadag_validity'] = seadag_v
            graph_eval_results.update(path_info)
            # Add identifier if needed
            # graph_eval_results['generation_index'] = valid_generated_graphs.index(g)

        except Exception as eval_e:
            logger.error(f"Error evaluating graph {i+1}: {eval_e}", exc_info=True)
            graph_eval_results = {"error": str(eval_e)} # Mark error for this graph

        evaluation_results_list.append(graph_eval_results)

    eval_time = time.time() - eval_start_time
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds.")

    # --- Aggregate Evaluation Results ---
    aggregated_results["num_graphs_evaluated"] = num_to_evaluate
    if num_to_evaluate > 0:
        valid_evals = [r for r in evaluation_results_list if 'error' not in r]
        num_valid_evals = len(valid_evals)
        if num_valid_evals > 0:
            aggregated_results["evaluated_dag_rate"] = sum(r.get('is_dag', False) for r in valid_evals) / num_valid_evals
            aggregated_results["avg_structural_validity_score"] = round(np.mean([r.get('validity_score', 0.0) for r in valid_evals]), 4)
            aggregated_results["avg_seadag_validity"] = round(np.mean([r.get('seadag_validity', 0.0) for r in valid_evals]), 4)
            aggregated_results["avg_nodes_eval"] = round(np.mean([r.get('num_nodes', 0) for r in valid_evals]), 2)
            aggregated_results["avg_pi_eval"] = round(np.mean([r.get('num_pi', 0) for r in valid_evals]), 2)
            aggregated_results["avg_po_eval"] = round(np.mean([r.get('num_po', 0) for r in valid_evals]), 2)
            aggregated_results["avg_and_eval"] = round(np.mean([r.get('num_and', 0) for r in valid_evals]), 2)
            aggregated_results["total_invalid_fanin_nodes_eval"] = sum(r.get('num_invalid_fanin', 0) for r in valid_evals)
            aggregated_results["total_unknown_type_nodes_eval"] = sum(r.get('num_unknown', 0) for r in valid_evals)
            # Path connectivity aggregation
            path_evals = [r for r in valid_evals if r.get('is_dag', False)] # Only consider DAGs for path stats
            if path_evals:
                 aggregated_results["avg_fraction_pis_connected"] = round(np.mean([r.get('fraction_pis_connected', 0.0) for r in path_evals]), 4)
                 aggregated_results["avg_fraction_pos_connected"] = round(np.mean([r.get('fraction_pos_connected', 0.0) for r in path_evals]), 4)
                 aggregated_results["avg_pis_reaching_po"] = round(np.mean([r.get('num_pis_reaching_po', 0) for r in path_evals]), 2)
                 aggregated_results["avg_pos_reachable_from_pi"] = round(np.mean([r.get('num_pos_reachable_from_pi', 0) for r in path_evals]), 2)

    # Log summary of aggregated results
    logger.info("--- Aggregated Evaluation Summary ---")
    for key, value in aggregated_results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        elif isinstance(value, list) and key.endswith("_list"):
             logger.info(f"  {key}: (list of {len(value)} items)") # Don't print long lists
        elif key != "generation_params_base": # Avoid printing dict again
            logger.info(f"  {key}: {value}")
    logger.info("------------------------------------")


    # 4. Select Graphs for Visualization & Visualize
    graphs_to_visualize_indices = []
    aggregated_results["num_graphs_visualized"] = 0
    if visualize and num_graphs_to_visualize > 0 and num_successfully_generated > 0:
        num_to_viz = min(num_graphs_to_visualize, num_successfully_generated)
        logger.info(f"Selecting up to {num_to_viz} graphs for visualization (best structural score, then size)...")

        # Use evaluation results already computed and stored on graphs
        graph_scores = []
        for i, g in enumerate(valid_generated_graphs): # Iterate through all successfully generated
            struct_info = g.graph.get('_structural_validity')
            # Select based on DAG property and validity score
            if struct_info and struct_info.get('is_dag', False):
                 score = struct_info.get('validity_score', 0.0)
                 nodes = g.number_of_nodes()
                 graph_scores.append((score, nodes, i)) # Store score, size, original index in valid_generated_graphs

        if not graph_scores:
             logger.warning("No valid DAGs found among generated graphs to select for visualization.")
        else:
             # Sort: higher score first, then larger size first
             graph_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
             # Get the indices of the top graphs to visualize
             graphs_to_visualize_indices = [idx for score, nodes, idx in graph_scores[:num_to_viz]]
             logger.info(f"Selected graph indices for visualization: {graphs_to_visualize_indices}")

             # Visualize the selected graphs
             logger.info(f"Visualizing {len(graphs_to_visualize_indices)} selected graphs...")
             for rank, graph_index in enumerate(graphs_to_visualize_indices):
                 g_to_viz = valid_generated_graphs[graph_index]
                 score, nodes, _ = graph_scores[rank] # Get score/nodes for filename
                 # Create a meaningful filename
                 fname = f"rank_{rank+1:02d}_score_{score:.3f}_nodes_{nodes}_idx_{graph_index}.png"
                 output_path = os.path.join(viz_dir, fname)
                 try:
                     # Call visualize function from evaluate_aigs.py
                     visualize_aig_structure(g_to_viz, output_file=output_path)
                 except Exception as e:
                     logger.error(f"Failed to visualize graph index {graph_index} ({output_path}): {e}", exc_info=True)

             logger.info(f"Visualizations saved in {viz_dir}")
             aggregated_results["num_graphs_visualized"] = len(graphs_to_visualize_indices)

    # 5. Save aggregated results
    results_file = os.path.join(output_dir, "aig_summary.json")
    try:
        # Convert numpy types to native python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        # Apply conversion recursively if needed, or just top-level
        serializable_results = json.loads(json.dumps(aggregated_results, default=convert_numpy))

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Aggregated results saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results JSON file: {e}")

    # 6. Return Results
    total_time = time.time() - start_time
    logger.info(f"AIG control process finished in {total_time:.2f} seconds.")
    aggregated_results['total_time_seconds'] = round(total_time, 2)

    return aggregated_results


# --- Main Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate, Evaluate, and Visualize AIGs using a trained model.")

    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save generated graphs, evaluations, and visualizations.")

    # Generation and Evaluation Control
    parser.add_argument("--num-generate", type=int, default=1000,
                        help="Number of AIGs to attempt generating.")
    parser.add_argument("--num-evaluate", type=int, default=1000,
                        help="Maximum number of successfully generated AIGs to evaluate.")

    # Visualization Control
    parser.add_argument("--visualize", action='store_true',
                        help="Enable visualization of generated AIGs.")
    parser.add_argument("--num-visualize", type=int, default=5,
                        help="Maximum number of 'best' AIGs to visualize (if --visualize is enabled).")

    # Generation Hyperparameters (kept in a dict, but could be args too)
    # You can add more argparse arguments here for temperature, top_k, top_p, patience etc. if needed.
    # Example:
    # parser.add_argument("--max-nodes", type=int, default=30, help="Max nodes for generation.")
    # parser.add_argument("--min-nodes", type=int, default=8, help="Min nodes before patience.")
    # parser.add_argument("--patience", type=int, default=5, help="Patience for stopping generation.")
    # parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    # parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling parameter.")
    # parser.add_argument("--top-p", type=float, default=0.0, help="Top-p (nucleus) sampling parameter.")
    # parser.add_argument("--edge-attempts", type=int, default=1, help="Attempts for MLP edge sampling.")

    args = parser.parse_args()

    # --- Base Generation Config Dictionary ---
    # Define default generation parameters here, these can be overridden if
    # you add corresponding argparse arguments later.
    GENERATION_CONFIG = {
        'max_nodes': 100,
        'min_nodes': 8,
        'patience': 12,
        'temperature': 1.0,
        'top_k': 0,
        'top_p': 0.0,
        'edge_sample_attempts': 3,
        # Add other fixed params from your original GENERATION_CONFIG if needed
    }
    # --- You could update GENERATION_CONFIG with args values if you added them to argparse ---
    # Example:
    # GENERATION_CONFIG['max_nodes'] = args.max_nodes
    # GENERATION_CONFIG['min_nodes'] = args.min_nodes
    # ... etc. ...


    logger.info("Starting AIG generation/evaluation script with command line arguments...")

    # Run the main control function using parsed arguments
    final_results = aig_control(
        model_checkpoint_path=args.model_path,
        output_dir=args.output_dir,
        gen_params=GENERATION_CONFIG, # Pass the config dictionary
        num_graphs_to_generate=args.num_generate,
        num_graphs_to_evaluate=args.num_evaluate,
        visualize=args.visualize,
        num_graphs_to_visualize=args.num_visualize
    )

    logger.info("--- Final Control Function Summary ---")
    # Pretty print the final results dictionary
    print(json.dumps(final_results, indent=2, default=str)) # Use default=str for any non-serializable types remaining
    logger.info("--- End of Script ---")