from model import *
from train import *
import torch
import yaml
from aig_dataset import *
import traceback

# utils.py - Placeholder for required utility functions
# You should have these defined correctly in your project.
# This includes a REVERTED version of restore_checkpoint.

import torch
import os
import yaml
import pickle
import traceback

# --- Placeholder for your model definitions (ensure they match config) ---
# from model import GraphLevelRNN, EdgeLevelMLP, ...

# --- Placeholder for your training step functions ---
# from train import train_mlp_step, train_rnn_step, ...

# --- Placeholder for your dataset definition ---
# from aig_dataset import AIGDataset, NUM_EDGE_FEATURES

def load_config(config_file):
    """Loads configuration from YAML file."""
    # (Keep your existing implementation)
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print("Loaded config:")
    print(yaml.dump(config, sort_keys=False))
    if 'data' not in config or 'model' not in config or 'train' not in config:
        raise ValueError("Config file must contain 'data', 'model', and 'train' sections.")
    return config

def setup_models(config, device, max_node_count, max_level):
    """Sets up node and edge models based on config."""
    # (Keep your existing implementation from the provided 'utils.py' section)
    use_bfs = config['data'].get('use_bfs', False)

    # Determine Effective Input/Output Size
    if use_bfs:
        effective_input_size = config['data'].get('m')
        if effective_input_size is None: raise ValueError("Config 'data.m' required for BFS.")
        print(f"INFO: Using BFS mode. Effective input/output size (m): {effective_input_size}")
    else: # TopSort mode
        if max_node_count <= 1: raise ValueError("max_node_count must be > 1 for training.")
        effective_input_size = max_node_count - 1
        print(f"INFO: Using Topological Sort mode. Effective input/output size (max_nodes-1): {effective_input_size}")

    # --- Model Selection Flags ---
    use_lstm = config['model'].get('use_lstm', False)
    use_node_attention = config['model'].get('use_attention', False)
    edge_model_choice = config['model'].get('edge_model', 'mlp').lower()

    node_model = None
    edge_model = None
    step_fn = None
    node_model_output_size_for_edge = None
    node_params = {}
    edge_params = {}

    # --- Determine Node Model Config Section and Params ---
    node_config_section = None
    if use_lstm:
        node_config_section = 'GraphAttentionLSTM' if use_node_attention else 'GraphLSTM'
    else: # GRU
        if use_node_attention:
            node_config_section = 'GraphAttentionRNN'
            if node_config_section not in config['model'] and 'GraphRNN' in config['model']:
                print(f"Warning: Config section 'model.{node_config_section}' not found, using 'GraphRNN'. Add 'attention_heads/dropout' there if needed.")
                node_config_section = 'GraphRNN'
        else:
            node_config_section = 'GraphRNN'

    if node_config_section not in config['model']:
        raise ValueError(f"Config missing required model section 'model.{node_config_section}'.")

    node_config = config['model'][node_config_section]
    node_params.update(node_config)

    node_params['input_size'] = effective_input_size
    node_params['max_level'] = max_level
    node_params['predict_node_types'] = False
    node_params['num_node_types'] = None
    node_params['use_conditioning'] = False
    node_params['tt_size'] = None
    node_edge_feature_len = node_params.get('edge_feature_len')
    if node_edge_feature_len is None:
        raise ValueError(f"'{node_config_section}.edge_feature_len' must be defined.")
    node_model_output_size_for_edge = node_params.get('output_size')

    # --- Instantiate Node Model ---
    # (Keep the instantiation logic from your provided utils.py)
    # Example for GRU (adapt based on your actual model classes):
    if use_lstm:
         # ... Your LSTM instantiation logic ...
         pass # Replace with actual LSTM logic if used
    else: # GRU
        if use_node_attention:
            # GraphLevelAttentionRNN expects attention_heads, attention_dropout
            node_params['attention_heads'] = node_params.get('attention_heads', 4) # Keep/default these
            node_params['attention_dropout'] = node_params.get('attention_dropout', 0.1) # Keep/default these
            # It does NOT expect use_attention
            node_params.pop('use_attention', None) # Remove if present
            # Replace GraphLevelAttentionRNN with your actual class name if different
            node_model = GraphLevelAttentionRNN(**node_params).to(device)
            print(f"INFO: Using GraphLevelAttentionRNN for node level.")
        else:
            # GraphLevelRNN does NOT expect any attention args
            node_params.pop('use_attention', None) # Remove if present
            node_params.pop('attention_heads', None)
            node_params.pop('attention_dropout', None)
            # Replace GraphLevelRNN with your actual class name if different
            node_model = GraphLevelRNN(**node_params).to(device)
            print(f"INFO: Using standard GraphLevelRNN for node level.")

    # --- Initialize Edge Model ---
    edge_params['edge_feature_len'] = node_edge_feature_len

    if edge_model_choice == 'mlp':
        config_section = 'EdgeMLP'
        if config_section not in config['model']: raise ValueError(f"Config missing 'model.{config_section}'.")
        node_hidden_size = node_params.get('hidden_size')
        if node_hidden_size is None: raise ValueError(f"Node model section '{node_config_section}' must define 'hidden_size' for EdgeMLP.")

        edge_config = config['model'][config_section]
        edge_params.update(edge_config)
        edge_params['output_size'] = effective_input_size
        edge_params['input_size'] = node_hidden_size
        if 'edge_feature_len' not in edge_params: edge_params['edge_feature_len'] = node_edge_feature_len

        # Replace EdgeLevelMLP with your actual class name if different
        edge_model = EdgeLevelMLP(**edge_params).to(device)
        step_fn = train_mlp_step # Ensure this is the correct step function
        print("Selected EdgeLevelMLP model.")

    elif edge_model_choice in ['rnn', 'attention_rnn']:
        # (Keep the RNN/LSTM edge model logic from your provided utils.py)
        # Ensure the correct step function (e.g., train_rnn_step) is assigned
        # and handle potential LSTM state differences if applicable.
        is_edge_lstm = use_lstm # Assume edge uses same RNN type as node for simplicity
        is_edge_attention = edge_model_choice.startswith('attention')
        EdgeModelClass = None
        if is_edge_lstm:
             # ... Your Edge LSTM instantiation logic ...
             step_fn = train_rnn_step # Or train_lstm_step if you made one
             pass # Replace with actual LSTM logic if used
        else: # GRU Edge Model
            if is_edge_attention:
                config_section = 'EdgeAttentionRNN'
                if config_section not in config['model'] and 'EdgeRNN' in config['model']:
                        print(f"Warning: Config section 'model.{config_section}' not found, using 'EdgeRNN'. Add 'attention_heads/dropout' there if needed.")
                        config_section = 'EdgeRNN'
                # Replace EdgeLevelAttentionRNN with your actual class name if different
                EdgeModelClass = EdgeLevelAttentionRNN
                step_fn = train_rnn_step
            else:
                config_section = 'EdgeRNN'
                # Replace EdgeLevelRNN with your actual class name if different
                EdgeModelClass = EdgeLevelRNN
                step_fn = train_rnn_step

            if config_section not in config['model']: raise ValueError(f"Config missing 'model.{config_section}'.")
            edge_config = config['model'][config_section]
            edge_params.update(edge_config)

            # Add/remove attention params appropriately
            if is_edge_attention and config_section in ['EdgeRNN', 'EdgeLSTM']: # If using fallback or base LSTM section
                    edge_params['attention_heads'] = edge_params.get('attention_heads', 4)
                    edge_params['attention_dropout'] = edge_params.get('attention_dropout', 0.1)
            elif not is_edge_attention:
                edge_params.pop('attention_heads', None)
                edge_params.pop('attention_dropout', None)

            # --- Critical Check: Node Output -> Edge Hidden ---
            edge_hidden_size = edge_params.get('hidden_size')
            if edge_hidden_size is None:
                raise ValueError(f"Edge model section '{config_section}' must define 'hidden_size'.")
            if node_model_output_size_for_edge is None: # Check if node model defines output_size when needed
                raise ValueError(f"Node model section '{node_config_section}' must define 'output_size' when edge model is '{edge_model_choice}'.")
            if node_model_output_size_for_edge != edge_hidden_size:
                raise ValueError(f"Mismatch: Node model output_size ({node_model_output_size_for_edge}) != Edge model hidden_size ({edge_hidden_size}). Check config sections '{node_config_section}' and '{config_section}'.")
            # --- End Check ---

            edge_model = EdgeModelClass(**edge_params).to(device)
            print(f"Selected {EdgeModelClass.__name__} model.")


    else:
        raise ValueError(f"Unsupported edge_model type in config: {edge_model_choice}")

    if node_model is None or edge_model is None or step_fn is None:
        raise RuntimeError("Failed to initialize node_model, edge_model, or step_fn.")

    return node_model, edge_model, step_fn

def setup_criteria(config, device, dataset):
    """Determines loss function and use_edge_features flag based on config."""
    # (Keep your existing implementation from the provided 'utils.py' section)
    # Determine edge_feature_len (you might need NUM_EDGE_FEATURES from aig_dataset as fallback)
    node_config_section = None # Logic to find the correct node model section
    use_lstm = config['model'].get('use_lstm', False)
    use_node_attention = config['model'].get('use_attention', False)
    if use_lstm:
        node_config_section = 'GraphAttentionLSTM' if use_node_attention else 'GraphLSTM'
    else: # GRU
        if use_node_attention:
            node_config_section = 'GraphAttentionRNN'
            if node_config_section not in config['model'] and 'GraphRNN' in config['model']:
                node_config_section = 'GraphRNN'
        else:
            node_config_section = 'GraphRNN'

    edge_feature_len = None
    if node_config_section and node_config_section in config['model']:
         edge_feature_len = config['model'][node_config_section].get('edge_feature_len')

    if edge_feature_len is None:
         print(f"Warning: Could not determine 'edge_feature_len' from config model section '{node_config_section}'. Falling back.")
         # Fallback logic - perhaps NUM_EDGE_FEATURES if that's appropriate? Or raise error.
         # Forcing a default for now, adjust as needed:
         edge_feature_len = config['model'].get('default_edge_feature_len', 1) # Or raise error
         if edge_feature_len == 1: print("Assuming binary edge features (edge_feature_len=1).")


    use_edge_features = edge_feature_len > 1

    if use_edge_features:
        print(f"Setting up CrossEntropyLoss for {edge_feature_len} edge classes.")
        edge_weights = None
        if hasattr(dataset, 'edge_weights') and dataset.edge_weights is not None:
            edge_weights = dataset.edge_weights.to(device)
            print(f"Applying edge class weights: {edge_weights.tolist()}")
            criterion_edge = torch.nn.CrossEntropyLoss(weight=edge_weights).to(device)
        else:
            print("Warning: Dataset object does not have 'edge_weights'. Using uniform weights.")
            criterion_edge = torch.nn.CrossEntropyLoss().to(device)
    else:
        print("Using BCELoss for binary edges.")
        criterion_edge = torch.nn.BCELoss().to(device)

    return criterion_edge, use_edge_features


def get_max_node_count_from_pkl(graph_file: str) -> int:
    """Gets max node count from dataset file."""
    # (Keep your existing implementation from the provided 'utils.py' section)
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Dataset file not found: {graph_file}")
    max_nodes = 0
    loaded_count = 0
    try:
        with open(graph_file, 'rb') as f:
            raw_graphs = pickle.load(f)
        num_graphs = len(raw_graphs)
        print(f"Checking {num_graphs} graphs for max node count...")
        for i, g in enumerate(raw_graphs):
            # Check if 'g' is a graph object (adjust based on your data structure)
            # Example using networkx: if isinstance(g, nx.DiGraph):
            # Adapt this check based on how graphs are stored in your pickle file
            if hasattr(g, 'number_of_nodes') and callable(g.number_of_nodes): # More robust check
                max_nodes = max(max_nodes, g.number_of_nodes())
                loaded_count += 1
            else:
                print(f"Warning: Item {i} in pickle file does not seem to be a graph object with number_of_nodes(). Skipping.")
        if loaded_count == 0: raise ValueError("No valid graph objects found in the pickle file.")
    except (pickle.UnpicklingError, EOFError) as e:
        raise IOError(f"Error reading pickle file {graph_file}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while reading {graph_file}: {e}")
        traceback.print_exc() # Print details
        raise RuntimeError(f"Could not process {graph_file}: {e}")
    return max_nodes


# --- REVERTED restore_checkpoint ---
# Removed load_schedulers flag and related conditional logic
def restore_checkpoint(path, device, node_model, edge_model,
                       optim_node_model, optim_edge_model,
                       scheduler_node_model, scheduler_edge_model): # Expect initialized schedulers
    """Restores state from checkpoint, including schedulers if present."""
    print(f"Restoring from checkpoint: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    state = torch.load(path, map_location=device)

    # Check essential keys
    required_keys = ["global_step", "node_model", "edge_model", "optim_node_model", "optim_edge_model"]
    for key in required_keys:
        if key not in state:
            raise KeyError(f"Checkpoint missing required key: '{key}'")

    global_step = state["global_step"]
    try:
        node_model.load_state_dict(state["node_model"])
        edge_model.load_state_dict(state["edge_model"])
        optim_node_model.load_state_dict(state["optim_node_model"])
        optim_edge_model.load_state_dict(state["optim_edge_model"])
        print(f"Successfully restored models and optimizers to step {global_step}.")

        # --- Always try to load schedulers if they exist in checkpoint and were passed in ---
        loaded_node_sched = False
        if "scheduler_node_model" in state and state["scheduler_node_model"] is not None and scheduler_node_model is not None:
            try:
                scheduler_node_model.load_state_dict(state["scheduler_node_model"])
                print("Restored node model scheduler state.")
                loaded_node_sched = True
            except Exception as e:
                print(f"Warning: Failed to load node scheduler state from checkpoint: {e}. Scheduler will continue from its initialized state.")
        if not loaded_node_sched and scheduler_node_model is not None:
             print("Node model scheduler state not found in checkpoint or scheduler not provided. Scheduler will continue from its initialized state.")


        loaded_edge_sched = False
        if "scheduler_edge_model" in state and state["scheduler_edge_model"] is not None and scheduler_edge_model is not None:
            try:
                scheduler_edge_model.load_state_dict(state["scheduler_edge_model"])
                print("Restored edge model scheduler state.")
                loaded_edge_sched = True
            except Exception as e:
                 print(f"Warning: Failed to load edge scheduler state from checkpoint: {e}. Scheduler will continue from its initialized state.")
        if not loaded_edge_sched and scheduler_edge_model is not None:
             print("Edge model scheduler state not found in checkpoint or scheduler not provided. Scheduler will continue from its initialized state.")
        # --- End Scheduler Load ---

    except Exception as e:
        print(f"Error loading state dictionaries from checkpoint: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Error loading state dictionaries from checkpoint: {e}")

    return global_step