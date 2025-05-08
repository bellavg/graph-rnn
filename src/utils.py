"""
Utility functions for the AIG model setup and evaluation.
Includes setup for node type prediction.
"""

import torch
import torch.nn as nn # Added for nn.CrossEntropyLoss
# Import necessary constants from aig_config
try:
    from aig_config import NUM_EDGE_FEATURES, NUM_NODE_TYPES
except ImportError:
    print("Warning: Could not import from .aig_config. Using default values in utils.py")
    NUM_EDGE_FEATURES = 3
    NUM_NODE_TYPES = 4

# Import model and training functions (assuming GRU/RNN only)
from .model import GraphLevelRNN, EdgeLevelRNN
from .train import train_rnn_step # Assuming only train_rnn_step is used now

def setup_models(config, device, max_node_count, max_level):
    """
    Sets up GraphLevelRNN and EdgeLevelRNN models based on the config.
    Handles node type prediction setup.

    Args:
        config (dict): Dictionary with model configuration.
        device (torch.device): PyTorch device to place models on.
        max_node_count (int): Maximum number of nodes in the training graphs.
        max_level (int): Maximum level in the training graphs.

    Returns:
        tuple: (node_model, edge_model, step_fn)
    """
    # --- Determine Effective Input Size (Assuming Topological Sort) ---
    if max_node_count <= 1: raise ValueError("max_node_count must be > 1 for training.")
    effective_input_size = max_node_count - 1
    print(f"INFO: Using Topological Sort mode. Effective input/output size (max_nodes-1): {effective_input_size}")

    # --- Get Node Prediction Config ---
    # Read from train section first, fallback to model section for backward compatibility
    predict_node_types = config.get('train', {}).get('predict_node_types',
                                                    config.get('model', {}).get('predict_node_types', False))
    # Use NUM_NODE_TYPES from aig_config if predicting
    num_node_types = NUM_NODE_TYPES if predict_node_types else None
    print(f"INFO: Setup models - Predict Node Types: {predict_node_types}")
    if predict_node_types: print(f"INFO: Setup models - Num Node Types: {num_node_types}")
    # --- End Node Prediction Config ---

    # --- Node Model Setup (GraphLevelRNN) ---
    node_config_section = 'GraphRNN'
    if node_config_section not in config.get('model', {}):
        raise ValueError(f"Config missing required model section 'model.{node_config_section}'.")

    node_config = config['model'][node_config_section]
    node_params = {}
    node_params.update(node_config) # Base params

    # Add derived/required params
    node_params['input_size'] = effective_input_size
    node_params['max_level'] = max_level
    node_params['predict_node_types'] = predict_node_types # Pass flag
    node_params['num_node_types'] = num_node_types       # Pass count
    node_params['edge_feature_len'] = node_params.get('edge_feature_len', NUM_EDGE_FEATURES) # Default to 3 if missing
    if node_params['edge_feature_len'] != NUM_EDGE_FEATURES:
         print(f"Warning: Node config edge_feature_len ({node_params['edge_feature_len']}) differs from internal constant ({NUM_EDGE_FEATURES}). Using internal value.")
         node_params['edge_feature_len'] = NUM_EDGE_FEATURES

    # --- Edge Model Setup (EdgeLevelRNN) ---
    edge_config_section = 'EdgeRNN'
    if edge_config_section not in config.get('model', {}):
        raise ValueError(f"Config missing required model section 'model.{edge_config_section}'.")

    edge_config = config['model'][edge_config_section]
    edge_params = {}
    edge_params.update(edge_config) # Base params
    edge_params['edge_feature_len'] = edge_params.get('edge_feature_len', NUM_EDGE_FEATURES) # Default to 3
    if edge_params['edge_feature_len'] != NUM_EDGE_FEATURES:
         print(f"Warning: Edge config edge_feature_len ({edge_params['edge_feature_len']}) differs from internal constant ({NUM_EDGE_FEATURES}). Using internal value.")
         edge_params['edge_feature_len'] = NUM_EDGE_FEATURES

    # --- Link Node Output to Edge Input ---
    # Node model's output size must match edge model's hidden size
    edge_hidden_size = edge_params.get('hidden_size')
    if edge_hidden_size is None:
        raise ValueError(f"Edge model section '{edge_config_section}' must define 'hidden_size'.")
    node_params['output_size'] = edge_hidden_size # Set node output size requirement
    print(f"INFO: Setting NodeRNN output_size to {edge_hidden_size} to match EdgeRNN hidden_size.")

    # --- Instantiate Models ---
    try:
        node_model = GraphLevelRNN(**node_params).to(device)
        print(f"INFO: Instantiated GraphLevelRNN (Predict Nodes: {predict_node_types}).")
    except Exception as e:
        print(f"Error instantiating GraphLevelRNN: {e}")
        raise

    try:
        edge_model = EdgeLevelRNN(**edge_params).to(device)
        print("INFO: Instantiated EdgeLevelRNN.")
    except Exception as e:
        print(f"Error instantiating EdgeLevelRNN: {e}")
        raise

    # Set training step function
    step_fn = train_rnn_step
    print("INFO: Using train_rnn_step.")

    return node_model, edge_model, step_fn


def setup_criteria(config, device, dataset=None):
    """
    Sets up loss criteria for edges and optionally for node types.
    Edge class weights are now hardcoded.
    num_node_types is now fetched from config['data'].
    """
    train_config = config.get('train', {})
    model_config = config.get('model', {})
    data_config = config.get('data', {})  # Get data sub-config

    loss_reduction = train_config.get('loss_reduction', 'mean')

    # --- Edge Criterion ---
    hardcoded_edge_weights = [0.015618128702044487, 1.4666249752044678, 1.5177571773529053]
    num_edge_classes = len(hardcoded_edge_weights)

    print(f"Setting up CrossEntropyLoss for {num_edge_classes} edge classes.")

    edge_class_weights_tensor = torch.tensor(hardcoded_edge_weights, dtype=torch.float32, device=device)
    print(f"Applying HARDCODED edge class weights: {edge_class_weights_tensor.tolist()}")
    criterion_edge = nn.CrossEntropyLoss(weight=edge_class_weights_tensor, reduction=loss_reduction)

    # Determine if edge features are used (for multi-class CrossEntropyLoss)
    # Default to 3 if not specified, aligning with hardcoded weights
    edge_feature_len_for_warning = model_config.get('GraphRNN', {}).get('edge_feature_len',
                                                                        model_config.get('edge_feature_len', 3))
    use_edge_features = edge_feature_len_for_warning > 1

    if not use_edge_features:  # This means edge_feature_len_for_warning is 1
        print(
            f"Warning: edge_feature_len is {edge_feature_len_for_warning}, but CrossEntropyLoss for multiple classes is set up. Ensure this is intended.")

    # --- Node Criterion (Optional) ---
    criterion_node = None
    predict_node_types = train_config.get('predict_node_types',
                                          model_config.get('predict_node_types', False))

    if predict_node_types:
        # **MODIFIED TO FETCH FROM data_config FIRST**
        num_node_types = data_config.get('num_node_types', model_config.get('num_node_types', None))

        if num_node_types is None:
            # This error message reflects the updated logic.
            raise ValueError(
                "num_node_types must be specified in config (preferably under 'data') if predict_node_types is True for criterion setup.")

        print(f"Setting up CrossEntropyLoss for {num_node_types} node classes.")

        node_class_weights_list = train_config.get('node_class_weights', None)
        node_class_weights_tensor = None
        if node_class_weights_list is not None:
            if len(node_class_weights_list) != num_node_types:
                raise ValueError(f"Length of node_class_weights ({len(node_class_weights_list)}) "
                                 f"must match num_node_types ({num_node_types}).")
            node_class_weights_tensor = torch.tensor(node_class_weights_list, dtype=torch.float32, device=device)
            print(f"Applying node class weights: {node_class_weights_tensor.tolist()}")

        criterion_node = nn.CrossEntropyLoss(weight=node_class_weights_tensor, reduction=loss_reduction)
        print("Node type prediction criterion created.")

    return criterion_edge, criterion_node, use_edge_features