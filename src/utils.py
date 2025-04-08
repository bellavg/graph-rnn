"""
Utility functions for the AIG model setup and evaluation.
Combines functions from multiple sources for a unified interface.
"""

import torch
from aig_dataset import NUM_EDGE_FEATURES

# src/utils.py

import torch
import os
from torch.nn import CrossEntropyLoss
# Assuming model classes are defined in model.py
from model import (GraphLevelRNN, GraphLevelLSTM, GraphLevelAttentionRNN,
                   GraphLevelAttentionLSTM, EdgeLevelMLP, EdgeLevelRNN,
                   EdgeLevelAttentionRNN, EdgeLevelLSTM, EdgeLevelAttentionLSTM,
                   NodeTypePredictor)
# Assuming step functions are defined in train.py
from train import train_mlp_step, train_rnn_step
from aig_dataset import NUM_EDGE_FEATURES, NODE_TYPES, NUM_NODE_TYPES



def load_weights_for_finetuning(checkpoint_path, device, node_model, edge_model, node_type_predictor=None):
    """
    Loads only model weights from a checkpoint for fine-tuning.
    Ignores optimizer, scheduler, and global step states.

    Args:
        checkpoint_path: Path to the checkpoint file (.pth).
        device: The torch device ('cpu' or 'cuda:x').
        node_model: The instantiated node-level model.
        edge_model: The instantiated edge-level model.
        node_type_predictor: The instantiated node type predictor model (optional).

    Returns:
        bool: True if weights were loaded successfully, False otherwise.
    """
    print(f"Loading weights for fine-tuning from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return False

    try:
        state = torch.load(checkpoint_path, map_location=device)
        loaded_something = False

        # --- Load Node Model Weights ---
        node_key = "node_model"
        if node_key not in state and "node_net" in state: # Handle old key name
            node_key = "node_net"
            print("  Using old key 'node_net' for node_model weights.")

        if node_key in state:
            try:
                node_model.load_state_dict(state[node_key])
                print(f"  Successfully loaded weights into {type(node_model).__name__}.")
                loaded_something = True
            except RuntimeError as e:
                print(f"  Warning: Error loading node_model weights (possibly architecture mismatch): {e}")
                print("  Attempting load with strict=False...")
                try:
                     node_model.load_state_dict(state[node_key], strict=False)
                     print(f"  Partially loaded weights into {type(node_model).__name__} with strict=False.")
                     loaded_something = True
                except Exception as e_strict:
                     print(f"  Error loading node_model weights even with strict=False: {e_strict}")
            except Exception as e_other:
                 print(f"  Error loading node_model weights: {e_other}")
        else:
            print(f"  Warning: Key '{node_key}' not found in checkpoint. Skipping node_model weights.")


        # --- Load Edge Model Weights ---
        edge_key = "edge_model"
        if edge_key not in state and "edge_net" in state: # Handle old key name
            edge_key = "edge_net"
            print("  Using old key 'edge_net' for edge_model weights.")

        if edge_key in state:
            try:
                edge_model.load_state_dict(state[edge_key])
                print(f"  Successfully loaded weights into {type(edge_model).__name__}.")
                loaded_something = True
            except RuntimeError as e:
                 print(f"  Warning: Error loading edge_model weights (possibly architecture mismatch): {e}")
                 print("  Attempting load with strict=False...")
                 try:
                     edge_model.load_state_dict(state[edge_key], strict=False)
                     print(f"  Partially loaded weights into {type(edge_model).__name__} with strict=False.")
                     loaded_something = True
                 except Exception as e_strict:
                     print(f"  Error loading edge_model weights even with strict=False: {e_strict}")
            except Exception as e_other:
                 print(f"  Error loading edge_model weights: {e_other}")
        else:
            print(f"  Warning: Key '{edge_key}' not found in checkpoint. Skipping edge_model weights.")

        # --- Load Node Type Predictor Weights (Optional) ---
        if node_type_predictor is not None:
            if "node_type_predictor" in state:
                try:
                    node_type_predictor.load_state_dict(state["node_type_predictor"])
                    print(f"  Successfully loaded weights into {type(node_type_predictor).__name__}.")
                    loaded_something = True
                except RuntimeError as e:
                    print(f"  Warning: Error loading node_type_predictor weights (possibly architecture mismatch): {e}")
                    print("  Attempting load with strict=False...")
                    try:
                        node_type_predictor.load_state_dict(state["node_type_predictor"], strict=False)
                        print(f"  Partially loaded weights into {type(node_type_predictor).__name__} with strict=False.")
                        loaded_something = True
                    except Exception as e_strict:
                        print(f"  Error loading node_type_predictor weights even with strict=False: {e_strict}")
                except Exception as e_other:
                    print(f"  Error loading node_type_predictor weights: {e_other}")
            else:
                print(f"  Warning: Key 'node_type_predictor' not found in checkpoint. Skipping node predictor weights.")
        # --- End Loading ---

        if loaded_something:
             print("Finished loading weights for fine-tuning.")
             return True
        else:
             print("Warning: No model weights were successfully loaded from the checkpoint.")
             return False

    except Exception as e:
        print(f"Error loading checkpoint file {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Refactored setup_models ---
def setup_models(config, device, max_node_count, max_level, train_node_type_flag=False):
    """
    Sets up node, edge, and optional independent node type predictor models using mappings.

    Args:
        config: Dictionary with model configuration.
        device: PyTorch device to place models on.
        max_node_count: Maximum number of nodes in the training graphs.
        max_level: Maximum level in the training graphs.
        train_node_type_flag: Boolean, if True, set up the independent node predictor.

    Returns:
        node_model: Instantiated node-level model.
        edge_model: Instantiated edge-level model.
        step_fn: Training step function to use for node/edge models.
        node_type_predictor: Instantiated node type predictor model (or None).
    """
    # --- Mappings for Models and Config Sections ---
    NODE_MODEL_MAP = {
        # (is_lstm, use_attention): (ModelClass, config_section_name)
        (False, False): (GraphLevelRNN, 'GraphRNN'),
        (False, True):  (GraphLevelAttentionRNN, 'GraphAttentionRNN'),
        (True, False):  (GraphLevelLSTM, 'GraphLSTM'),
        (True, True):   (GraphLevelAttentionLSTM, 'GraphAttentionLSTM'),
    }
    EDGE_MODEL_MAP = {
        # edge_model_choice: (ModelClass, config_section_name, step_function)
        'mlp': (EdgeLevelMLP, 'EdgeMLP', train_mlp_step),
        'rnn': (None, 'EdgeRNN', train_rnn_step), # Class determined later by is_lstm
        'attention_rnn': (None, 'EdgeAttentionRNN', train_rnn_step), # Class determined later
    }
    EDGE_RNN_MAP = {
         # (is_lstm, use_attention): ModelClass
         (False, False): EdgeLevelRNN,
         (False, True):  EdgeLevelAttentionRNN,
         (True, False):  EdgeLevelLSTM,
         (True, True):   EdgeLevelAttentionLSTM,
    }

    # --- Determine Effective Input Size ---
    use_bfs = config['data'].get('use_bfs', False)
    if use_bfs:
        effective_input_size = config['data'].get('m')
        if effective_input_size is None: raise ValueError("Config 'data.m' required for BFS.")
        mode_info = f"BFS mode, m={effective_input_size}"
    else: # TopSort mode
        if max_node_count <= 1: raise ValueError("max_node_count must be > 1 for training.")
        effective_input_size = max_node_count - 1
        mode_info = f"TopSort mode, eff_input={effective_input_size}"
    print(f"INFO (utils.py): Setting up models ({mode_info})")


    # --- Node Model Setup ---
    use_lstm = config['model'].get('use_lstm', False)
    use_node_attention = config['model'].get('use_attention', False)
    node_key = (use_lstm, use_node_attention)

    if node_key not in NODE_MODEL_MAP:
        raise ValueError(f"Invalid node model configuration: use_lstm={use_lstm}, use_attention={use_node_attention}")

    NodeModelClass, node_config_section = NODE_MODEL_MAP[node_key]

    # Fallback for AttentionRNN config section
    if node_config_section == 'GraphAttentionRNN' and node_config_section not in config['model'] and 'GraphRNN' in config['model']:
        print(f"Warning (utils.py): Config section 'model.GraphAttentionRNN' not found, using 'GraphRNN'.")
        node_config_section = 'GraphRNN'

    if node_config_section not in config['model']:
        raise ValueError(f"Config missing required section 'model.{node_config_section}'.")

    node_config_specific = config['model'][node_config_section]
    node_params = {
        'input_size': effective_input_size,
        'max_level': max_level,
        'predict_node_types': False, # Independent predictor handled separately
        'num_node_types': None,
        'use_conditioning': False,
        'tt_size': None,
        **node_config_specific # Add specific params from config
    }

    # Ensure required params exist and clean up unused ones
    node_edge_feature_len = node_params.get('edge_feature_len')
    if node_edge_feature_len is None:
        raise ValueError(f"Config 'model.{node_config_section}.edge_feature_len' is required.")
    if not use_node_attention: # Remove attention params if not used by the selected class
        node_params.pop('attention_heads', None)
        node_params.pop('attention_dropout', None)

    # Instantiate node model
    node_model = NodeModelClass(**node_params).to(device)
    print(f"INFO (utils.py): Using Node Model: {NodeModelClass.__name__}")

    # Get necessary outputs from node model setup
    node_hidden_size = node_params.get('hidden_size')
    node_model_output_size_for_edge = node_params.get('output_size') # May be None


    # --- Edge Model Setup ---
    edge_model_choice = config['model'].get('edge_model', 'mlp').lower()
    if edge_model_choice not in EDGE_MODEL_MAP:
        raise ValueError(f"Unsupported edge_model type in config: {edge_model_choice}")

    EdgeModelClass, edge_config_section, step_fn = EDGE_MODEL_MAP[edge_model_choice]
    is_edge_attention = edge_model_choice.startswith('attention') # Only relevant for rnn types

    # Determine specific RNN/LSTM class if needed
    if EdgeModelClass is None: # i.e., edge_model_choice is 'rnn' or 'attention_rnn'
        edge_key = (use_lstm, is_edge_attention) # Edge type depends on node's lstm flag and edge attention flag
        if edge_key not in EDGE_RNN_MAP:
             raise ValueError(f"Invalid edge RNN configuration: use_lstm={use_lstm}, is_edge_attention={is_edge_attention}")
        EdgeModelClass = EDGE_RNN_MAP[edge_key]
        # Fallback for AttentionRNN config section
        if edge_config_section == 'EdgeAttentionRNN' and edge_config_section not in config['model'] and 'EdgeRNN' in config['model']:
             print(f"Warning (utils.py): Config section 'model.EdgeAttentionRNN' not found, using 'EdgeRNN'.")
             edge_config_section = 'EdgeRNN'

    if edge_config_section not in config['model']:
        raise ValueError(f"Config missing required section 'model.{edge_config_section}'.")

    edge_config_specific = config['model'][edge_config_section]
    edge_params = {
        'edge_feature_len': node_edge_feature_len, # Inherit from node model config
         **edge_config_specific # Add specific params from config
    }

    # Add/adjust params specific to edge model type
    if edge_model_choice == 'mlp':
        if node_hidden_size is None: raise ValueError(f"Node model must define 'hidden_size' for EdgeMLP.")
        edge_params['input_size'] = node_hidden_size
        edge_params['output_size'] = effective_input_size
    else: # RNN-based edge models
        edge_hidden_size = edge_params.get('hidden_size')
        if edge_hidden_size is None: raise ValueError(f"Edge model config must define 'hidden_size'.")
        if node_model_output_size_for_edge is None: raise ValueError(f"Node model config must define 'output_size' for RNN edge models.")
        if node_model_output_size_for_edge != edge_hidden_size:
            raise ValueError(f"Mismatch: Node output_size ({node_model_output_size_for_edge}) != Edge hidden_size ({edge_hidden_size}).")
        # Handle attention params for RNNs
        if not is_edge_attention:
            edge_params.pop('attention_heads', None)
            edge_params.pop('attention_dropout', None)
        elif is_edge_attention and edge_config_section in ['EdgeRNN', 'EdgeLSTM']: # Add defaults if using fallback section
            edge_params['attention_heads'] = edge_params.get('attention_heads', 4)
            edge_params['attention_dropout'] = edge_params.get('attention_dropout', 0.1)

    # Instantiate edge model
    edge_model = EdgeModelClass(**edge_params).to(device)
    print(f"INFO (utils.py): Using Edge Model: {EdgeModelClass.__name__}")


    # --- Independent Node Type Predictor Setup ---
    node_type_predictor = None
    if train_node_type_flag:
        print("INFO (utils.py): Setting up independent node type predictor...")
        if node_hidden_size is None: # Should have been determined during node model setup
            raise ValueError("Cannot determine node model hidden_size for NodeTypePredictor.")

        print(f"  Using node model hidden size: {node_hidden_size} for predictor.")
        print(f"  Number of node types: {NUM_NODE_TYPES}")
        try:
            node_type_predictor = NodeTypePredictor(
                hidden_size=node_hidden_size,
                num_node_types=NUM_NODE_TYPES
            ).to(device)
            print("INFO (utils.py): Independent node type predictor instantiated.")
        except ImportError:
             print("\nFATAL ERROR (utils.py): Could not import NodeTypePredictor from model.py."); raise
        except Exception as e_pred:
             print(f"\nFATAL ERROR (utils.py): Failed to instantiate NodeTypePredictor: {e_pred}\n"); raise


    # --- Return all models and step function ---
    return node_model, edge_model, step_fn, node_type_predictor


def setup_criteria(config, device, dataset, train_node_type_flag=False): # <-- Added flag
    """
    Sets up loss criteria for edges and optionally for node types.

    Args:
        config: Dictionary with model configuration.
        device: PyTorch device.
        dataset: The dataset object (used for edge weights).
        train_node_type_flag: Boolean, if True, set up the node criterion.

    Returns:
        criterion_edge: Loss function for edge prediction.
        use_edge_features: Boolean indicating if edge loss is multi-class.
        criterion_node: Loss function for node type prediction (or None).
    """
    criterion_node = None # Initialize as None

    # --- Edge Criterion Setup (Existing Logic) ---
    # Determine edge_feature_len based on the config section of the *chosen* node model
    use_lstm = config['model'].get('use_lstm', False)
    use_node_attention = config['model'].get('use_attention', False)
    node_config_section = None
    if use_lstm: node_config_section = 'GraphAttentionLSTM' if use_node_attention else 'GraphLSTM'
    else: # GRU
        if use_node_attention: node_config_section = 'GraphAttentionRNN'
        if node_config_section not in config['model'] and 'GraphRNN' in config['model']: node_config_section = 'GraphRNN'
        else: node_config_section = 'GraphRNN'
    # Check if section exists (should be guaranteed by setup_models if called first)
    if node_config_section not in config['model']: node_config_section = 'GraphRNN' # Fallback guess

    if node_config_section not in config['model']:
        print(f"Warning (utils.py): Cannot find node section '{node_config_section}' in setup_criteria. Defaulting edge_feature_len.")
        edge_feature_len = NUM_EDGE_FEATURES
    else:
        edge_feature_len = config['model'][node_config_section].get('edge_feature_len')
        if edge_feature_len is None: raise ValueError(f"Config 'model.{node_config_section}.edge_feature_len' required.")

    use_edge_features = edge_feature_len > 1
    if use_edge_features:
        print(f"INFO (utils.py): Setting up CrossEntropyLoss for {edge_feature_len} edge classes.")
        edge_weights = getattr(dataset, 'edge_weights', None) # Safely get weights
        if edge_weights is not None:
            edge_weights = edge_weights.to(device)
            print(f"  Applying edge class weights: {edge_weights.tolist()}")
            criterion_edge = CrossEntropyLoss(weight=edge_weights).to(device)
        else:
            print("  Warning: Dataset has no 'edge_weights'. Using uniform weights.")
            criterion_edge = CrossEntropyLoss().to(device)
    else:
        print("INFO (utils.py): Using BCELoss for binary edges.")
        criterion_edge = torch.nn.BCELoss().to(device)
    # --- End Edge Criterion Setup ---


    # --- NEW: Node Criterion Setup (Conditional) ---
    if train_node_type_flag:
        print(f"INFO (utils.py): Setting up CrossEntropyLoss for {NUM_NODE_TYPES} node types.")
        # Use ignore_index=-1 if your dataset uses -1 for padding node types
        criterion_node = CrossEntropyLoss(ignore_index=-1).to(device)
        print("  Node type criterion set up.")
    # --- End Node Criterion Setup ---

    return criterion_edge, use_edge_features, criterion_node # <-- Return node criterion