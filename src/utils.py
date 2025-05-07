"""
Utility functions for the AIG model setup and evaluation.
Combines functions from multiple sources for a unified interface.
"""

import torch
from .aig_dataset import NUM_EDGE_FEATURES_RNN


def setup_models(config, device, max_node_count, max_level):
    """
    Sets up node and edge models based on the provided configuration.

    Args:
        config: Dictionary with model configuration
        device: PyTorch device to place models on
        max_node_count: Maximum number of nodes in the training graphs
        max_level: Maximum level in the training graphs

    Returns:
        node_model: Instantiated node-level model
        edge_model: Instantiated edge-level model
        step_fn: Training step function to use
    """
    use_bfs = config['data'].get('use_bfs', False)

    # Determine Effective Input/Output Size
    if use_bfs:
        effective_input_size = config['data'].get('m')
        if effective_input_size is None: raise ValueError("Config 'data.m' required for BFS.")
        print(f"INFO: Using BFS mode. Effective input/output size (m): {effective_input_size}")
    else:  # TopSort mode
        if max_node_count <= 1: raise ValueError("max_node_count must be > 1 for training.")
        effective_input_size = max_node_count - 1
        print(f"INFO: Using Topological Sort mode. Effective input/output size (max_nodes-1): {effective_input_size}")

    # --- Model Selection Flags ---
    use_lstm = config['model'].get('use_lstm', False)
    # Use 'use_attention' for node model attention specifically
    use_node_attention = config['model'].get('use_attention', False)
    edge_model_choice = config['model'].get('edge_model', 'mlp').lower()  # 'mlp', 'rnn', 'attention_rnn'

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
    else:  # GRU
        # Allow dedicated AttentionRNN section, fallback to GraphRNN
        if use_node_attention:
            node_config_section = 'GraphAttentionRNN'
            if node_config_section not in config['model'] and 'GraphRNN' in config['model']:
                print(
                    f"Warning: Config section 'model.{node_config_section}' not found, using 'GraphRNN'. Add 'attention_heads/dropout' there if needed.")
                node_config_section = 'GraphRNN'
        else:
            node_config_section = 'GraphRNN'

    if node_config_section not in config['model']:
        raise ValueError(f"Config missing required model section 'model.{node_config_section}'.")

    node_config = config['model'][node_config_section]
    node_params.update(node_config)  # Base params from chosen section

    # Add common/derived params
    node_params['input_size'] = effective_input_size
    node_params['max_level'] = max_level
    node_params['predict_node_types'] = False
    node_params['num_node_types'] = None
    node_params['use_conditioning'] = False
    node_params['tt_size'] = None
    # Store edge_feature_len from node config for later use
    node_edge_feature_len = node_params.get('edge_feature_len')
    if node_edge_feature_len is None:
        raise ValueError(f"'{node_config_section}.edge_feature_len' must be defined.")

    # Store output size needed for edge model init
    node_model_output_size_for_edge = node_params.get('output_size')  # May be None if edge='mlp'

    # --- Instantiate Node Model ---
    if use_lstm:
        if use_node_attention:
            from .model import GraphLevelAttentionLSTM
            node_model = GraphLevelAttentionLSTM(**node_params).to(device)
            print(f"INFO: Using GraphLevelAttentionLSTM for node level.")
        else:
            # Ensure attention keys aren't accidentally passed
            node_params.pop('attention_heads', None)
            node_params.pop('attention_dropout', None)
            from .model import GraphLevelLSTM
            node_model = GraphLevelLSTM(**node_params).to(device)
            print(f"INFO: Using GraphLevelLSTM for node level.")
    else:  # GRU
        if use_node_attention:
            from .model import GraphLevelAttentionRNN
            node_model = GraphLevelAttentionRNN(**node_params).to(device)
            print(f"INFO: Using GraphLevelAttentionRNN for node level.")
        else:
            # Ensure attention keys aren't passed
            node_params.pop('attention_heads', None)
            node_params.pop('attention_dropout', None)
            node_params.pop('use_attention', None)  # Remove the flag itself if it exists
            from .model import GraphLevelRNN
            node_model = GraphLevelRNN(**node_params).to(device)
            print(f"INFO: Using standard GraphLevelRNN for node level.")

    # --- Initialize Edge Model ---
    edge_params['edge_feature_len'] = node_edge_feature_len  # Use value from node config

    if edge_model_choice == 'mlp':
        config_section = 'EdgeMLP'
        if config_section not in config['model']: raise ValueError(f"Config missing 'model.{config_section}'.")
        # MLP input size depends on the actual hidden size of the node model instantiated
        node_hidden_size = node_params.get('hidden_size')
        if node_hidden_size is None: raise ValueError(
            f"Node model section '{node_config_section}' must define 'hidden_size' for EdgeMLP.")

        edge_config = config['model'][config_section]
        edge_params.update(edge_config)
        edge_params['output_size'] = effective_input_size
        edge_params['input_size'] = node_hidden_size  # Input FROM node model's internal hidden state
        # Ensure EdgeMLP config has its own edge_feature_len
        if 'edge_feature_len' not in edge_params: edge_params['edge_feature_len'] = node_edge_feature_len

        from .model import EdgeLevelMLP
        edge_model = EdgeLevelMLP(**edge_params).to(device)
        from .train import train_mlp_step
        step_fn = train_mlp_step
        print("Selected EdgeLevelMLP model.")

    elif edge_model_choice in ['rnn', 'attention_rnn']:
        # Determine if edge model should be LSTM based on node model flag
        is_edge_lstm = use_lstm
        # Determine if edge model should have attention based on edge_model_choice
        is_edge_attention = edge_model_choice.startswith('attention')

        # Determine edge config section and class
        EdgeModelClass = None
        if is_edge_lstm:
            if is_edge_attention:
                config_section = 'EdgeAttentionLSTM'
                from .model import EdgeLevelAttentionLSTM
                EdgeModelClass = EdgeLevelAttentionLSTM
                from .train import train_rnn_step
                step_fn = train_rnn_step  # Assumes train_rnn_step handles LSTM state tuple
                # Or: step_fn = train_lstm_step
            else:
                config_section = 'EdgeLSTM'
                from .model import EdgeLevelLSTM
                EdgeModelClass = EdgeLevelLSTM
                from .train import train_rnn_step
                step_fn = train_rnn_step  # Or train_lstm_step
        else:  # GRU Edge Model
            if is_edge_attention:
                # Allow dedicated AttentionRNN section, fallback to EdgeRNN
                config_section = 'EdgeAttentionRNN'
                if config_section not in config['model'] and 'EdgeRNN' in config['model']:
                    print(
                        f"Warning: Config section 'model.{config_section}' not found, using 'EdgeRNN'. Add 'attention_heads/dropout' there if needed.")
                    config_section = 'EdgeRNN'
                from .model import EdgeLevelAttentionRNN
                EdgeModelClass = EdgeLevelAttentionRNN
                from .train import train_rnn_step
                step_fn = train_rnn_step
            else:
                config_section = 'EdgeRNN'
                from .model import EdgeLevelRNN
                EdgeModelClass = EdgeLevelRNN
                from .train import train_rnn_step
                step_fn = train_rnn_step

        # Load edge config and instantiate
        if config_section not in config['model']: raise ValueError(f"Config missing 'model.{config_section}'.")
        edge_config = config['model'][config_section]
        edge_params.update(edge_config)

        # Add attention defaults if needed and using fallback section
        if is_edge_attention and config_section == 'EdgeRNN':
            edge_params['attention_heads'] = edge_params.get('attention_heads', 4)
            edge_params['attention_dropout'] = edge_params.get('attention_dropout', 0.1)
        # Remove attention keys if base RNN/LSTM selected
        elif not is_edge_attention:
            edge_params.pop('attention_heads', None)
            edge_params.pop('attention_dropout', None)

        # --- Critical Check: Node Output -> Edge Hidden ---
        edge_hidden_size = edge_params.get('hidden_size')
        if edge_hidden_size is None:
            raise ValueError(f"Edge model section '{config_section}' must define 'hidden_size'.")
        if node_model_output_size_for_edge is None:
            raise ValueError(
                f"Node model section '{node_config_section}' must define 'output_size' when edge model is '{edge_model_choice}'.")
        if node_model_output_size_for_edge != edge_hidden_size:
            raise ValueError(
                f"Mismatch: Node model output_size ({node_model_output_size_for_edge}) != Edge model hidden_size ({edge_hidden_size}). Check config sections '{node_config_section}' and '{config_section}'.")
        # --- End Check ---

        edge_model = EdgeModelClass(**edge_params).to(device)
        print(f"Selected {EdgeModelClass.__name__} model.")

    else:
        raise ValueError(f"Unsupported edge_model type in config: {edge_model_choice}")

    # Final check
    if node_model is None or edge_model is None or step_fn is None:
        raise RuntimeError("Failed to initialize node_model, edge_model, or step_fn.")

    return node_model, edge_model, step_fn


def setup_criteria(config, device, dataset):
    """ Sets up loss criterion for edges, potentially using class weights from dataset. """

    # Determine edge_feature_len based on the config section of the *chosen* node model
    use_lstm = config['model'].get('use_lstm', False)
    use_node_attention = config['model'].get('use_attention', False)
    node_config_section = None
    if use_lstm:
        node_config_section = 'GraphAttentionLSTM' if use_node_attention else 'GraphLSTM'
    else:  # GRU
        if use_node_attention:
            node_config_section = 'GraphAttentionRNN'
            if node_config_section not in config['model'] and 'GraphRNN' in config['model']:
                node_config_section = 'GraphRNN'
        else:
            node_config_section = 'GraphRNN'

    if node_config_section not in config['model']:
        # Fallback or error if primary section not found (should have been caught in setup_models)
        # Using NUM_EDGE_FEATURES as a last resort default
        print(f"Warning: Could not find section {node_config_section} in setup_criteria. Defaulting edge_feature_len.")
        edge_feature_len = NUM_EDGE_FEATURES_RNN
    else:
        # Get edge_feature_len from the correct node model's config section
        edge_feature_len = config['model'][node_config_section].get('edge_feature_len')
        if edge_feature_len is None:
            raise ValueError(f"Config 'model.{node_config_section}.edge_feature_len' is required.")

    # Now determine loss based on edge_feature_len
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

    # Return only edge criterion and flag indicating if multi-class features are used
    return criterion_edge, use_edge_features