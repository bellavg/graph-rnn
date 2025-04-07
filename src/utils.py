from model import *
from aig_dataset import *
from train import *

def setup_models(config, device, max_node_count, max_level):
    use_bfs = config['data'].get('use_bfs', False)
    # Determine default edge feature len more robustly
    default_edge_feature_len = config.get('model',{}).get('GraphRNN', {}).get('edge_feature_len', NUM_EDGE_FEATURES)
    default_edge_feature_len = config.get('model',{}).get('GraphLSTM', {}).get('edge_feature_len', default_edge_feature_len)


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
    use_attention = config['model'].get('use_attention', False) # Assuming this applies to node model
    edge_model_choice = config['model'].get('edge_model', 'mlp').lower() # 'mlp', 'rnn', 'attention_rnn'

    node_model = None
    edge_model = None
    step_fn = None
    node_model_output_size_for_edge = None # Will be set after node model creation

    # --- Initialize Node Model ---
    node_params = {}
    node_params['input_size'] = effective_input_size
    node_params['max_level'] = max_level
    node_params['predict_node_types'] = False # Hardcoded for AIG
    node_params['num_node_types'] = None
    node_params['use_conditioning'] = False
    node_params['tt_size'] = None

    if use_lstm:
        if use_attention:
            # GraphLevelAttentionLSTM
            config_section = 'GraphAttentionLSTM'
            if config_section not in config['model']: raise ValueError(f"Config missing 'model.{config_section}'.")
            node_config = config['model'][config_section]
            node_params.update(node_config) # Add params from config section
            node_model = GraphLevelAttentionLSTM(**node_params).to(device)
            node_model_output_size_for_edge = node_params.get('output_size')
            print(f"INFO: Using GraphLevelAttentionLSTM for node level.")
        else:
            # GraphLevelLSTM
            config_section = 'GraphLSTM'
            if config_section not in config['model']: raise ValueError(f"Config missing 'model.{config_section}'.")
            node_config = config['model'][config_section]
            node_params.update(node_config)
            node_model = GraphLevelLSTM(**node_params).to(device)
            node_model_output_size_for_edge = node_params.get('output_size')
            print(f"INFO: Using GraphLevelLSTM for node level.")
    else: # Use GRU
        if use_attention:
            # GraphLevelAttentionRNN
            config_section = 'GraphAttentionRNN' # Might need a dedicated section like GraphAttentionLSTM
            # Fallback to GraphRNN section if dedicated one doesn't exist
            if config_section not in config['model'] and 'GraphRNN' in config['model']:
                 print(f"Warning: Config section 'model.{config_section}' not found, using 'GraphRNN' section for AttentionRNN.")
                 config_section = 'GraphRNN'
            elif config_section not in config['model']:
                 raise ValueError(f"Config missing 'model.{config_section}' or 'model.GraphRNN'.")

            node_config = config['model'][config_section]
            # Ensure attention params are present if using GraphRNN section
            node_params.update(node_config)
            node_params['attention_heads'] = node_params.get('attention_heads', 4) # Add defaults if needed
            node_params['attention_dropout'] = node_params.get('attention_dropout', 0.1)
            node_model = GraphLevelAttentionRNN(**node_params).to(device)
            node_model_output_size_for_edge = node_params.get('output_size')
            print(f"INFO: Using GraphLevelAttentionRNN for node level.")
        else:
            # GraphLevelRNN
            config_section = 'GraphRNN'
            if config_section not in config['model']: raise ValueError(f"Config missing 'model.{config_section}'.")
            node_config = config['model'][config_section]
            node_params.update(node_config)
            # Remove attention keys if they accidentally exist in this section for base GRU
            node_params.pop('attention_heads', None)
            node_params.pop('attention_dropout', None)
            node_model = GraphLevelRNN(**node_params).to(device)
            node_model_output_size_for_edge = node_params.get('output_size')
            print(f"INFO: Using standard GraphLevelRNN for node level.")

    # --- Initialize Edge Model ---
    edge_params = {}
    # Get edge_feature_len from the chosen node model config if possible, else use default
    edge_params['edge_feature_len'] = node_params.get('edge_feature_len', default_edge_feature_len)


    if edge_model_choice == 'mlp':
        config_section = 'EdgeMLP'
        if config_section not in config['model']: raise ValueError(f"Config missing 'model.{config_section}'.")
        if node_params.get('hidden_size') is None: raise ValueError("Node model hidden_size needed for EdgeMLP input.")

        edge_config = config['model'][config_section]
        edge_params.update(edge_config)
        edge_params['output_size'] = effective_input_size
        edge_params['input_size'] = node_params['hidden_size'] # Input from node model's hidden state
        edge_model = EdgeLevelMLP(**edge_params).to(device)
        step_fn = train_mlp_step
        print("Selected EdgeLevelMLP model.")

    # Logic for RNN/LSTM based edge models
    elif edge_model_choice in ['rnn', 'lstm', 'attention_rnn', 'attention_lstm']:
        is_edge_lstm = use_lstm # Assume edge uses same RNN type as node, or add separate edge flag
        is_edge_attention = edge_model_choice.startswith('attention')

        if is_edge_lstm:
            if is_edge_attention:
                # EdgeLevelAttentionLSTM
                config_section = 'EdgeAttentionLSTM'
                if config_section not in config['model']: raise ValueError(f"Config missing 'model.{config_section}'.")
                edge_config = config['model'][config_section]
                edge_params.update(edge_config)
                # Check if node model provides the required output size
                if node_model_output_size_for_edge is None or node_model_output_size_for_edge != edge_params.get('hidden_size'):
                     raise ValueError(f"Node model output_size ({node_model_output_size_for_edge}) must be defined and match {config_section}.hidden_size ({edge_params.get('hidden_size')})")
                edge_model = EdgeLevelAttentionLSTM(**edge_params).to(device)
                step_fn = train_rnn_step # Or train_lstm_step
                print(f"Selected EdgeLevelAttentionLSTM model.")
            else:
                # EdgeLevelLSTM
                config_section = 'EdgeLSTM'
                if config_section not in config['model']: raise ValueError(f"Config missing 'model.{config_section}'.")
                edge_config = config['model'][config_section]
                edge_params.update(edge_config)
                if node_model_output_size_for_edge is None or node_model_output_size_for_edge != edge_params.get('hidden_size'):
                     raise ValueError(f"Node model output_size ({node_model_output_size_for_edge}) must be defined and match {config_section}.hidden_size ({edge_params.get('hidden_size')})")
                edge_model = EdgeLevelLSTM(**edge_params).to(device)
                step_fn = train_rnn_step # Or train_lstm_step
                print(f"Selected EdgeLevelLSTM model.")
        else: # Use GRU for edge
             if is_edge_attention:
                # EdgeLevelAttentionRNN
                config_section = 'EdgeAttentionRNN' # Use dedicated section if available
                if config_section not in config['model'] and 'EdgeRNN' in config['model']:
                    print(f"Warning: Config section 'model.{config_section}' not found, using 'EdgeRNN' section for AttentionRNN.")
                    config_section = 'EdgeRNN'
                elif config_section not in config['model']:
                    raise ValueError(f"Config missing 'model.{config_section}' or 'model.EdgeRNN'.")

                edge_config = config['model'][config_section]
                edge_params.update(edge_config)
                # Ensure attention params are present if using EdgeRNN section
                edge_params['attention_heads'] = edge_params.get('attention_heads', 4)
                edge_params['attention_dropout'] = edge_params.get('attention_dropout', 0.1)
                if node_model_output_size_for_edge is None or node_model_output_size_for_edge != edge_params.get('hidden_size'):
                     raise ValueError(f"Node model output_size ({node_model_output_size_for_edge}) must be defined and match {config_section}.hidden_size ({edge_params.get('hidden_size')})")
                edge_model = EdgeLevelAttentionRNN(**edge_params).to(device)
                step_fn = train_rnn_step
                print(f"Selected EdgeLevelAttentionRNN model.")
             else:
                 # EdgeLevelRNN (GRU)
                 config_section = 'EdgeRNN'
                 if config_section not in config['model']: raise ValueError(f"Config missing 'model.{config_section}'.")
                 edge_config = config['model'][config_section]
                 edge_params.update(edge_config)
                 # Remove attention keys if accidentally present
                 edge_params.pop('attention_heads', None)
                 edge_params.pop('attention_dropout', None)
                 if node_model_output_size_for_edge is None or node_model_output_size_for_edge != edge_params.get('hidden_size'):
                      raise ValueError(f"Node model output_size ({node_model_output_size_for_edge}) must be defined and match {config_section}.hidden_size ({edge_params.get('hidden_size')})")
                 edge_model = EdgeLevelRNN(**edge_params).to(device)
                 step_fn = train_rnn_step
                 print(f"Selected standard EdgeLevelRNN (GRU) model.")

    else:
        raise ValueError(f"Unsupported edge_model type: {edge_model_choice}")

    # Final check for step function
    if step_fn is None:
        raise RuntimeError("Step function (step_fn) was not assigned during model setup.")

    return node_model, edge_model, step_fn