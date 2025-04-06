import argparse
import yaml
import torch
import os
import time
import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import pickle

# Assuming these imports are correct relative to your project structure
from train import train_rnn_step, train_mlp_step
from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP
from aig_dataset import AIGDataset, NUM_EDGE_FEATURES # Import NUM_EDGE_FEATURES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="configs/config_aig_base.yaml",
                        help='Path of the config file to use for training')
    parser.add_argument('-r', '--restore', dest='restore_path', default=None,
                        help='Checkpoint to continue training from')
    parser.add_argument('--gpu', dest='gpu_id', default=0, type=int,
                        help='Id of the GPU to use')
    parser.add_argument('--save_dir', dest='save_dir', default="./runs", type=str)
    return parser.parse_args()


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print("Loaded config:")
    print(yaml.dump(config, sort_keys=False))
    # Basic config validation
    if 'data' not in config or 'model' not in config or 'train' not in config:
        raise ValueError("Config file must contain 'data', 'model', and 'train' sections.")
    if 'use_bfs' not in config['data']:
         raise ValueError("Config must specify 'data.use_bfs' (true or false).")
    if config['data']['use_bfs'] and 'm' not in config['data']:
         raise ValueError("Config must specify 'data.m' when 'data.use_bfs' is true.")
    return config


# MODIFIED setup_models function
def setup_models(config, device, max_node_count, max_level): # ADDED max_level
    """
    Initializes the GraphRNN and Edge models based on the config,
    using the correct dimensions derived from max_node_count and use_bfs,
    and passing max_level for positional encoding. Reads rnn_type for both models.
    """
    # Determine use_bfs (hardcoded to False based on previous dataset modification)
    use_bfs = False # Keep consistent with modified AIGDataset
    edge_feature_len = config['model']['GraphRNN'].get('edge_feature_len', NUM_EDGE_FEATURES)

    # Determine Effective Input/Output Size for TopSort
    if max_node_count <= 1:
        raise ValueError("max_node_count must be greater than 1 for training.")
    effective_input_size = max_node_count - 1
    # print(f"INFO: Using Topological Sort mode. Effective input/output size (max_node_count - 1): {effective_input_size}")

    # --- Keep Optional features setup ---
    predict_node_types = config['model'].get('predict_node_types', False)
    use_conditioning = config['model'].get('truth_table_conditioning', False)
    num_node_types = config['model'].get('num_node_types', None)
    tt_size = None
    if use_conditioning:
        n_outputs = config['model'].get('n_outputs', 2); n_inputs = config['model'].get('n_inputs', 8)
        tt_size = n_outputs * (2 ** n_inputs); # print(f"TT conditioning enabled: size {tt_size}")


    # --- Initialize Edge Model and select Step Function ---
    edge_model_type = config['model'].get('edge_model', 'mlp').lower()
    node_model_output_size = None # Default for MLP
    edge_model = None
    step_fn = None

    if edge_model_type == 'rnn':
        if 'EdgeRNN' not in config['model']: raise ValueError("Config missing 'model.EdgeRNN'.")
        edge_model_args = config['model']['EdgeRNN'].copy()
        edge_model_args['edge_feature_len'] = edge_feature_len
        edge_model_args['tt_size'] = tt_size
        # --- ADDED: Read rnn_type for EdgeLevelRNN ---
        edge_model_args['rnn_type'] = config['model']['EdgeRNN'].get('rnn_type', 'gru')
        # --- END ADDED ---

        edge_model = EdgeLevelRNN(**edge_model_args).to(device) # Pass all args
        step_fn = train_rnn_step
        node_model_output_size = edge_model_args.get('hidden_size')
        if node_model_output_size is None: raise ValueError("Config 'model.EdgeRNN.hidden_size' needed.")
        # print(f"Selected EdgeLevelRNN model (type: {edge_model_args['rnn_type']}).")

    elif edge_model_type == 'mlp':
        # ... (MLP setup as before, no rnn_type needed here) ...
        if 'EdgeMLP' not in config['model']: raise ValueError("Config missing 'model.EdgeMLP'.")
        if 'GraphRNN' not in config['model'] or 'hidden_size' not in config['model']['GraphRNN']:
             raise ValueError("Config 'model.GraphRNN.hidden_size' needed.")
        edge_model_args = config['model']['EdgeMLP'].copy()
        edge_model_args['edge_feature_len'] = edge_feature_len
        edge_model_args['tt_size'] = tt_size
        edge_model_args['output_size'] = effective_input_size
        edge_model_args['input_size'] = config['model']['GraphRNN']['hidden_size']
        edge_model = EdgeLevelMLP(**edge_model_args).to(device)
        step_fn = train_mlp_step
        node_model_output_size = None
        # print("Selected EdgeLevelMLP model.")
    else:
        raise ValueError(f"Unsupported edge_model type: {edge_model_type}")

    # --- Initialize Node Model ---
    if 'GraphRNN' not in config['model']: raise ValueError("Config missing 'model.GraphRNN'.")
    node_model_args = config['model']['GraphRNN'].copy()
    node_model_args['input_size'] = effective_input_size
    node_model_args['edge_feature_len'] = edge_feature_len
    node_model_args['output_size'] = node_model_output_size
    node_model_args['predict_node_types'] = predict_node_types
    node_model_args['num_node_types'] = num_node_types
    node_model_args['tt_size'] = tt_size
    node_model_args['max_level'] = max_level

    # --- Read rnn_type for GraphLevelRNN (already correct) ---
    node_model_args['rnn_type'] = config['model']['GraphRNN'].get('rnn_type', 'gru')

    # Initialize GraphLevelRNN with all arguments
    node_model = GraphLevelRNN(**node_model_args).to(device)
    # print(f"Initialized GraphLevelRNN model (type: {node_model_args['rnn_type']}).")

    return node_model, edge_model, step_fn


def get_max_node_count_from_pkl(graph_file: str) -> int:
    """
    Efficiently loads raw graphs from a pickle file and finds the maximum node count.
    """

    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Dataset file not found: {graph_file}")

    max_nodes = 0
    loaded_count = 0
    try:
        with open(graph_file, 'rb') as f:
            raw_graphs = pickle.load(f)

        num_to_check = len(raw_graphs)

        for i, g in enumerate(raw_graphs):
            if i >= num_to_check:
                break
            # Basic check if it looks like a graph object
            if hasattr(g, 'number_of_nodes'):
                max_nodes = max(max_nodes, g.number_of_nodes())
                loaded_count += 1
            else:
                print(f"Warning: Item {i} in pickle file doesn't seem to be a graph. Skipping.")

        if loaded_count == 0:
             raise ValueError("No valid graph objects found in the pickle file.")

    except (pickle.UnpicklingError, EOFError) as e:
        raise IOError(f"Error reading pickle file {graph_file}: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while reading {graph_file}: {e}")

    #print(f"Max node count found: {max_nodes}")
    return max_nodes


def setup_criteria(config, device):
    # Assumes edge features length > 1 means multiclass classification
    # Make sure NUM_EDGE_FEATURES reflects the actual classes (e.g., 3 for AIG)
    use_edge_features = config['model']['GraphRNN'].get('edge_feature_len', NUM_EDGE_FEATURES) > 1
    if use_edge_features:
         print(f"Using CrossEntropyLoss for {config['model']['GraphRNN'].get('edge_feature_len', NUM_EDGE_FEATURES)} edge classes.")
         # reduction='mean' is default
         criterion_edge = torch.nn.CrossEntropyLoss().to(device)
    else:
         # Binary case (not typical for AIG multiclass)
         print("Using BCELoss for binary edges.")
         criterion_edge = torch.nn.BCELoss().to(device)

    # Node criterion only if predicting node types
    criterion_node = None
    if config['model'].get('predict_node_types', False):
        # Use ignore_index to handle padding in labels if necessary
        criterion_node = torch.nn.CrossEntropyLoss(ignore_index=-100).to(device)
        print("Node type prediction enabled. Using CrossEntropyLoss for nodes.")

    return criterion_edge, criterion_node, use_edge_features


def restore_checkpoint(path, device, node_model, edge_model,
                       optim_node_model, optim_edge_model,
                       scheduler_node_model, scheduler_edge_model): # Removed criteria from args
    print(f"Restoring from checkpoint: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    state = torch.load(path, map_location=device)

    # Check if essential keys exist
    required_keys = ["global_step", "node_model", "edge_model", "optim_node_model",
                     "optim_edge_model", "scheduler_node_model", "scheduler_edge_model"]
    for key in required_keys:
        if key not in state:
            raise KeyError(f"Checkpoint missing required key: '{key}'")

    global_step = state["global_step"]
    try:
        node_model.load_state_dict(state["node_model"])
        edge_model.load_state_dict(state["edge_model"])
        optim_node_model.load_state_dict(state["optim_node_model"])
        optim_edge_model.load_state_dict(state["optim_edge_model"])
        scheduler_node_model.load_state_dict(state["scheduler_node_model"])
        scheduler_edge_model.load_state_dict(state["scheduler_edge_model"])
        print(f"Successfully restored models and optimizers to step {global_step}.")
    except Exception as e:
        raise RuntimeError(f"Error loading state dictionaries from checkpoint: {e}")

    # Optionally restore criteria state if needed and present (less common)
    # if "criterion_edge" in state:
    #     try: criterion_edge.load_state_dict(state["criterion_edge"])
    #     except Exception as e: print(f"Warning: Could not restore criterion_edge state: {e}")
    # if criterion_node and "criterion_node" in state:
    #      try: criterion_node.load_state_dict(state["criterion_node"])
    #      except Exception as e: print(f"Warning: Could not restore criterion_node state: {e}")

    return global_step


# MODIFIED train_loop function
def train_loop(config, node_model, edge_model, step_fn, criterion_edge, criterion_node,
               optim_node_model, optim_edge_model, scheduler_node_model, scheduler_edge_model,
               device,
               dataset, # ACCEPT dataset object as argument
               global_step, writer, base_path):

    data_loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)

    # Determine these flags once from config
    predict_node_types = config['model'].get('predict_node_types', False)
    use_conditioning = config['model'].get('truth_table_conditioning', False)
    use_edge_features = config['model']['GraphRNN'].get('edge_feature_len', NUM_EDGE_FEATURES) > 1

    # Ensure models are in training mode
    node_model.train()
    edge_model.train()

    done = False
    epoch = 0 # Simple epoch counter for potential logging
    start_step = global_step
    start_time = time.time()

    while not done:
        epoch += 1
        #print(f"--- Epoch {epoch} ---")
        epoch_loss_sum = 0
        epoch_steps = 0

        for batch_idx, data in enumerate(data_loader):
            global_step += 1
            if global_step > config['train']['steps']:
                done = True
                break

            # Perform one training step
            loss_dict = step_fn(node_model, edge_model, data,
                                criterion_edge, criterion_node,
                                optim_node_model, optim_edge_model,
                                scheduler_node_model, scheduler_edge_model,
                                device, use_edge_features,
                                predict_node_types, use_conditioning)

            epoch_loss_sum += loss_dict['total']
            epoch_steps += 1

            # Log to TensorBoard
            writer.add_scalar('loss/step_total', loss_dict['total'], global_step)
            writer.add_scalar('loss/step_edge', loss_dict['edge'], global_step)
            if predict_node_types and 'node' in loss_dict:
                writer.add_scalar('loss/step_node', loss_dict['node'], global_step)
            writer.add_scalar('learning_rate/node_model', scheduler_node_model.get_last_lr()[0], global_step)
            writer.add_scalar('learning_rate/edge_model', scheduler_edge_model.get_last_lr()[0], global_step)


            # Print progress periodically
            if global_step % config['train']['print_iter'] == 0:
                avg_loss_since_print = epoch_loss_sum / epoch_steps # Avg loss within this epoch so far, or over last print_iter steps
                time_per_iter = (time.time() - start_time) / (global_step - start_step)
                eta_seconds = (config['train']['steps'] - global_step) * time_per_iter
                eta_formatted = datetime.timedelta(seconds=int(eta_seconds))
                print(f"[{global_step}/{config['train']['steps']}] loss={avg_loss_since_print:.4f} "
                      f"lr={scheduler_node_model.get_last_lr()[0]:.1E} "
                      f"time/iter={time_per_iter:.3f}s eta={eta_formatted}")
                # Reset epoch stats for next print interval if desired, or keep running epoch avg
                # epoch_loss_sum = 0
                # epoch_steps = 0

            # Save checkpoint periodically or at the end
            if global_step % config['train']['checkpoint_iter'] == 0 or global_step >= config['train']['steps']:
                checkpoint_dir = os.path.join(base_path, config['train'].get('checkpoint_dir', 'checkpoints')) # Default dir
                os.makedirs(checkpoint_dir, exist_ok=True) # Ensure directory exists

                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}.pth")
                print(f"Saving checkpoint to {checkpoint_path}...")
                save_state = {
                    "global_step": global_step,
                    "config": config, # Save config used for this training run
                    "node_model": node_model.state_dict(),
                    "edge_model": edge_model.state_dict(),
                    "optim_node_model": optim_node_model.state_dict(),
                    "optim_edge_model": optim_edge_model.state_dict(),
                    "scheduler_node_model": scheduler_node_model.state_dict(),
                    "scheduler_edge_model": scheduler_edge_model.state_dict(),
                    # Optionally save criteria state (less common)
                    # "criterion_edge": criterion_edge.state_dict(),
                    # "criterion_node": criterion_node.state_dict() if criterion_node else None,
                }
                try:
                    torch.save(save_state, checkpoint_path)
                    print("Checkpoint saved.")
                except Exception as e:
                     print(f"Error saving checkpoint: {e}")

        # Log average epoch loss
        if epoch_steps > 0:
             avg_epoch_loss = epoch_loss_sum / epoch_steps
             writer.add_scalar('loss/epoch_avg_total', avg_epoch_loss, epoch)
             #print(f"--- Epoch {epoch} finished. Avg Loss: {avg_epoch_loss:.4f} ---")


    print("Training loop finished.")
    writer.close()


# MODIFIED main function
def main():
    args = parse_args()
    try:
        config = load_config(args.config_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading config: {e}")
        return 1 # Exit if config is invalid

    # --- Setup ---
    base_path = args.save_dir # Use save_dir as the base path for logs/checkpoints
    os.makedirs(base_path, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    #print(f"Using device: {device}")

    # --- Determine max_node_count BEFORE setting up models ---
    #print("Determining max_node_count from dataset...")
    try:
        # Use the new helper function instead of loading full AIGDataset
        max_node_count = get_max_node_count_from_pkl(
            config['data']['graph_file'],
        )
    except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        print(f"FATAL: Failed to determine max_node_count from pickle file: {e}")
        return 1

    max_level = 0  # Default
    try:
        #print("Initializing main dataset instance...")
        dataset = AIGDataset(
            graph_file=config['data']['graph_file'],
            training=True,  # Create the full dataset instance
            # train_split is used internally by dataset for slicing later
            max_graphs=config['data'].get('max_graphs'),
            include_node_types=config['model'].get('predict_node_types', False)
        )
        max_level = dataset.max_level  # Get max_level
        #print(f"Dataset initialized. Determined max_level: {max_level}")
    except Exception as e:
        print(f"FATAL: Failed to initialize main dataset: {e}. Cannot proceed.")
        return 1

    try:
        # MODIFIED: Pass max_level
        node_model, edge_model, step_fn = setup_models(config, device, max_node_count, max_level)
        criterion_edge, criterion_node, use_edge_features = setup_criteria(config, device)
    except (ValueError, KeyError) as e:
        print(f"Error setting up models or criteria: {e}");
        return 1

    # Setup optimizers and schedulers
    try:
        optim_node_model = torch.optim.Adam(node_model.parameters(), lr=config['train']['lr'])
        optim_edge_model = torch.optim.Adam(edge_model.parameters(), lr=config['train']['lr'])
        scheduler_node_model = MultiStepLR(optim_node_model,
                                           milestones=config['train']['lr_schedule_milestones'],
                                           gamma=config['train']['lr_schedule_gamma'])
        scheduler_edge_model = MultiStepLR(optim_edge_model,
                                           milestones=config['train']['lr_schedule_milestones'],
                                           gamma=config['train']['lr_schedule_gamma'])
    except KeyError as e:
        print(f"Error setting up optimizers/schedulers: Missing key {e} in config['train']")
        return 1

    # --- TensorBoard Writer ---
    log_dir = os.path.join(base_path, config['train'].get('log_dir', 'logs')) # Default dir
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # --- Restore Checkpoint if provided ---
    global_step = 0
    if args.restore_path:
        try:
            global_step = restore_checkpoint(args.restore_path, device, node_model, edge_model,
                                             optim_node_model, optim_edge_model,
                                             scheduler_node_model, scheduler_edge_model)
            print(f"Restored checkpoint. Starting from global step: {global_step}")
        except (FileNotFoundError, KeyError, RuntimeError) as e:
             print(f"Error restoring checkpoint: {e}. Starting training from scratch.")
             global_step = 0 # Ensure we start from 0 if restore fails

    # --- Start Training Loop ---
    train_loop(config, node_model, edge_model, step_fn,
               criterion_edge, criterion_node,
               optim_node_model, optim_edge_model,
               scheduler_node_model, scheduler_edge_model,
               device,
               dataset,  # Pass the initialized dataset object
               global_step, writer, base_path)

    print("Main function finished.")
    return 0


if __name__ == "__main__":
    # Example Usage:
    # python main.py --config_file configs/config_aig_topsort.yaml --save_dir runs/aig_topsort_run1
    # python main.py --config_file configs/config_aig_bfs.yaml --save_dir runs/aig_bfs_run1 -r runs/aig_bfs_run1/checkpoints/checkpoint-10000.pth
    exit_code = main()
    exit(exit_code)