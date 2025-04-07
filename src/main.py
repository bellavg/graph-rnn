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
from utils import setup_models, setup_criteria

# Assuming these imports are correct relative to your project structure
# Make sure train_rnn_step/train_mlp_step signatures are updated
# to remove criterion_node, predict_node_types, use_conditioning args
from train import train_rnn_step, train_mlp_step
from model import *
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
    # Removed use_bfs check here as dataset now handles TopSort default
    # if 'use_bfs' not in config['data']:
    #      raise ValueError("Config must specify 'data.use_bfs' (true or false).")
    # if config['data']['use_bfs'] and 'm' not in config['data']:
    #      raise ValueError("Config must specify 'data.m' when 'data.use_bfs' is true.")
    return config




def get_max_node_count_from_pkl(graph_file: str) -> int:
    """
    Efficiently loads raw graphs from a pickle file and finds the maximum node count.
    """
    # (Implementation remains the same)
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

    return max_nodes


def restore_checkpoint(path, device, node_model, edge_model,
                       optim_node_model, optim_edge_model,
                       scheduler_node_model, scheduler_edge_model):
    # (Implementation remains the same)
    print(f"Restoring from checkpoint: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    state = torch.load(path, map_location=device)

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

    return global_step


# MODIFIED train_loop function
def train_loop(config, node_model, edge_model, step_fn,
               criterion_edge, # REMOVED criterion_node
               optim_node_model, optim_edge_model, scheduler_node_model, scheduler_edge_model,
               device,
               dataset,
               global_step, writer, base_path):

    data_loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)

    # Determine flags once from config
    # REMOVED predict_node_types, use_conditioning
    use_edge_features = config['model']['GraphRNN'].get('edge_feature_len', NUM_EDGE_FEATURES) > 1

    node_model.train()
    edge_model.train()

    done = False
    epoch = 0
    start_step = global_step
    start_time = time.time()

    while not done:
        epoch += 1
        epoch_loss_sum = 0
        epoch_steps = 0

        for batch_idx, data in enumerate(data_loader):
            global_step += 1
            if global_step > config['train']['steps']:
                done = True
                break

            # Perform one training step
            # REMOVED criterion_node, predict_node_types, use_conditioning args from step_fn call
            # Ensure step_fn (train_rnn_step/train_mlp_step) signatures are updated accordingly
            loss_dict = step_fn(node_model, edge_model, data,
                                criterion_edge, # Pass only edge criterion
                                optim_node_model, optim_edge_model,
                                scheduler_node_model, scheduler_edge_model,
                                device, use_edge_features)

            # Assuming loss_dict might still contain 'total' and 'edge' keys
            # It might not contain 'node' anymore
            total_loss = loss_dict.get('total', 0.0) # Default to 0 if key missing
            edge_loss = loss_dict.get('edge', 0.0)

            epoch_loss_sum += total_loss
            epoch_steps += 1

            # Log to TensorBoard
            writer.add_scalar('loss/step_total', total_loss, global_step)
            writer.add_scalar('loss/step_edge', edge_loss, global_step)
            # REMOVED node loss logging
            # if predict_node_types and 'node' in loss_dict: ...
            writer.add_scalar('learning_rate/node_model', scheduler_node_model.get_last_lr()[0], global_step)
            writer.add_scalar('learning_rate/edge_model', scheduler_edge_model.get_last_lr()[0], global_step)


            # Print progress periodically
            if global_step % config['train']['print_iter'] == 0:
                avg_loss_since_print = epoch_loss_sum / epoch_steps if epoch_steps > 0 else 0.0
                time_per_iter = (time.time() - start_time) / (global_step - start_step) if (global_step - start_step) > 0 else 0.0
                eta_seconds = (config['train']['steps'] - global_step) * time_per_iter
                eta_formatted = datetime.timedelta(seconds=int(eta_seconds))
                print(f"[{global_step}/{config['train']['steps']}] loss={avg_loss_since_print:.4f} "
                      f"lr={scheduler_node_model.get_last_lr()[0]:.1E} "
                      f"time/iter={time_per_iter:.3f}s eta={eta_formatted}")
                # Reset epoch stats for next print interval if desired
                # epoch_loss_sum = 0
                # epoch_steps = 0

            # Save checkpoint periodically or at the end
            if global_step % config['train']['checkpoint_iter'] == 0 or global_step >= config['train']['steps']:
                checkpoint_dir = os.path.join(base_path, config['train'].get('checkpoint_dir', 'checkpoints'))
                os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}.pth")
                print(f"Saving checkpoint to {checkpoint_path}...")
                save_state = {
                    "global_step": global_step,
                    "config": config,
                    "node_model": node_model.state_dict(),
                    "edge_model": edge_model.state_dict(),
                    "optim_node_model": optim_node_model.state_dict(),
                    "optim_edge_model": optim_edge_model.state_dict(),
                    "scheduler_node_model": scheduler_node_model.state_dict(),
                    "scheduler_edge_model": scheduler_edge_model.state_dict(),
                    # REMOVED criteria saving
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

    print("Training loop finished.")
    writer.close()


# MODIFIED main function
def main():
    args = parse_args()
    try:
        config = load_config(args.config_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading config: {e}")
        return 1

    # --- Setup ---
    base_path = args.save_dir
    os.makedirs(base_path, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # --- Determine max_node_count ---
    try:
        max_node_count = get_max_node_count_from_pkl(
            config['data']['graph_file'],
        )
    except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        print(f"FATAL: Failed to determine max_node_count: {e}")
        return 1

    # --- Initialize Dataset and get max_level ---
    max_level = 0
    try:
        dataset = AIGDataset(
            graph_file=config['data']['graph_file'],
            training=True,
            max_graphs=config['data'].get('max_graphs'),
            # REMOVED include_node_types based on config
            # It will default to False in AIGDataset unless explicitly set true in config
            include_node_types=False # Force False as we removed logic
        )
        max_level = dataset.max_level
        print(f"Dataset initialized. Max node count: {max_node_count}, Max level: {max_level}")
    except Exception as e:
        print(f"FATAL: Failed to initialize main dataset: {e}. Cannot proceed.")
        return 1

    # --- Setup Models and Criteria ---
    try:
        node_model, edge_model, step_fn = setup_models(config, device, max_node_count, max_level)
        # REMOVED criterion_node from return value
        criterion_edge, use_edge_features = setup_criteria(config, device, dataset)
    except (ValueError, KeyError) as e:
        print(f"Error setting up models or criteria: {e}");
        return 1

    # --- Setup Optimizers and Schedulers ---
    try:
        optim_node_model = torch.optim.AdamW(node_model.parameters(), lr=config['train']['lr'])
        optim_edge_model = torch.optim.AdamW(edge_model.parameters(), lr=config['train']['lr'])
        eta_min_value = 0  # Or config['train'].get('eta_min', 0)
        total_steps = config['train']['steps']

        scheduler_node_model = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_node_model,
            T_max=total_steps,
            eta_min=eta_min_value
        )
        scheduler_edge_model = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_edge_model,
            T_max=total_steps,
            eta_min=eta_min_value
        )
    except KeyError as e:
        print(f"Error setting up optimizers/schedulers: Missing key {e} in config['train']")
        return 1

    # --- TensorBoard Writer ---
    log_dir = os.path.join(base_path, config['train'].get('log_dir', 'logs'))
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
             global_step = 0

    # --- Start Training Loop ---
    # REMOVED criterion_node from args
    train_loop(config, node_model, edge_model, step_fn,
               criterion_edge, # Pass only edge criterion
               optim_node_model, optim_edge_model,
               scheduler_node_model, scheduler_edge_model,
               device,
               dataset,
               global_step, writer, base_path)

    print("Main function finished.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)