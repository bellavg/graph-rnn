#!/usr/bin/env python3
"""
Training script for AIG graph generation models.
Uses a simple file logger to track training progress.
"""

import argparse
import os
import time
import datetime
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

# Import from our modules
from data_utils import load_config, get_max_node_count_from_pkl
from logger import SimpleLogger
from utils import setup_models, setup_criteria
from aig_dataset import AIGDataset, NUM_EDGE_FEATURES
from data_utils import get_max_level_from_pkl
from train import train_rnn_step, train_mlp_step


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="configs/config_aig_base.yaml",
                        help='Path of the config file to use for training')
    parser.add_argument('-r', '--restore', dest='restore_path', default=None,
                        help='Checkpoint to continue training from')
    parser.add_argument('--gpu', dest='gpu_id', default=0, type=int,
                        help='Id of the GPU to use')
    parser.add_argument('--save_dir', dest='save_dir', default="./runs", type=str,
                        help='Directory to save checkpoints')
    return parser.parse_args()


def restore_checkpoint(path, device, node_model, edge_model,
                       optim_node_model, optim_edge_model,
                       scheduler_node_model, scheduler_edge_model):
    """
    Restore models and optimizers from a checkpoint.
    """
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


def train_loop(config, node_model, edge_model, step_fn,
               criterion_edge,
               optim_node_model, optim_edge_model, scheduler_node_model, scheduler_edge_model,
               device,
               dataset,
               global_step, logger, base_path):
    """
    Main training loop function that handles the training process.
    """
    data_loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)

    # Always use edge features
    use_edge_features = True

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
            loss_dict = step_fn(node_model, edge_model, data,
                                criterion_edge,
                                optim_node_model, optim_edge_model,
                                scheduler_node_model, scheduler_edge_model,
                                device, use_edge_features)

            # Get loss values
            total_loss = loss_dict.get('total', 0.0)
            edge_loss = loss_dict.get('edge', 0.0)

            epoch_loss_sum += total_loss
            epoch_steps += 1

            # Log metrics periodically
            if global_step % config['train']['print_iter'] == 0:
                avg_loss_since_print = epoch_loss_sum / epoch_steps if epoch_steps > 0 else 0.0
                time_per_iter = (time.time() - start_time) / (global_step - start_step) if (
                        global_step - start_step) > 0 else 0.0
                eta_seconds = (config['train']['steps'] - global_step) * time_per_iter
                eta_formatted = datetime.timedelta(seconds=int(eta_seconds))

                # Log to console
                print(f"[{global_step}/{config['train']['steps']}] loss={avg_loss_since_print:.4f} "
                      f"lr={scheduler_node_model.get_last_lr()[0]:.1E} "
                      f"time/iter={time_per_iter:.3f}s eta={eta_formatted}")

                # Log to file
                logger.log_step(
                    global_step=global_step,
                    epoch=epoch,
                    total_loss=total_loss,
                    edge_loss=edge_loss,
                    lr_node_model=scheduler_node_model.get_last_lr()[0],
                    lr_edge_model=scheduler_edge_model.get_last_lr()[0],
                    time_per_iter=time_per_iter,
                    avg_epoch_loss=avg_loss_since_print,
                    dataset_size=len(dataset)
                )

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
                }
                try:
                    torch.save(save_state, checkpoint_path)
                    print("Checkpoint saved.")
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")

        # Log epoch summary
        if epoch_steps > 0:
            avg_epoch_loss = epoch_loss_sum / epoch_steps
            logger.log_epoch(
                epoch=epoch,
                avg_loss=avg_epoch_loss,
                steps=epoch_steps,
                global_step=global_step,
                dataset_size=len(dataset)
            )
            print(f"Epoch {epoch} complete. Average loss: {avg_epoch_loss:.4f}")

    print("Training loop finished.")
    logger.close()


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

    # --- Determine max_node_count and max_level efficiently ---
    try:
        print("Computing maximum node count from dataset...")
        max_node_count = get_max_node_count_from_pkl(config['data']['graph_file'])
        print(f"Maximum node count: {max_node_count}")

        print("Computing maximum level from dataset...")
        max_level = get_max_level_from_pkl(config['data']['graph_file'])
        print(f"Maximum level: {max_level}")
    except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        print(f"FATAL: Failed to determine dataset statistics: {e}")
        return 1

    # --- Initialize Dataset ---
    try:
        dataset = AIGDataset(
            graph_file=config['data']['graph_file'],
            training=True,
            train_split=config['data'].get('train_split', 0.9),
            max_graphs=config['data'].get('max_graphs'),
            max_train_graphs=config['data'].get('max_train_graphs'),
            include_node_types=False  # Force False as we removed node type logic
        )

        # Sanity check - ensure our precomputed values match the dataset's computed values
        if dataset.max_node_count != max_node_count:
            print(
                f"Warning: Precomputed max_node_count {max_node_count} differs from dataset's {dataset.max_node_count}")
            max_node_count = dataset.max_node_count

        if dataset.max_level != max_level:
            print(f"Warning: Precomputed max_level {max_level} differs from dataset's {dataset.max_level}")
            max_level = dataset.max_level

        print(f"Dataset initialized with {len(dataset)} training graphs")
        print(f"Final statistics - Max node count: {max_node_count}, Max level: {max_level}")
    except Exception as e:
        print(f"FATAL: Failed to initialize main dataset: {e}. Cannot proceed.")
        return 1

    # --- Setup Models and Criteria ---
    try:
        node_model, edge_model, step_fn = setup_models(config, device, max_node_count, max_level)
        criterion_edge, use_edge_features = setup_criteria(config, device, dataset)
    except (ValueError, KeyError) as e:
        print(f"Error setting up models or criteria: {e}")
        return 1

    # --- Setup Optimizers and Schedulers ---
    try:
        optim_node_model = torch.optim.AdamW(node_model.parameters(), lr=config['train']['lr'])
        optim_edge_model = torch.optim.AdamW(edge_model.parameters(), lr=config['train']['lr'])
        eta_min_value = config['train'].get('eta_min', 0)
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

    # --- Initialize Simple Logger ---
    from logger import SimpleLogger

    # Get log file path from config or use default
    log_file_path = config['train'].get('log_file', os.path.join(base_path, 'training_log.csv'))

    # Make sure parent directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Create the logger
    logger = SimpleLogger(log_file_path, config)
    print(f"Logger initialized. Writing to: {log_file_path}")

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
    train_loop(config, node_model, edge_model, step_fn,
               criterion_edge,
               optim_node_model, optim_edge_model,
               scheduler_node_model, scheduler_edge_model,
               device,
               dataset,
               global_step, logger, base_path)

    print("Training completed successfully.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)