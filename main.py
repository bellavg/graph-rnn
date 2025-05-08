#!/usr/bin/env python3
"""
Training script for AIG graph generation models.
Includes node type prediction loss.
Uses a simple file logger to track training progress.
Handles loading data from multiple PKL files specified in the config.
"""

import argparse
import os
import time
import datetime
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List
import sys # Added for stderr
from tqdm import tqdm # ADDED: Import tqdm

# Import from our modules
from src.data_utils import load_config
from src.logger import SimpleLogger
from src.utils import setup_models, setup_criteria
from src.aig_dataset import AIGDataset
from src.train import train_rnn_step # Assuming only RNN step needed

# --- Constants ---
DEFAULT_MAX_NODE_COUNT = 64
DEFAULT_MAX_LEVEL = 22

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="src/config_aig_rnn.yaml", # Adjusted default path
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
            # Allow missing keys if node prediction was added later
            # A more robust check might involve comparing model state_dict keys
            if key == "node_model" and hasattr(node_model, 'predict_node_types') and node_model.predict_node_types:
                 print(f"Warning: Checkpoint missing '{key}', likely due to added node prediction head. Initializing node model from scratch.")
                 continue # Skip loading this specific key
            raise KeyError(f"Checkpoint missing required key: '{key}'")

    global_step = state["global_step"]
    try:
        # Use strict=False to handle potentially added node_type_predictor layer
        node_load_result = node_model.load_state_dict(state.get("node_model", {}), strict=False) # Use .get with default
        edge_load_result = edge_model.load_state_dict(state.get("edge_model", {}), strict=False) # Use .get with default

        # Log missing/unexpected keys for debugging
        if node_load_result.missing_keys: print(f"Warning: Node model missing keys in checkpoint: {node_load_result.missing_keys}")
        if node_load_result.unexpected_keys: print(f"Warning: Node model unexpected keys in checkpoint: {node_load_result.unexpected_keys}")
        if edge_load_result.missing_keys: print(f"Warning: Edge model missing keys in checkpoint: {edge_load_result.missing_keys}")
        if edge_load_result.unexpected_keys: print(f"Warning: Edge model unexpected keys in checkpoint: {edge_load_result.unexpected_keys}")

        # Load optimizers and schedulers if present
        if "optim_node_model" in state: optim_node_model.load_state_dict(state["optim_node_model"])
        else: print("Warning: optim_node_model state not found in checkpoint.")
        if "optim_edge_model" in state: optim_edge_model.load_state_dict(state["optim_edge_model"])
        else: print("Warning: optim_edge_model state not found in checkpoint.")
        if "scheduler_node_model" in state: scheduler_node_model.load_state_dict(state["scheduler_node_model"])
        else: print("Warning: scheduler_node_model state not found in checkpoint.")
        if "scheduler_edge_model" in state: scheduler_edge_model.load_state_dict(state["scheduler_edge_model"])
        else: print("Warning: scheduler_edge_model state not found in checkpoint.")

        print(f"Successfully restored components from checkpoint to step {global_step}.")
    except Exception as e:
        print(f"Error loading state dictionaries from checkpoint: {e}")
        print("Attempting to proceed with potentially partially loaded models/optimizers.")
        global_step = 0 # Reset step if loading fails significantly

    return global_step


def train_loop(config, node_model, edge_model, step_fn,
               criterion_edge, criterion_node, # ADDED criterion_node
               optim_node_model, optim_edge_model, scheduler_node_model, scheduler_edge_model,
               device,
               dataset,
               global_step, logger, base_path):
    """
    Main training loop function that handles the training process.
    Includes node loss calculation.
    """
    if len(dataset) == 0:
        print("Error: Training dataset is empty. Cannot start training loop.")
        return

    # Consider adding num_workers and pin_memory for potentially faster loading
    data_loader = DataLoader(
        dataset,
        batch_size=config['train'].get('batch_size', 32), # Use .get with default
        shuffle=True,
        num_workers=config['train'].get('num_workers', 0), # Configurable workers
        pin_memory=True if device.type == 'cuda' else False # Pin memory if using CUDA
    )

    # Determine flags from config safely
    use_edge_features = config.get('model', {}).get('GraphRNN', {}).get('edge_feature_len', 1) > 1
    predict_node_types = config.get('train', {}).get('predict_node_types', config.get('model', {}).get('predict_node_types', False))
    node_loss_weight = config.get('train', {}).get('node_loss_weight', 1.0)

    node_model.train()
    edge_model.train()

    done = False
    epoch = 0
    start_step = global_step
    start_time = time.time()

    while not done:
        epoch += 1
        epoch_loss_sum = 0.0
        epoch_edge_loss_sum = 0.0
        epoch_node_loss_sum = 0.0
        epoch_steps = 0

        # --- FIXED: Use tqdm.tqdm ---
        pbar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
        # --- END FIXED ---
        for batch_idx, data in enumerate(pbar):
            global_step += 1
            if global_step > config['train']['steps']:
                done = True
                break

            # Perform one training step
            try:
                loss_dict = step_fn(
                    node_model, edge_model, data,
                    criterion_edge, criterion_node, # Pass both criteria
                    optim_node_model, optim_edge_model,
                    scheduler_node_model, scheduler_edge_model,
                    device, use_edge_features,
                    node_loss_weight=node_loss_weight # Pass weight
                )
            except Exception as e:
                print(f"\nError during training step {global_step}: {e}")
                # import traceback
                # traceback.print_exc() # Uncomment for detailed traceback
                print("Skipping step.")
                continue # Skip to next step if an error occurs

            # Get loss values
            total_loss = loss_dict.get('total', 0.0)
            edge_loss = loss_dict.get('edge', 0.0)
            node_loss = loss_dict.get('node', 0.0)

            # Accumulate losses for epoch average
            epoch_loss_sum += total_loss
            epoch_edge_loss_sum += edge_loss
            epoch_node_loss_sum += node_loss
            epoch_steps += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss:.4f}",
                'edge_l': f"{edge_loss:.4f}",
                'node_l': f"{node_loss:.4f}" if predict_node_types else "N/A",
                'lr': f"{scheduler_node_model.get_last_lr()[0]:.1E}"
            })

            # Log metrics periodically
            if global_step % config['train']['print_iter'] == 0:
                avg_total_loss_print = epoch_loss_sum / epoch_steps if epoch_steps > 0 else 0.0
                avg_edge_loss_print = epoch_edge_loss_sum / epoch_steps if epoch_steps > 0 else 0.0
                avg_node_loss_print = epoch_node_loss_sum / epoch_steps if epoch_steps > 0 else 0.0

                time_per_iter = (time.time() - start_time) / (global_step - start_step) if (global_step - start_step) > 0 else 0.0
                eta_seconds = (config['train']['steps'] - global_step) * time_per_iter if time_per_iter > 0 else 0
                eta_formatted = datetime.timedelta(seconds=int(eta_seconds))

                # Log to console
                log_str = (f"[{global_step}/{config['train']['steps']}] "
                           f"Avg Loss={avg_total_loss_print:.4f} "
                           f"(E:{avg_edge_loss_print:.4f} "
                           f"N:{avg_node_loss_print:.4f}) " if predict_node_types else f"Avg Loss={avg_total_loss_print:.4f} ")
                log_str += (f"LR={scheduler_node_model.get_last_lr()[0]:.1E} "
                            f"IterTime={time_per_iter:.3f}s ETA={eta_formatted}")
                # Use tqdm's write method to avoid interfering with the progress bar
                pbar.write(log_str)


                # Log to file
                logger.log_step(
                    global_step=global_step, epoch=epoch,
                    total_loss=total_loss, edge_loss=edge_loss, node_loss=node_loss,
                    lr_node_model=scheduler_node_model.get_last_lr()[0],
                    lr_edge_model=scheduler_edge_model.get_last_lr()[0],
                    time_per_iter=time_per_iter,
                    avg_epoch_loss=avg_total_loss_print, # Log average since last print
                    dataset_size=len(dataset)
                )

            # Save checkpoint periodically or at the end
            if global_step % config['train']['checkpoint_iter'] == 0 or global_step >= config['train']['steps']:
                checkpoint_dir = os.path.join(base_path, config['train'].get('checkpoint_dir', 'checkpoints'))
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}.pth")
                pbar.write(f"\nSaving checkpoint to {checkpoint_path}...") # Use pbar.write
                save_state = {
                    "global_step": global_step, "config": config,
                    "node_model": node_model.state_dict(), "edge_model": edge_model.state_dict(),
                    "optim_node_model": optim_node_model.state_dict(), "optim_edge_model": optim_edge_model.state_dict(),
                    "scheduler_node_model": scheduler_node_model.state_dict(), "scheduler_edge_model": scheduler_edge_model.state_dict(),
                }
                try:
                    torch.save(save_state, checkpoint_path)
                    pbar.write("Checkpoint saved.") # Use pbar.write
                except Exception as e:
                    pbar.write(f"Error saving checkpoint: {e}") # Use pbar.write

        # Log epoch summary
        pbar.close() # Close epoch progress bar
        if epoch_steps > 0:
            avg_epoch_loss = epoch_loss_sum / epoch_steps
            avg_epoch_edge_loss = epoch_edge_loss_sum / epoch_steps
            avg_epoch_node_loss = epoch_node_loss_sum / epoch_steps
            logger.log_epoch(
                epoch=epoch, avg_loss=avg_epoch_loss,
                avg_edge_loss=avg_epoch_edge_loss, avg_node_loss=avg_epoch_node_loss,
                steps=epoch_steps, global_step=global_step, dataset_size=len(dataset)
            )
            epoch_summary = (f"Epoch {epoch} complete. Avg Loss: {avg_epoch_loss:.4f} "
                             f"(E:{avg_epoch_edge_loss:.4f} N:{avg_epoch_node_loss:.4f})" if predict_node_types else
                             f"Epoch {epoch} complete. Avg Loss: {avg_epoch_loss:.4f}")
            print(epoch_summary)

    print("Training loop finished.")
    logger.close()


def main():
    args = parse_args()
    try:
        config = load_config(args.config_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading config '{args.config_file}': {e}")
        return 1

    # --- Setup ---
    base_path = args.save_dir
    os.makedirs(base_path, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Get graph files list ---
    try:
        graph_files_list = config['data']['graph_files']
        if not isinstance(graph_files_list, list) or not graph_files_list:
             raise ValueError("Config 'data.graph_files' must be a non-empty list.")
        print(f"Using graph files: {graph_files_list}")
    except KeyError:
        print("FATAL: Config missing 'data.graph_files'.")
        return 1
    except ValueError as e:
        print(f"FATAL: {e}")
        return 1

    # --- Determine max_node_count and max_level from config or defaults ---
    max_node_count = config.get('data', {}).get('max_node_count', DEFAULT_MAX_NODE_COUNT)
    max_level = config.get('data', {}).get('max_level', DEFAULT_MAX_LEVEL)
    print(f"Using Max Node Count: {max_node_count}")
    print(f"Using Max Level: {max_level}")

    # --- Initialize Dataset ---
    try:
        dataset = AIGDataset(
            graph_files=graph_files_list,
            training=True,
            train_split=config.get('data', {}).get('train_split', 0.9),
            max_graphs=config.get('data', {}).get('max_graphs'),
            max_train_graphs=config.get('data', {}).get('max_train_graphs')
        )
        # Update max_node_count and max_level based on dataset's computed values
        if dataset.max_node_count != max_node_count:
            print(f"Warning: Config/Default max_node_count ({max_node_count}) differs from dataset's ({dataset.max_node_count}). Using dataset value.")
            max_node_count = dataset.max_node_count
        if dataset.max_level != max_level:
             print(f"Warning: Config/Default max_level ({max_level}) differs from dataset's ({dataset.max_level}). Using dataset value.")
             max_level = dataset.max_level

        print(f"Dataset initialized with {len(dataset)} training graphs.")
        print(f"Final Stats - Max Nodes: {max_node_count}, Max Level: {max_level}")
    except Exception as e:
        print(f"FATAL: Failed to initialize dataset: {e}. Cannot proceed.", file=sys.stderr)
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        return 1


    # --- Setup Models and Criteria ---
    try:
        node_model, edge_model, step_fn = setup_models(config, device, max_node_count, max_level)
        # Unpack criteria including criterion_node
        criterion_edge, criterion_node, use_edge_features = setup_criteria(config, device, dataset)
        # Check if node criterion was created if prediction enabled
        predict_node_types = config.get('train', {}).get('predict_node_types', False)
        if predict_node_types and criterion_node is None:
             print("FATAL: Node prediction enabled but criterion_node was not created in setup_criteria.")
             return 1

    except (ValueError, KeyError) as e:
        print(f"Error setting up models or criteria: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        return 1
    except Exception as e: # Catch other potential errors during setup
        print(f"Unexpected error during setup: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        return 1


    # --- Setup Optimizers and Schedulers ---
    try:
        lr = config.get('train', {}).get('lr', 0.001)
        optim_node_model = torch.optim.AdamW(node_model.parameters(), lr=lr)
        optim_edge_model = torch.optim.AdamW(edge_model.parameters(), lr=lr)
        eta_min_value = config.get('train', {}).get('eta_min', 0)
        total_steps = config.get('train', {}).get('steps')
        if total_steps is None: raise KeyError("'steps' missing in config['train']")

        scheduler_node_model = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_node_model, T_max=total_steps, eta_min=eta_min_value
        )
        scheduler_edge_model = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_edge_model, T_max=total_steps, eta_min=eta_min_value
        )
    except KeyError as e:
        print(f"Error setting up optimizers/schedulers: Missing key {e} in config['train']")
        return 1
    except Exception as e:
        print(f"Error setting up optimizers/schedulers: {e}")
        return 1

    # --- Initialize Simple Logger ---
    log_file_path = config.get('train', {}).get('log_file', os.path.join(base_path, 'training_log.csv'))
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    try:
        logger = SimpleLogger(log_file_path, config)
        print(f"Logger initialized. Writing to: {log_file_path}")
    except Exception as e:
        print(f"Error initializing logger: {e}")
        return 1


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
    try:
        train_loop(config, node_model, edge_model, step_fn,
                   criterion_edge, criterion_node, # Pass both criteria
                   optim_node_model, optim_edge_model,
                   scheduler_node_model, scheduler_edge_model,
                   device,
                   dataset,
                   global_step, logger, base_path)
    except Exception as e:
        print(f"\nFATAL ERROR during training loop: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        logger.close() # Attempt to close logger cleanly
        return 1


    print("Training completed successfully.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
