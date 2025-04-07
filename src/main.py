# src/main.py - REVERTED TO ORIGINAL STATE

import argparse
import yaml
import torch
import os
import time
import datetime
from torch.utils.data import DataLoader
# Import relevant schedulers (CosineAnnealingLR is typically used)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import pickle
import traceback # For detailed error printing

# Assuming these imports are correct relative to your project structure
from train import train_rnn_step, train_mlp_step # Add train_lstm_step if created/needed
from model import * # Import all model classes
from aig_dataset import AIGDataset, NUM_EDGE_FEATURES # Import NUM_EDGE_FEATURES
from utils import * # Assuming utils contains load_config, setup_models, setup_criteria, get_max_node_count_from_pkl, restore_checkpoint

# --- Reverted parse_args ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="configs/config_aig_base.yaml",
                        help='Path of the config file to use for training')
    parser.add_argument('-r', '--restore', dest='restore_path', default=None,
                        help='Checkpoint to continue training from')
    parser.add_argument('--gpu', dest='gpu_id', default=0, type=int,
                        help='Id of the GPU to use')
    parser.add_argument('--save_dir', dest='save_dir', default="./runs", type=str)
    # --- Fine-tuning arguments removed ---
    return parser.parse_args()


# --- Reverted train_loop ---
# Removed use_edge_features and target_total_steps parameters
# Assumes use_edge_features is handled within setup_criteria and passed correctly from main
def train_loop(config, node_model, edge_model, step_fn,
               criterion_edge, use_edge_features, # Added use_edge_features param back
               optim_node_model, optim_edge_model, scheduler_node_model, scheduler_edge_model,
               device,
               dataset,
               global_step, writer, base_path): # Removed target_total_steps

    data_loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4, pin_memory=True) # Added workers/pin_memory

    node_model.train()
    edge_model.train()

    done = False
    epoch = 0
    start_step = global_step # Start from restored step
    start_time = time.time()
    # Use the max steps defined in the config
    max_steps = config['train']['steps']

    # Calculate approximate starting epoch for logging continuity
    approx_steps_per_epoch = (len(dataset) + config['train']['batch_size'] - 1) // config['train']['batch_size']
    start_epoch = (start_step // approx_steps_per_epoch) if approx_steps_per_epoch > 0 else 0

    print(f"Starting training loop from step {start_step} up to {max_steps} (approx epoch {start_epoch})...")

    # Use start_epoch for loop, but check global_step against max_steps
    epoch = start_epoch
    while global_step < max_steps:
        epoch += 1
        epoch_loss_sum = 0.0 # Use float
        epoch_steps = 0

        batch_iter_start_time = time.time() # Time batches within epoch

        for batch_idx, data in enumerate(data_loader):
            if global_step >= max_steps:
                done = True
                break
            global_step += 1

            # --- Perform one training step ---
            try:
                # Zero gradients before the step
                optim_node_model.zero_grad()
                optim_edge_model.zero_grad()

                loss_dict = step_fn(node_model, edge_model, data,
                                    criterion_edge,
                                    optim_node_model, optim_edge_model,
                                    # Pass None for schedulers if not using scheduler.step() per iteration (but Cosine usually does)
                                    None, None,
                                    device, use_edge_features) # Pass use_edge_features

                total_loss_tensor = loss_dict.get('total')

                # Check for valid loss tensor before backward()
                if total_loss_tensor is None or not isinstance(total_loss_tensor, torch.Tensor) or torch.isnan(total_loss_tensor) or torch.isinf(total_loss_tensor):
                    print(f"WARNING: Invalid or missing 'total' loss tensor ({total_loss_tensor}) at step {global_step}. Skipping backward/step.")
                    continue

                total_loss = total_loss_tensor.item() # Get float value for logging
                edge_loss = loss_dict.get('edge', 0.0) # Allow edge loss to be missing or float

                # Backward pass and optimizer step
                total_loss_tensor.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(node_model.parameters(), 1.0)
                # torch.nn.utils.clip_grad_norm_(edge_model.parameters(), 1.0)
                optim_node_model.step()
                optim_edge_model.step()

                # Scheduler step (CosineAnnealingLR typically steps per iteration)
                if scheduler_node_model: scheduler_node_model.step()
                if scheduler_edge_model: scheduler_edge_model.step()

            except Exception as step_e:
                print(f"\nERROR during training step {global_step}: {step_e}")
                traceback.print_exc()
                print("Attempting to continue training...")
                continue # Skip rest of loop for this batch

            epoch_loss_sum += total_loss
            epoch_steps += 1

            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('loss/step_total', total_loss, global_step)
                # Only log edge loss if present and valid
                if isinstance(edge_loss, (float, int)):
                    writer.add_scalar('loss/step_edge', edge_loss, global_step)
                # Log learning rates
                writer.add_scalar('learning_rate/node_model', optim_node_model.param_groups[0]['lr'], global_step)
                writer.add_scalar('learning_rate/edge_model', optim_edge_model.param_groups[0]['lr'], global_step)

            # Print progress periodically
            print_iter = config['train'].get('print_iter', 100)
            if global_step % print_iter == 0:
                avg_loss_since_print = epoch_loss_sum / epoch_steps if epoch_steps > 0 else 0.0
                current_lr = optim_node_model.param_groups[0]['lr'] # Get current LR from optimizer
                elapsed_steps_total = global_step - start_step
                time_now = time.time()
                time_per_iter = (time_now - start_time) / elapsed_steps_total if elapsed_steps_total > 0 else 0.0
                eta_seconds = (max_steps - global_step) * time_per_iter if time_per_iter > 0 else 0
                eta_formatted = datetime.timedelta(seconds=int(eta_seconds))
                print(f"[{global_step}/{max_steps}] loss={avg_loss_since_print:.4f} "
                      f"lr={current_lr:.1E} "
                      f"time/iter={time_per_iter:.3f}s eta={eta_formatted}")

            # Save checkpoint periodically or at the very end
            save_iter = config['train'].get('checkpoint_save_iter', config['train'].get('checkpoint_iter', 1000))
            if global_step % save_iter == 0 or global_step >= max_steps:
                checkpoint_dir = os.path.join(base_path, config['train'].get('checkpoint_dir', 'checkpoints'))
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}.pth")
                print(f"Saving checkpoint to {checkpoint_path}...")
                sched_node_state = scheduler_node_model.state_dict() if scheduler_node_model else None
                sched_edge_state = scheduler_edge_model.state_dict() if scheduler_edge_model else None
                save_state = {
                    "global_step": global_step, "config": config,
                    "node_model": node_model.state_dict(), "edge_model": edge_model.state_dict(),
                    "optim_node_model": optim_node_model.state_dict(), "optim_edge_model": optim_edge_model.state_dict(),
                    "scheduler_node_model": sched_node_state, "scheduler_edge_model": sched_edge_state,
                }
                try: torch.save(save_state, checkpoint_path); print("Checkpoint saved.")
                except Exception as e: print(f"Error saving checkpoint: {e}")

        # End of epoch
        if epoch_steps > 0 and writer is not None:
            avg_epoch_loss = epoch_loss_sum / epoch_steps
            writer.add_scalar('loss/epoch_avg_total', avg_epoch_loss, epoch)
        print(f"Epoch {epoch} finished. Avg Loss: {avg_epoch_loss:.4f}")


    print("Training loop finished.")
    if writer is not None:
        writer.close()


# --- Reverted main function ---
def main():
    args = parse_args()
    try:
        config = load_config(args.config_file)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"Error loading config '{args.config_file}': {e}")
        return 1

    # --- Fine-tuning checks removed ---

    # --- Setup ---
    base_path = args.save_dir
    # Create a unique run directory using config name and timestamp
    config_name = os.path.splitext(os.path.basename(args.config_file))[0]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Run name simplified
    run_path = os.path.join(base_path, f"{config_name}_{timestamp}")
    os.makedirs(run_path, exist_ok=True) # Use unique run path as base
    print(f"Run directory: {run_path}")

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Determine max_node_count ---
    try:
        graph_file_path = config['data']['graph_file']
        print(f"Determining max node count from: {graph_file_path}")
        if not os.path.exists(graph_file_path): raise FileNotFoundError(f"Dataset file not found: {graph_file_path}")
        max_node_count = get_max_node_count_from_pkl(graph_file_path)
        print(f"Max node count found: {max_node_count}")
    except (KeyError, FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        print(f"FATAL: Failed to determine max_node_count: {e}")
        traceback.print_exc()
        return 1

    # --- Initialize Dataset ---
    max_level = 0
    dataset = None
    try:
        print("Initializing dataset...")
        dataset = AIGDataset(
            graph_file=graph_file_path,
            training=True,
            max_graphs=config['data'].get('max_graphs'),
            include_node_types=False
        )
        max_level = dataset.max_level
        # Use the max_node_count determined *by the dataset class* after processing
        max_node_count = dataset.max_node_count
        print(f"Dataset initialized. Max node count: {max_node_count}, Max level: {max_level}")

    except Exception as e:
        print(f"FATAL: Failed to initialize main dataset: {e}")
        traceback.print_exc()
        return 1

    # --- Setup Models and Criteria ---
    try:
        print("Setting up models...")
        node_model, edge_model, step_fn = setup_models(config, device, max_node_count, max_level)

        print("Setting up criteria...")
        # Pass dataset to setup_criteria
        criterion_edge, use_edge_features_flag = setup_criteria(config, device, dataset)

    except (ValueError, KeyError, RuntimeError) as e:
        print(f"Error during model/criteria setup: {e}")
        traceback.print_exc()
        return 1

    # --- Setup Optimizers ---
    try:
        # Use LR directly from config
        initial_lr = config['train']['lr']
        print(f"Setting up optimizers with initial LR: {initial_lr:.1E}")
        # Add weight decay option from config if present
        weight_decay = config['train'].get('weight_decay', 0.0)
        optim_node_model = torch.optim.AdamW(node_model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        optim_edge_model = torch.optim.AdamW(edge_model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    except KeyError as e:
        print(f"Error setting up optimizers: Missing key {e} in config['train']")
        return 1

    # --- Setup Schedulers & Restore Checkpoint ---
    scheduler_node_model = None
    scheduler_edge_model = None
    global_step = 0
    # Total steps always come from the config now
    total_steps = config['train']['steps']

    # Initialize schedulers based on config FIRST (needed before restore)
    try:
        # Assuming CosineAnnealingLR as the default/common scheduler here
        # Adjust T_max to the *total* steps defined in the config
        eta_min_value = config['train'].get('lr_schedule_eta_min', 0)
        scheduler_node_model = CosineAnnealingLR(optim_node_model, T_max=total_steps, eta_min=eta_min_value)
        scheduler_edge_model = CosineAnnealingLR(optim_edge_model, T_max=total_steps, eta_min=eta_min_value)
        print(f"Initialized CosineAnnealingLR schedulers with T_max={total_steps}, eta_min={eta_min_value}")
    except KeyError as e:
        print(f"Error setting up schedulers: Missing key {e} in config['train']")
        return 1

    if args.restore_path:
        try:
            # Pass the initialized schedulers to restore_checkpoint
            # restore_checkpoint should now always try to load scheduler state
            global_step = restore_checkpoint(
                args.restore_path, device, node_model, edge_model,
                optim_node_model, optim_edge_model,
                scheduler_node_model, scheduler_edge_model # Pass initialized schedulers
            )
            print(f"Restored checkpoint. Resuming from global step: {global_step}")

        except (FileNotFoundError, KeyError, RuntimeError) as e:
            print(f"Warning: Error restoring checkpoint '{args.restore_path}': {e}. Training from scratch.")
            traceback.print_exc()
            global_step = 0
            # Schedulers are already initialized for training from scratch above

    else:
        # No restore path, training from scratch. Schedulers are already initialized.
        print("No checkpoint specified. Starting training from scratch.")

    # --- TensorBoard Writer ---
    log_dir_name = config['train'].get('log_dir', 'logs')
    # Use the unique run_path created earlier
    log_dir = os.path.join(run_path, log_dir_name) # Place logs inside unique run path
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    try:
        with open(os.path.join(run_path, 'config.yaml'), 'w') as f_cfg: # Save config in run path
            yaml.dump(config, f_cfg, sort_keys=False)
        with open(os.path.join(run_path, 'args.txt'), 'w') as f_args: # Save args in run path
            print(vars(args), file=f_args)
    except Exception as e:
        print(f"Warning: Could not save config/args to run dir: {e}")

    # --- Start Training Loop ---
    print(f"Target total steps for this run: {total_steps}")
    try:
        # Pass the unique run_path as base_path for saving checkpoints inside it
        train_loop(config, node_model, edge_model, step_fn,
                   criterion_edge, use_edge_features_flag, # Pass flag
                   optim_node_model, optim_edge_model,
                   scheduler_node_model, scheduler_edge_model,
                   device,
                   dataset,
                   global_step, writer, run_path) # Pass run_path here
    except Exception as e:
        print(f"FATAL: Error occurred during training loop: {e}")
        traceback.print_exc()
        if writer: writer.close()
        return 1

    print("Main function finished successfully.")
    return 0


if __name__ == "__main__":
    # Assuming the utility functions are defined in utils.py or imported above
    # You'll need load_config, setup_models, setup_criteria, get_max_node_count_from_pkl, restore_checkpoint
    exit_code = main()
    exit(exit_code)