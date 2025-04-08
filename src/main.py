#!/usr/bin/env python3
"""
Training script for AIG graph generation models.
Supports optional independent training of a node type predictor.
"""

import argparse
import datetime
import os
import time

import torch
import tqdm
from torch.utils.data import DataLoader

from aig_dataset import AIGDataset, NODE_TYPES  # Import NODE_TYPES
# Import from our modules
from data_utils import load_config, get_max_node_count_from_pkl, get_max_level_from_pkl
from logger import SimpleLogger
from utils import setup_models, setup_criteria, load_weights_for_finetuning

# --- Constants ---
# Ensure NODE_TYPES is correctly imported or defined
try:
    NUM_NODE_TYPES = len(NODE_TYPES) # Determine number of types from aig_dataset
except NameError:
    print("Warning: NODE_TYPES not found from aig_dataset import. Defaulting to 4.")
    NUM_NODE_TYPES = 4 # Fallback: PI, PO, AND, ZERO

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

    parser.add_argument('--node_type', default=False, action='store_true',
                        help='Enable training an independent node type prediction model.')
    parser.add_argument('--finetune_from', default=None, type=str,
                        help='Checkpoint to start FINE-TUNING from (loads model weights only)')
    # --- NEW Fine-tuning LR Args ---
    parser.add_argument('--lr_finetune_base', type=float, default=1e-5,
                        help='Learning rate for base models (node/edge) during fine-tuning.')
    parser.add_argument('--lr_finetune_add', type=float, default=1e-4,
                        help='Learning rate for whatever is being added during fine-tuning.')

    return parser.parse_args()


# --- restore_checkpoint remains the same as the previous version ---
def restore_checkpoint(path, device, node_model, edge_model,
                       optim_node_model, optim_edge_model,
                       scheduler_node_model, scheduler_edge_model,
                       # Optional args for node predictor
                       node_type_predictor=None,
                       optim_node_type=None,
                       scheduler_node_type=None
                       ):
    """
    Restore models and optimizers from a checkpoint. Handles optional node predictor.
    """
    print(f"Restoring from checkpoint: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    state = torch.load(path, map_location=device)

    # Check required keys for base models
    required_keys = ["global_step", "node_model", "edge_model", "optim_node_model",
                     "optim_edge_model", "scheduler_node_model", "scheduler_edge_model"]
    for key in required_keys:
        if key not in state:
            # Check for older format keys if needed (e.g., node_net -> node_model)
            alt_key_map = {'node_model': 'node_net', 'edge_model': 'edge_net'}
            alt_key = alt_key_map.get(key)
            if alt_key and alt_key in state:
                print(f"Checkpoint using older key '{alt_key}' for '{key}'.")
                state[key] = state[alt_key] # Map old key to new key
            else:
                raise KeyError(f"Checkpoint missing required key: '{key}'")


    global_step = state["global_step"]
    try:
        node_model.load_state_dict(state["node_model"])
        edge_model.load_state_dict(state["edge_model"])
        optim_node_model.load_state_dict(state["optim_node_model"])
        optim_edge_model.load_state_dict(state["optim_edge_model"])
        scheduler_node_model.load_state_dict(state["scheduler_node_model"])
        scheduler_edge_model.load_state_dict(state["scheduler_edge_model"])
        print(f"Successfully restored base models and optimizers to step {global_step}.")

        # Restore node predictor state if it exists in checkpoint AND we are using it
        if node_type_predictor is not None and "node_type_predictor" in state:
            # Only restore if optim/scheduler are provided (meaning flag was likely on)
            if optim_node_type is not None and scheduler_node_type is not None:
                 try:
                     node_type_predictor.load_state_dict(state["node_type_predictor"])
                     # Check if optimizer/scheduler states exist before loading
                     if "optim_node_type" in state:
                         optim_node_type.load_state_dict(state["optim_node_type"])
                     else:
                         print("Warning: Checkpoint missing 'optim_node_type' state.")
                     if "scheduler_node_type" in state:
                        scheduler_node_type.load_state_dict(state["scheduler_node_type"])
                     else:
                         print("Warning: Checkpoint missing 'scheduler_node_type' state.")
                     print("Successfully restored node type predictor state (and optim/scheduler if found).")
                 except KeyError as ke:
                     print(f"Warning: Checkpoint missing some node predictor state ({ke}). Node predictor not fully restored.")
                 except Exception as e_np:
                     print(f"Error restoring node predictor state: {e_np}. Node predictor not fully restored.")
            else:
                 print("Warning: Node predictor state found in checkpoint, but predictor is not active in current run or optim/scheduler missing.")
        elif node_type_predictor is not None:
             print("Warning: Training node predictor, but no state found in checkpoint.")

    except Exception as e:
        raise RuntimeError(f"Error loading state dictionaries from checkpoint: {e}")

    return global_step



# --- train_loop remains the same, just uses the renamed flag ---
def train_loop(config, node_model, edge_model, step_fn,
               criterion_edge,
               optim_node_model, optim_edge_model, scheduler_node_model, scheduler_edge_model,
               device,
               dataset,
               global_step, logger, base_path,
               # --- Args for node predictor ---
               node_type_flag: bool, # Use the flag value passed from main
               node_type_predictor=None,
               criterion_node=None,
               optim_node_type=None,
               scheduler_node_type=None
               ):
    """
    Main training loop function. Handles optional independent node type training.
    """
    # Use num_workers > 0 and pin_memory=True for potentially faster data loading
    data_loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True if 4 > 0 else False)

    # Always use edge features for AIGs
    use_edge_features = True

    node_model.train()
    edge_model.train()
    # Set node predictor to train mode if it exists and flag is set
    if node_type_flag and node_type_predictor:
        node_type_predictor.train()

    done = False
    epoch = 0
    start_step = global_step
    start_time = time.time()


    while not done:
        epoch += 1
        epoch_total_loss_sum = 0
        epoch_edge_loss_sum = 0
        epoch_node_loss_sum = 0 # Track node loss separately
        epoch_steps = 0

        # Use tqdm for progress bar
        pbar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False, unit="batch")
        for batch_idx, data in enumerate(pbar):
            global_step += 1
            if global_step > config['train']['steps']:
                done = True
                break


            loss_dict = step_fn(
                node_model=node_model,
                edge_model=edge_model,
                data=data,
                criterion_edge=criterion_edge,
                optim_node_model=optim_node_model,
                optim_edge_model=optim_edge_model,
                scheduler_node_model=scheduler_node_model,
                scheduler_edge_model=scheduler_edge_model,
                device=device,
                use_edge_features=use_edge_features,
                # --- New args passed to step_fn ---
                train_node_type_flag=node_type_flag, # Pass the flag value
                node_type_predictor=node_type_predictor,
                criterion_node=criterion_node,
                optim_node_type=optim_node_type,
                scheduler_node_type=scheduler_node_type
            )

            # Get loss values
            total_loss = loss_dict.get('total', 0.0)
            edge_loss = loss_dict.get('edge', 0.0)
            node_loss = loss_dict.get('node', 0.0) # Get node loss

            # Check for NaN/Inf losses immediately
            if torch.isnan(torch.tensor(total_loss)) or torch.isinf(torch.tensor(total_loss)):
                 print(f"\nWarning: Invalid total loss detected (NaN/Inf) at step {global_step}. Skipping step.")
                 # Optional: Reset optimizers?
                 # optim_node_model.zero_grad()
                 # optim_edge_model.zero_grad()
                 # if node_type_flag and optim_node_type: optim_node_type.zero_grad()
                 continue # Skip optimizer steps for this batch

            epoch_total_loss_sum += total_loss
            epoch_edge_loss_sum += edge_loss
            epoch_node_loss_sum += node_loss
            epoch_steps += 1

            # Update progress bar description
            postfix_dict = {
                'loss': f"{total_loss:.4f}",
                'edge_l': f"{edge_loss:.4f}"
            }
            if node_type_flag:
                postfix_dict['node_l'] = f"{node_loss:.4f}"
            postfix_dict['lr'] = f"{scheduler_node_model.get_last_lr()[0]:.1E}"
            pbar.set_postfix(postfix_dict)


            # Log metrics periodically
            if global_step % config['train']['print_iter'] == 0:
                avg_total_loss_epoch = epoch_total_loss_sum / epoch_steps if epoch_steps > 0 else 0.0
                avg_edge_loss_epoch = epoch_edge_loss_sum / epoch_steps if epoch_steps > 0 else 0.0
                avg_node_loss_epoch = epoch_node_loss_sum / epoch_steps if epoch_steps > 0 else 0.0

                time_per_iter = (time.time() - start_time) / (global_step - start_step) if (
                        global_step - start_step) > 0 else 0.0
                eta_seconds = (config['train']['steps'] - global_step) * time_per_iter
                eta_formatted = datetime.timedelta(seconds=int(eta_seconds))

                # Prepare log string
                log_str = (f"[{global_step}/{config['train']['steps']}] "
                           f"Loss (Avg E): {avg_total_loss_epoch:.4f} | "
                           f"Edge (Avg E): {avg_edge_loss_epoch:.4f} | ")
                lr_node_type_val = "N/A"
                if node_type_flag:
                    log_str += f"Node (Avg E): {avg_node_loss_epoch:.4f} | "
                    # Safely get LR
                    try:
                        lr_node_type_val = f"{scheduler_node_type.get_last_lr()[0]:.1E}" if scheduler_node_type else f"{optim_node_type.param_groups[0]['lr']:.1E}"
                    except Exception: lr_node_type_val = "Err" # Handle potential errors
                    log_str += f"LR (N/E/T): {scheduler_node_model.get_last_lr()[0]:.1E}/{scheduler_edge_model.get_last_lr()[0]:.1E}/{lr_node_type_val} | "
                else:
                    log_str += f"LR (N/E): {scheduler_node_model.get_last_lr()[0]:.1E}/{scheduler_edge_model.get_last_lr()[0]:.1E} | "
                log_str += f"T/Iter: {time_per_iter:.3f}s | ETA: {eta_formatted}"
                # Use tqdm.write to avoid interfering with the progress bar
                pbar.write(log_str)


                # Log to file using logger (assuming log_step is updated/flexible)
                logger.log_step(
                    global_step=global_step, epoch=epoch,
                    total_loss=total_loss, # Log current step loss
                    edge_loss=edge_loss,
                    node_loss=node_loss if node_type_flag else None, # Log node loss if active
                    lr_node_model=scheduler_node_model.get_last_lr()[0],
                    lr_edge_model=scheduler_edge_model.get_last_lr()[0],
                    lr_node_type=lr_node_type_val if node_type_flag and lr_node_type_val not in ["N/A", "Err"] else None,
                    time_per_iter=time_per_iter,
                    avg_epoch_loss=avg_total_loss_epoch, # Log avg total epoch loss so far
                    dataset_size=len(dataset)
                    # You might want to add avg_epoch_edge_loss and avg_epoch_node_loss to logger too
                )

            # Save checkpoint periodically or at the end
            if global_step % config['train']['checkpoint_iter'] == 0 or global_step >= config['train']['steps']:
                checkpoint_dir = os.path.join(base_path, config['train'].get('checkpoint_dir', 'checkpoints'))
                os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}.pth")
                pbar.write(f"\nSaving checkpoint to {checkpoint_path}...")
                save_state = {
                    "global_step": global_step,
                    "config": config, # Save config used for this run
                    "node_model": node_model.state_dict(),
                    "edge_model": edge_model.state_dict(),
                    "optim_node_model": optim_node_model.state_dict(),
                    "optim_edge_model": optim_edge_model.state_dict(),
                    "scheduler_node_model": scheduler_node_model.state_dict(),
                    "scheduler_edge_model": scheduler_edge_model.state_dict(),
                }
                # Add node predictor state if training
                if node_type_flag and node_type_predictor and optim_node_type and scheduler_node_type:
                    save_state["node_type_predictor"] = node_type_predictor.state_dict()
                    save_state["optim_node_type"] = optim_node_type.state_dict()
                    save_state["scheduler_node_type"] = scheduler_node_type.state_dict()

                try:
                    torch.save(save_state, checkpoint_path)
                    pbar.write("Checkpoint saved.")
                except Exception as e:
                    pbar.write(f"Error saving checkpoint: {e}")

        # End of epoch
        pbar.close() # Close the tqdm progress bar for the epoch
        if epoch_steps > 0:
            avg_epoch_total_loss = epoch_total_loss_sum / epoch_steps
            avg_epoch_edge_loss = epoch_edge_loss_sum / epoch_steps
            avg_epoch_node_loss = epoch_node_loss_sum / epoch_steps if node_type_flag else 0.0

            # Log epoch summary (ensure logger supports node_loss)
            logger.log_epoch(
                epoch=epoch,
                avg_loss=avg_epoch_total_loss, # Log average total loss for epoch
                # Add avg_edge_loss, avg_node_loss if logger supports them
                steps=epoch_steps,
                global_step=global_step,
                dataset_size=len(dataset)
            )
            epoch_summary_str = (f"Epoch {epoch} complete. Avg Total Loss: {avg_epoch_total_loss:.4f} | "
                                 f"Avg Edge Loss: {avg_epoch_edge_loss:.4f}")
            if node_type_flag:
                epoch_summary_str += f" | Avg Node Loss: {avg_epoch_node_loss:.4f}"
            print(epoch_summary_str) # Print final epoch summary

    print("\nTraining loop finished.")
    logger.close()


def main():
    args = parse_args()
    # --- Check incompatible flags ---
    if args.restore_path and args.finetune_from:
        print("Error: Cannot use both --restore and --finetune_from. Choose one.")
        return 1

    try: config = load_config(args.config_file)
    except Exception as e: print(f"Error loading config: {e}"); return 1

    # --- Setup ---
    base_path = args.save_dir; os.makedirs(base_path, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")

    # --- Determine dataset stats ---
    try:
        print("Computing dataset statistics..."); max_node_count = get_max_node_count_from_pkl(config['data']['graph_file']); max_level = get_max_level_from_pkl(config['data']['graph_file'])
        print(f"Max Nodes: {max_node_count}, Max Level: {max_level}")
    except Exception as e: print(f"FATAL: Failed to get dataset stats: {e}"); return 1
    config['data']['max_node_count_train'] = max_node_count; config['data']['max_level_train'] = max_level

    # --- Initialize Dataset ---
    try:
        print(f"Initializing dataset...")
        dataset = AIGDataset(graph_file=config['data']['graph_file'], training=True,
                                                                train_split=config['data'].get('train_split', 0.9),
                                                                max_graphs=config['data'].get('max_graphs'),
                                                                max_train_graphs=config['data'].get('max_train_graphs'),
                                                                )
        if hasattr(dataset, 'max_node_count') and dataset.max_node_count != max_node_count: max_node_count = dataset.max_node_count; print(f"Warn: Updated max_nodes from dataset: {max_node_count}")
        if hasattr(dataset, 'max_level') and dataset.max_level != max_level: max_level = dataset.max_level; print(f"Warn: Updated max_level from dataset: {max_level}")
        print(f"Dataset init OK ({len(dataset)} train graphs)")
        if args.node_type: assert 'node_types' in dataset[0], "Dataset needs 'node_types' for --node_type"; print("Dataset provides 'node_types'.")
    except Exception as e: print(f"FATAL: Dataset init failed: {e}"); return 1

    # --- Setup Models and Criteria ---
    try:
        print("Setting up models and criteria...")
        node_model, edge_model, step_fn_base, node_type_predictor = setup_models(config, device, max_node_count, max_level, train_node_type_flag=args.node_type)
        criterion_edge, use_edge_features, criterion_node = setup_criteria(config, device, dataset, train_node_type_flag=args.node_type)
        print("Models and criteria set up.")
    except Exception as e: print(f"Error setting up models/criteria: {e}"); import traceback; traceback.print_exc(); return 1


    # --- MODIFIED: Setup Optimizers and Schedulers ---
    optim_node_type = None; scheduler_node_type = None
    try:
        print("Setting up optimizers/schedulers...")
        # Determine Learning Rates based on mode (finetune vs normal/restore)
        is_finetuning = args.finetune_from is not None
        if is_finetuning:
            lr_base_models = args.lr_finetune_base
            lr_predictor = args.lr_finetune_pred
            print(f"  Fine-tuning mode LRs: Base={lr_base_models}, Predictor={lr_predictor}")
        else:
            lr_base_models = config['train']['lr']
            # Use specific predictor LR from config if available, else fallback to base LR
            lr_predictor = config['train'].get('lr_node_predictor', lr_base_models)
            print(f"  Normal/Restore mode LRs: Base={lr_base_models}, Predictor={lr_predictor}")

        # Common parameters
        weight_decay_base = config['train'].get('weight_decay', 0.01)
        # Use specific WD for predictor if available, else fallback to base WD
        weight_decay_pred = config['train'].get('weight_decay_node_predictor', weight_decay_base)
        eta_min_value = config['train'].get('eta_min', 0)
        total_steps = config['train']['steps'] # Scheduler T_max based on total intended steps

        # Base models Optimizer/Scheduler
        optim_node_model = torch.optim.AdamW(node_model.parameters(), lr=lr_base_models, weight_decay=weight_decay_base)
        optim_edge_model = torch.optim.AdamW(edge_model.parameters(), lr=lr_base_models, weight_decay=weight_decay_base)
        scheduler_node_model = torch.optim.lr_scheduler.CosineAnnealingLR(optim_node_model, T_max=total_steps, eta_min=eta_min_value)
        scheduler_edge_model = torch.optim.lr_scheduler.CosineAnnealingLR(optim_edge_model, T_max=total_steps, eta_min=eta_min_value)
        print("  Base optims/schedulers OK.")

        # Node predictor Optimizer/Scheduler (if applicable)
        if args.node_type and node_type_predictor is not None:
            print(f"  Node predictor optim: LR={lr_predictor}, WD={weight_decay_pred}")
            optim_node_type = torch.optim.AdamW(node_type_predictor.parameters(), lr=lr_predictor, weight_decay=weight_decay_pred)
            scheduler_node_type = torch.optim.lr_scheduler.CosineAnnealingLR(optim_node_type, T_max=total_steps, eta_min=eta_min_value)
            print("  Node predictor optim/scheduler OK.")

    except KeyError as e: print(f"Error setting up optims/schedulers: Missing key {e} in config['train']"); return 1
    except Exception as e: print(f"Error setting up optims/schedulers: {e}"); return 1
    # --- END MODIFIED ---


    # --- Initialize Logger ---
    log_file_path = config['train'].get('log_file', os.path.join(base_path, 'training_log.csv')); os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_config = config.copy(); log_config['args'] = vars(args)
    logger = SimpleLogger(log_file_path, log_config); print(f"Logger writing to: {log_file_path}")

    # --- Restore Checkpoint OR Load Weights for Fine-tuning ---
    if args.restore_path:
        print("Attempting to resume training...")
        try: global_step = restore_checkpoint(args.restore_path, device, node_model, edge_model, optim_node_model,
                                              optim_edge_model, scheduler_node_model, scheduler_edge_model,
                                              node_type_predictor=node_type_predictor if args.node_type else None,
                                              optim_node_type=optim_node_type if args.node_type else None,
                                              scheduler_node_type=scheduler_node_type if args.node_type else None); print(f"Resumed from step: {global_step}")
        except Exception as e: print(f"Warn: Restore failed: {e}. Starting from scratch."); global_step = 0
    elif args.finetune_from:
        print("Attempting to load weights for fine-tuning...")
        load_successful = load_weights_for_finetuning(args.finetune_from, device, node_model, edge_model,
                                                      node_type_predictor=node_type_predictor if args.node_type else None)
        if not load_successful: print("Warning: Failed to load weights for fine-tuning. Starting from scratch.")
        global_step = 0 # Start step count from 0 for fine-tuning
        print(f"Starting fine-tuning from step {global_step}.")
    else:
        print("Starting training from scratch."); global_step = 0


    # --- Select Step Function ---
    assert step_fn_base is not None, "Base step function is None!"; final_step_fn = step_fn_base
    if args.node_type: print("Using training step function (expected to handle independent node prediction).")
    else: print("Using training step function for edge prediction only.")

    # --- Start Training Loop ---
    print(f"\nStarting training loop from global_step {global_step}...")
    try:
        train_loop(config=config, node_model=node_model, edge_model=edge_model,
                   step_fn=final_step_fn, criterion_edge=criterion_edge, optim_node_model=optim_node_model,
                   optim_edge_model=optim_edge_model, scheduler_node_model=scheduler_node_model,
                   scheduler_edge_model=scheduler_edge_model, device=device, dataset=dataset,
                   global_step=global_step, logger=logger, base_path=base_path, node_type_flag=args.node_type,
                   node_type_predictor=node_type_predictor, criterion_node=criterion_node,
                   optim_node_type=optim_node_type, scheduler_node_type=scheduler_node_type)
    except Exception as loop_e: print(f"\nFATAL ERROR during training loop: {loop_e}"); import traceback; traceback.print_exc(); return 1

    print("\nTraining completed successfully.")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)