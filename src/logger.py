# src/logger.py

import os
import csv
import time
import json
from datetime import datetime

class SimpleLogger:
    """
    A simple logger that writes training metrics to a single file.
    Handles optional node type prediction logging.
    """

    def __init__(self, log_file_path, config):
        """
        Initialize the logger.

        Args:
            log_file_path: Path to the log file
            config: Configuration dictionary (including 'args' from main.py)
        """
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        self.log_file_path = log_file_path
        self.model_name = self._get_model_name(config)
        # Check if node type prediction is active from args stored in config
        self.log_node_type = config.get('args', {}).get('node_type', False)

        file_exists = os.path.exists(log_file_path)
        self.file = open(log_file_path, 'a', newline='') # Added newline='' for csv
        self.writer = csv.writer(self.file)

        # Write header if file is new
        if not file_exists:
            header = [
                'timestamp', 'model_name', 'global_step', 'epoch',
                'total_loss', 'edge_loss',
                # --- Modified Header ---
                'node_loss', # Added node_loss column
                'lr_node', 'lr_edge',
                'lr_node_type', # Added lr_node_type column
                # --- End Modified Header ---
                'time_per_iter', 'avg_epoch_loss', 'dataset_size'
            ]
            self.writer.writerow(header)

        # Write config and args as comments
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"# New training run started at {timestamp}\n")
        self.file.write(f"# Model: {self.model_name}\n")
        # Log command line args if available
        if 'args' in config:
             self.file.write(f"# Args: {json.dumps(config['args'])}\n")
        # Log main config (simplified)
        config_copy = config.copy()
        config_copy.pop('args', None) # Remove args from main config log
        if 'data' in config_copy and 'graph_file' in config_copy['data']:
            config_copy['data']['graph_file'] = os.path.basename(config_copy['data']['graph_file'])
        self.file.write(f"# Config: {json.dumps(config_copy, default=str)}\n")
        self.file.flush()

        print(f"Logger initialized. Logging to {log_file_path}")
        if self.log_node_type:
             print("  Logging node_loss and lr_node_type.")

    def _get_model_name(self, config):
        """Generate a descriptive model name based on configuration."""
        # --- (Implementation remains the same) ---
        use_lstm = config['model'].get('use_lstm', False)
        use_node_attention = config['model'].get('use_attention', False)
        node_type = ("AttLSTM" if use_node_attention else "LSTM") if use_lstm else ("AttGRU" if use_node_attention else "GRU")
        edge_model_type = config['model'].get('edge_model', 'mlp').lower()
        if edge_model_type == 'mlp': edge_type = "MLP"
        elif edge_model_type == 'rnn': edge_type = "RNN"
        elif edge_model_type == 'attention_rnn': edge_type = "AttRNN"
        else: edge_type = edge_model_type.capitalize()
        model_name = f"{node_type}_{edge_type}"
        node_hidden, edge_hidden = None, None
        node_section_map = { (True, True): 'GraphAttentionLSTM', (True, False): 'GraphLSTM', (False, True): 'GraphAttentionRNN', (False, False): 'GraphRNN'}
        node_config_section = node_section_map[(use_lstm, use_node_attention)]
        if node_config_section not in config['model'] and 'GraphRNN' in config['model']: node_config_section = 'GraphRNN' # Fallback
        if node_config_section in config['model']: node_hidden = config['model'][node_config_section].get('hidden_size')
        edge_section_map = {'mlp': 'EdgeMLP', 'rnn': 'EdgeRNN', 'attention_rnn': 'EdgeAttentionRNN'}
        edge_config_section = edge_section_map.get(edge_model_type)
        if edge_config_section and edge_config_section in config['model']:
             edge_hidden = config['model'][edge_config_section].get('hidden_size')
        if node_hidden and edge_hidden: model_name += f"_h{node_hidden}_{edge_hidden}"
        # Optionally add indicator if node prediction is active
        if config.get('args', {}).get('node_type', False):
             model_name += "_NodePred"
        return model_name

    # --- Modified log_step ---
    def log_step(self, global_step, epoch, total_loss, edge_loss,
                 lr_node_model, lr_edge_model,
                 # --- New optional args ---
                 node_loss=None,
                 lr_node_type=None,
                 # --- End new args ---
                 time_per_iter=None,
                 avg_epoch_loss=None, dataset_size=None):
        """
        Log metrics for a training step, including optional node prediction info.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare row data, handling None values
        row_data = [
            timestamp, self.model_name, global_step, epoch,
            f"{float(total_loss):.6f}" if total_loss is not None else "", # Format floats
            f"{float(edge_loss):.6f}" if edge_loss is not None else "",
            # --- Add node_loss ---
            f"{float(node_loss):.6f}" if node_loss is not None else "",
            # --- Add LRs ---
            f"{float(lr_node_model):.3E}" if lr_node_model is not None else "", # Format LRs
            f"{float(lr_edge_model):.3E}" if lr_edge_model is not None else "",
            # --- Add lr_node_type ---
            f"{float(lr_node_type):.3E}" if lr_node_type is not None else "",
            # --- Remaining columns ---
            f"{float(time_per_iter):.3f}" if time_per_iter is not None else "",
            f"{float(avg_epoch_loss):.6f}" if avg_epoch_loss is not None else "",
            dataset_size if dataset_size is not None else ""
        ]

        self.writer.writerow(row_data)
        self.file.flush()
    # --- End Modified log_step ---

    def log_epoch(self, epoch, avg_loss, steps, global_step, dataset_size=None):
        """
        Log metrics for a training epoch.
        (Note: Currently only logs overall average loss for the epoch)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare row data
        # If you modify train_loop to calculate avg_epoch_edge/node_loss,
        # you could add them here too by modifying the header and this row.
        row_data = [
            timestamp, self.model_name, global_step, epoch,
            f"{float(avg_loss):.6f}", # avg total loss
            "", # Placeholder for avg edge loss
            "", # Placeholder for avg node loss
            "", # Placeholder for lr_node
            "", # Placeholder for lr_edge
            "", # Placeholder for lr_node_type
            "", # Placeholder for time_per_iter
            f"{float(avg_loss):.6f}", # avg_epoch_loss (same as total avg here)
            dataset_size if dataset_size is not None else ""
        ]

        self.writer.writerow(row_data)
        self.file.flush()

    def close(self):
        """Close the logger and underlying file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"# Training run completed at {timestamp}\n")
        self.file.close()
        print(f"Logging complete. File saved to {self.log_file_path}")