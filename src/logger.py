"""
Simple file logger that writes training metrics to a single log file.
"""

import os
import csv
import time
import json
from datetime import datetime

class SimpleLogger:
    """
    A simple logger that writes training metrics to a single file.
    """

    def __init__(self, log_file_path, config):
        """
        Initialize the logger.

        Args:
            log_file_path: Path to the log file
            config: Configuration dictionary
        """
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        self.log_file_path = log_file_path

        # Generate a model descriptor
        self.model_name = self._get_model_name(config)

        # Check if file exists
        file_exists = os.path.exists(log_file_path)

        # Open the file in append mode
        self.file = open(log_file_path, 'a')
        self.writer = csv.writer(self.file)

        # Write header if file is new
        if not file_exists:
            self.writer.writerow([
                'timestamp', 'model_name', 'global_step', 'epoch',
                'total_loss', 'edge_loss', 'lr_node', 'lr_edge',
                'time_per_iter', 'avg_epoch_loss', 'dataset_size'
            ])

        # Write config as a comment
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"# New training run started at {timestamp}\n")
        self.file.write(f"# Model: {self.model_name}\n")

        # Remove excessively large parts of config before writing
        config_copy = config.copy()
        if 'data' in config_copy and 'graph_file' in config_copy['data']:
            config_copy['data']['graph_file'] = os.path.basename(config_copy['data']['graph_file'])

        self.file.write(f"# Config: {json.dumps(config_copy, default=str)}\n")
        self.file.flush()

        print(f"Logger initialized. Logging to {log_file_path}")

    def _get_model_name(self, config):
        """Generate a descriptive model name based on configuration."""
        # Get node model type
        use_lstm = config['model'].get('use_lstm', False)
        use_node_attention = config['model'].get('use_attention', False)

        if use_lstm:
            node_type = "AttLSTM" if use_node_attention else "LSTM"
        else:
            node_type = "AttGRU" if use_node_attention else "GRU"

        # Get edge model type
        edge_model_type = config['model'].get('edge_model', 'mlp').lower()
        if edge_model_type == 'mlp':
            edge_type = "MLP"
        elif edge_model_type == 'rnn':
            edge_type = "RNN"
        elif edge_model_type == 'attention_rnn':
            edge_type = "AttRNN"
        else:
            edge_type = edge_model_type.capitalize()

        # Create name with node and edge types
        model_name = f"{node_type}_{edge_type}"

        # Add hidden sizes if available
        node_hidden = None
        edge_hidden = None

        if use_lstm:
            section = 'GraphAttentionLSTM' if use_node_attention else 'GraphLSTM'
            if section in config['model']:
                node_hidden = config['model'][section].get('hidden_size')
        else:
            section = 'GraphAttentionRNN' if use_node_attention else 'GraphRNN'
            if section in config['model']:
                node_hidden = config['model'][section].get('hidden_size')

        if edge_model_type == 'mlp' and 'EdgeMLP' in config['model']:
            edge_hidden = config['model']['EdgeMLP'].get('hidden_size')
        elif edge_model_type == 'rnn' and 'EdgeRNN' in config['model']:
            edge_hidden = config['model']['EdgeRNN'].get('hidden_size')
        elif edge_model_type == 'attention_rnn' and 'EdgeAttentionRNN' in config['model']:
            edge_hidden = config['model']['EdgeAttentionRNN'].get('hidden_size')

        if node_hidden and edge_hidden:
            model_name += f"_h{node_hidden}_{edge_hidden}"

        return model_name

    def log_step(self, global_step, epoch, total_loss, edge_loss,
                lr_node_model, lr_edge_model, time_per_iter=None,
                avg_epoch_loss=None, dataset_size=None):
        """
        Log metrics for a training step.

        Args:
            global_step: Current global step
            epoch: Current epoch
            total_loss: Total loss value
            edge_loss: Edge prediction loss value
            lr_node_model: Learning rate for node model
            lr_edge_model: Learning rate for edge model
            time_per_iter: Time per iteration (optional)
            avg_epoch_loss: Average loss for the epoch so far (optional)
            dataset_size: Size of the dataset (optional)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.writer.writerow([
            timestamp, self.model_name, global_step, epoch,
            float(total_loss), float(edge_loss),
            float(lr_node_model), float(lr_edge_model),
            float(time_per_iter) if time_per_iter is not None else "",
            float(avg_epoch_loss) if avg_epoch_loss is not None else "",
            dataset_size if dataset_size is not None else ""
        ])
        self.file.flush()

    def log_epoch(self, epoch, avg_loss, steps, global_step, dataset_size=None):
        """
        Log metrics for a training epoch.

        Args:
            epoch: Current epoch number
            avg_loss: Average loss for the epoch
            steps: Number of steps in the epoch
            global_step: Current global step
            dataset_size: Size of the dataset (optional)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.writer.writerow([
            timestamp, self.model_name, global_step, epoch,
            float(avg_loss), "", "", "",  # No loss breakdown or LR for epoch summary
            "", float(avg_loss), dataset_size if dataset_size is not None else ""
        ])
        self.file.flush()

    def close(self):
        """Close the logger and underlying file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"# Training run completed at {timestamp}\n")
        self.file.close()
        print(f"Logging complete. File saved to {self.log_file_path}")