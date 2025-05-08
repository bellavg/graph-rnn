import csv
import os
import datetime # Added for timestamp in log_message
import sys # Added for stderr in log_message

class SimpleLogger:
    def __init__(self, file_path, config):
        self.file_path = file_path
        self.config = config
        self.fieldnames = [
            'timestamp', 'global_step', 'epoch',
            'total_loss', 'edge_loss', 'node_loss',
            'lr_node_model', 'lr_edge_model',
            'time_per_iter', 'avg_epoch_loss', 'dataset_size'
        ]
        
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Open file and write header if new
        is_new_file = not os.path.exists(file_path)
        try:
            # Use 'a' to append if file exists, 'w' behavior is handled by writing header if new
            self.f = open(self.file_path, 'a', newline='')
            self.writer = csv.DictWriter(self.f, fieldnames=self.fieldnames)
            if is_new_file:
                self.writer.writeheader()
                self.log_config() # Log config details at the beginning of a new log file
        except IOError as e:
            print(f"Error opening log file {self.file_path}: {e}", file=sys.stderr)
            self.f = None
            self.writer = None


    def log_config(self):
        """Logs the configuration to the file as a comment or separate lines."""
        if self.f and not self.f.closed:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                self.f.write(f"# [{timestamp}] --- Experiment Configuration ---\n")
                import json # For pretty printing dict
                # Convert config to string (e.g., JSON) for logging
                # Be careful with complex objects in config if not easily serializable
                try:
                    config_str = json.dumps(self.config, indent=2) # Assuming config is a dict
                    for line in config_str.splitlines():
                        self.f.write(f"# {line}\n")
                except TypeError:
                    self.f.write(f"# Config could not be serialized to JSON: {str(self.config)}\n")
                self.f.write(f"# [{timestamp}] --- End Configuration ---\n")
                self.f.flush()
            except Exception as e:
                 print(f"Error writing config to log: {e}", file=sys.stderr)


    def log_step(self, global_step, epoch, total_loss, edge_loss, node_loss,
                 lr_node_model, lr_edge_model, time_per_iter, avg_epoch_loss, dataset_size):
        if self.writer and self.f and not self.f.closed:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                self.writer.writerow({
                    'timestamp': timestamp,
                    'global_step': global_step,
                    'epoch': epoch,
                    'total_loss': f"{total_loss:.4f}" if isinstance(total_loss, float) else total_loss,
                    'edge_loss': f"{edge_loss:.4f}" if isinstance(edge_loss, float) else edge_loss,
                    'node_loss': f"{node_loss:.4f}" if isinstance(node_loss, float) else node_loss,
                    'lr_node_model': f"{lr_node_model:.1E}" if isinstance(lr_node_model, float) else lr_node_model,
                    'lr_edge_model': f"{lr_edge_model:.1E}" if isinstance(lr_edge_model, float) else lr_edge_model,
                    'time_per_iter': f"{time_per_iter:.3f}" if isinstance(time_per_iter, float) else time_per_iter,
                    'avg_epoch_loss': f"{avg_epoch_loss:.4f}" if isinstance(avg_epoch_loss, float) else avg_epoch_loss,
                    'dataset_size': dataset_size
                })
                self.f.flush() # Ensure data is written to disk
            except Exception as e:
                print(f"Error writing step to log: {e}", file=sys.stderr)

    def log_epoch(self, epoch, avg_loss, avg_edge_loss, avg_node_loss, steps, global_step, dataset_size):
        # This could be a summary line or integrated into log_step if preferred
        # For now, let's assume it's a distinct log entry if needed, or just print to console.
        # If writing to CSV, ensure it fits the schema or use log_message.
        message = (f"Epoch {epoch} Summary: AvgTotalLoss={avg_loss:.4f}, "
                   f"AvgEdgeLoss={avg_edge_loss:.4f}, AvgNodeLoss={avg_node_loss:.4f}, "
                   f"StepsInEpoch={steps}, GlobalStep={global_step}, DatasetSize={dataset_size}")
        self.log_message(message)


    def log_message(self, message: str):
        """Logs a generic message to the log file, prefixed with a timestamp and '# ' to mark as a comment."""
        if self.f and not self.f.closed:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                self.f.write(f"# [{timestamp}] {message}\n")
                self.f.flush() # Ensure it's written immediately
            except Exception as e:
                print(f"Error writing message to log file: {e}", file=sys.stderr)


    def close(self):
        if self.f and not self.f.closed:
            try:
                self.f.close()
            except Exception as e:
                print(f"Error closing log file: {e}", file=sys.stderr)
            self.f = None
            self.writer = None

