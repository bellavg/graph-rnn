import argparse
import yaml
import torch
import os
import time
import datetime

from data import GraphDataSet
from extension_data import DirectedGraphDataSet
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP
from aig_dataset import AIGDataset
from train import train_mlp_step, train_rnn_step

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Path of the config file to use for training',
                        required=False, default="runs/config_aig.yaml")
    parser.add_argument('-r', '--restore', dest='restore_path', required=False, default=None,
                        help='Checkpoint to continue training from')
    parser.add_argument('--gpu', dest='gpu_id', required=False, default=0, type=int,
                        help='Id of the GPU to use')
    args = parser.parse_args()

    base_path = os.path.dirname(args.config_file)

    # Load config
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(os.path.join(base_path, config['train']['checkpoint_dir']), exist_ok=True)
    os.makedirs(os.path.join(base_path, config['train']['log_dir']), exist_ok=True)

    # Calculate truth table size if available
    tt_size = None
    if 'truth_table_conditioning' in config['model'] and config['model']['truth_table_conditioning']:
        n_outputs = config['model'].get('n_outputs', 8)  # Default to 8 outputs
        n_inputs = config['model'].get('n_inputs', 8)  # Default to 8 inputs
        tt_size = n_outputs * (2 ** n_inputs)  # Total size of flattened truth table
        print(f"Using truth table conditioning with size: {tt_size}")

    # Check if node type prediction is enabled
    predict_node_types = config['model'].get('predict_node_types', False)
    if predict_node_types:
        print("Node type prediction is enabled")
        num_node_types = config['model'].get('num_node_types', 4)  # Default: ZERO, PI, AND, PO
        print(f"Using {num_node_types} node types")
    else:
        print("Node type prediction is disabled")
        num_node_types = 4  # Default value, won't be used

    # Create models
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    if config['model']['edge_model'] == 'rnn':
        node_model = GraphLevelRNN(
            input_size=config['data']['m'],
            output_size=config['model']['EdgeRNN']['hidden_size'],
            tt_size=tt_size,
            predict_node_types=predict_node_types,
            num_node_types=num_node_types,
            **config['model']['GraphRNN']
        ).to(device)

        edge_model = EdgeLevelRNN(
            tt_size=tt_size,
            **config['model']['EdgeRNN']
        ).to(device)
        step_fn = train_rnn_step
    else:
        node_model = GraphLevelRNN(
            input_size=config['data']['m'],
            output_size=None,  # No output layer needed
            tt_size=tt_size,
            predict_node_types=predict_node_types,
            num_node_types=num_node_types,
            **config['model']['GraphRNN']
        ).to(device)

        edge_model = EdgeLevelMLP(
            input_size=config['model']['GraphRNN']['hidden_size'],
            output_size=config['data']['m'],
            tt_size=tt_size,
            **config['model']['EdgeMLP']
        ).to(device)
        step_fn = train_mlp_step

    # If we use directed graphs we need edge features, requiring
    # Cross Entropy Loss
    use_edge_features = 'edge_feature_len' in config['model']['GraphRNN'] \
                        and config['model']['GraphRNN']['edge_feature_len'] > 1

    # Create loss functions
    criterions = {'edge': torch.nn.CrossEntropyLoss() if use_edge_features else torch.nn.BCELoss().to(device)}

    # Add node type loss if node type prediction is enabled
    if predict_node_types:
        criterions['node'] = torch.nn.CrossEntropyLoss().to(device)

    # Create optimizers and schedulers
    optim_node_model = torch.optim.Adam(list(node_model.parameters()), lr=config['train']['lr'])
    optim_edge_model = torch.optim.Adam(list(edge_model.parameters()), lr=config['train']['lr'])

    scheduler_node_model = MultiStepLR(optim_node_model,
                                       milestones=config['train']['lr_schedule_milestones'],
                                       gamma=config['train']['lr_schedule_gamma'])
    scheduler_edge_model = MultiStepLR(optim_edge_model,
                                       milestones=config['train']['lr_schedule_milestones'],
                                       gamma=config['train']['lr_schedule_gamma'])

    # Tensorboard
    writer = SummaryWriter(os.path.join(base_path, config['train']['log_dir']))

    global_step = 0

    # Restore from checkpoint
    if args.restore_path:
        print("Restoring from checkpoint: {}".format(args.restore_path))
        state = torch.load(args.restore_path, map_location=device)
        global_step = state["global_step"]
        node_model.load_state_dict(state["node_model"])
        edge_model.load_state_dict(state["edge_model"])
        optim_node_model.load_state_dict(state["optim_node_model"])
        optim_edge_model.load_state_dict(state["optim_edge_model"])
        scheduler_node_model.load_state_dict(state["scheduler_node_model"])
        scheduler_edge_model.load_state_dict(state["scheduler_edge_model"])
        # Load loss functions if available in checkpoint
        if "criterions" in state:
            criterions = state["criterions"]

    # Create dataset
    dataset = AIGDataset(
        graph_file=config['data']['graph_file'],
        m=config['data']['m'],
        training=True,
        use_bfs=config['data'].get('use_bfs', True),
        max_graphs=config['data'].get('max_graphs'),
        include_node_types=predict_node_types  # Only include node types if needed
    )

    data_loader = DataLoader(dataset, batch_size=config['train']['batch_size'])

    node_model.train()
    edge_model.train()

    done = False
    losses = {'total': 0, 'edge': 0, 'node': 0}
    start_step = global_step
    start_time = time.time()

    while not done:
        for data in data_loader:
            global_step += 1
            if global_step > config['train']['steps']:
                done = True
                break

            step_losses = step_fn(
                node_model, edge_model, data, criterions,
                optim_node_model, optim_edge_model,
                scheduler_node_model, scheduler_edge_model,
                device, use_edge_features
            )

            # Accumulate losses
            for key in step_losses:
                if key in losses:
                    losses[key] += step_losses[key]

            # Tensorboard logging
            writer.add_scalar('loss/total', step_losses['total'], global_step)
            writer.add_scalar('loss/edge', step_losses['edge'], global_step)
            if predict_node_types:
                writer.add_scalar('loss/node', step_losses['node'], global_step)

            if global_step % config['train']['print_iter'] == 0:
                running_time = time.time() - start_time
                time_per_iter = running_time / (global_step - start_step)
                eta = (config['train']['steps'] - global_step) * time_per_iter

                # Report all losses
                loss_str = f"total={losses['total'] / config['train']['print_iter']:.4f}"
                loss_str += f", edge={losses['edge'] / config['train']['print_iter']:.4f}"
                if predict_node_types:
                    loss_str += f", node={losses['node'] / config['train']['print_iter']:.4f}"

                print(
                    f"[{global_step}] {loss_str} time_per_iter={time_per_iter:.4f}s eta={datetime.timedelta(seconds=eta)}")

                # Reset loss tracking
                losses = {'total': 0, 'edge': 0, 'node': 0}

            if global_step % config['train']['checkpoint_iter'] == 0 or global_step + 1 > config['train']['steps']:
                state = {
                    "global_step": global_step,
                    "config": config,
                    "node_model": node_model.state_dict(),
                    "edge_model": edge_model.state_dict(),
                    "optim_node_model": optim_node_model.state_dict(),
                    "optim_edge_model": optim_edge_model.state_dict(),
                    "scheduler_node_model": scheduler_node_model.state_dict(),
                    "scheduler_edge_model": scheduler_edge_model.state_dict(),
                    "criterions": criterions
                }
                print("Saving checkpoint...")
                torch.save(state, os.path.join(base_path, config['train']['checkpoint_dir'],
                                               f"checkpoint-{global_step}.pth"))

    writer.close()
    print("Training complete!")