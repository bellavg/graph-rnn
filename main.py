import argparse
import yaml
import torch
import os
import time
import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from train import train_rnn_step, train_mlp_step
from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP
from aig_dataset import AIGDataset


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
    return config


def setup_models(config, device):
    predict_node_types = config['model'].get('predict_node_types', False)
    use_conditioning = config['model'].get('truth_table_conditioning', False)
    num_node_types = config['model'].get('num_node_types', None)

    tt_size = None
    if use_conditioning:
        n_outputs = config['model'].get('n_outputs', 2)
        n_inputs = config['model'].get('n_inputs', 8)
        tt_size = n_outputs * (2 ** n_inputs)
        print(f"Using truth table conditioning with size: {tt_size}")

    if config['model']['edge_model'] == 'rnn':
        edge_model = EdgeLevelRNN(
            use_conditioning=use_conditioning, tt_size=tt_size,
            **config['model']['EdgeRNN']).to(device)
        step_fn = train_rnn_step
    else:
        edge_model = EdgeLevelMLP(
            input_size=config['model']['GraphRNN']['hidden_size'],
            output_size=config['data']['m'],
            use_conditioning=use_conditioning, tt_size=tt_size,
            **config['model']['EdgeMLP']).to(device)
        step_fn = train_mlp_step

    node_model = GraphLevelRNN(
        input_size=config['data']['m'],
        output_size=None if config['model']['edge_model'] == 'mlp' else config['model']['EdgeRNN']['hidden_size'],
        predict_node_types=predict_node_types,
        num_node_types=num_node_types,
        use_conditioning=use_conditioning,
        tt_size=tt_size,
        **config['model']['GraphRNN']).to(device)

    return node_model, edge_model, step_fn, predict_node_types, use_conditioning, num_node_types, tt_size


def setup_criteria(config, device, predict_node_types):
    use_edge_features = config['model']['GraphRNN'].get('edge_feature_len', 1) > 1
    criterion_edge = torch.nn.CrossEntropyLoss().to(device) if use_edge_features else torch.nn.BCELoss().to(device)

    criterion_node = None
    if predict_node_types:
        criterion_node = torch.nn.CrossEntropyLoss(ignore_index=-100).to(device)
        print("Node type prediction enabled. Using CrossEntropyLoss for nodes.")
    return criterion_edge, criterion_node, use_edge_features


def restore_checkpoint(path, device, node_model, edge_model,
                       optim_node_model, optim_edge_model,
                       scheduler_node_model, scheduler_edge_model,
                       criterion_edge, criterion_node):
    print(f"Restoring from checkpoint: {path}")
    state = torch.load(path, map_location=device)
    global_step = state["global_step"]
    node_model.load_state_dict(state["node_model"])
    edge_model.load_state_dict(state["edge_model"])
    optim_node_model.load_state_dict(state["optim_node_model"])
    optim_edge_model.load_state_dict(state["optim_edge_model"])
    scheduler_node_model.load_state_dict(state["scheduler_node_model"])
    scheduler_edge_model.load_state_dict(state["scheduler_edge_model"])

    try:
        if "criterion_edge" in state:
            criterion_edge.load_state_dict(state["criterion_edge"])
        if criterion_node and "criterion_node" in state:
            criterion_node.load_state_dict(state["criterion_node"])
    except RuntimeError as e:
        print(f"Warning: could not restore some criteria. Reason: {e}")

    return global_step


def train_loop(config, node_model, edge_model, step_fn, criterion_edge, criterion_node,
               optim_node_model, optim_edge_model, scheduler_node_model, scheduler_edge_model,
               device, use_edge_features, predict_node_types, use_conditioning,
               global_step, writer, base_path):

    dataset = AIGDataset(
        graph_file=config['data']['graph_file'],
        m=config['data']['m'],
        training=True,
        use_bfs=config['data']['use_bfs'],
        max_graphs=config['data'].get('max_graphs'),
        include_node_types=config['model'].get('predict_node_types'),
    )
    data_loader = DataLoader(dataset, batch_size=config['train']['batch_size'])

    node_model.train()
    edge_model.train()

    done = False
    loss_sum = 0
    start_step = global_step
    start_time = time.time()

    while not done:
        for data in data_loader:
            global_step += 1
            if global_step > config['train']['steps']:
                done = True
                break

            loss_dict = step_fn(node_model, edge_model, data,
                                criterion_edge, criterion_node,
                                optim_node_model, optim_edge_model,
                                scheduler_node_model, scheduler_edge_model,
                                device, use_edge_features,
                                predict_node_types, use_conditioning)

            loss_sum += loss_dict['total']

            writer.add_scalar('loss/total', loss_dict['total'], global_step)
            writer.add_scalar('loss/edge', loss_dict['edge'], global_step)
            if 'node' in loss_dict:
                writer.add_scalar('loss/node', loss_dict['node'], global_step)

            if global_step % config['train']['print_iter'] == 0:
                time_per_iter = (time.time() - start_time) / (global_step - start_step)
                eta = (config['train']['steps'] - global_step) * time_per_iter
                print(f"[{global_step}] loss={loss_sum / config['train']['print_iter']:.4f} "
                      f"time_per_iter={time_per_iter:.4f}s eta={datetime.timedelta(seconds=int(eta))}")
                loss_sum = 0

            if global_step % config['train']['checkpoint_iter'] == 0 or global_step + 1 > config['train']['steps']:
                checkpoint_dir = os.path.join(base_path, config['train']['checkpoint_dir'])
                os.makedirs(checkpoint_dir, exist_ok=True)  # ✅ Ensure directory exists

                checkpoint_path = os.path.join(base_path, config['train']['checkpoint_dir'],
                                               f"checkpoint-{global_step}.pth")
                print("Saving checkpoint...")
                torch.save({
                    "global_step": global_step,
                    "config": config,
                    "node_model": node_model.state_dict(),
                    "edge_model": edge_model.state_dict(),
                    "optim_node_model": optim_node_model.state_dict(),
                    "optim_edge_model": optim_edge_model.state_dict(),
                    "scheduler_node_model": scheduler_node_model.state_dict(),
                    "scheduler_edge_model": scheduler_edge_model.state_dict(),
                    "criterion_edge": criterion_edge.state_dict(),
                    "criterion_node": criterion_node.state_dict() if criterion_node else None,
                }, checkpoint_path)

    writer.close()


def main():
    args = parse_args()
    config = load_config(args.config_file)

    base_path = os.path.dirname(args.save_dir)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    node_model, edge_model, step_fn, predict_node_types, use_conditioning, _, _ = setup_models(config, device)
    criterion_edge, criterion_node, use_edge_features = setup_criteria(config, device, predict_node_types)

    optim_node_model = torch.optim.Adam(node_model.parameters(), lr=config['train']['lr'])
    optim_edge_model = torch.optim.Adam(edge_model.parameters(), lr=config['train']['lr'])
    scheduler_node_model = MultiStepLR(optim_node_model, milestones=config['train']['lr_schedule_milestones'],
                                       gamma=config['train']['lr_schedule_gamma'])
    scheduler_edge_model = MultiStepLR(optim_edge_model, milestones=config['train']['lr_schedule_milestones'],
                                       gamma=config['train']['lr_schedule_gamma'])

    writer = SummaryWriter(os.path.join(base_path, config['train']['log_dir']))
    global_step = 0

    if args.restore_path:
        global_step = restore_checkpoint(args.restore_path, device, node_model, edge_model,
                                         optim_node_model, optim_edge_model,
                                         scheduler_node_model, scheduler_edge_model,
                                         criterion_edge, criterion_node)

    train_loop(config, node_model, edge_model, step_fn,
               criterion_edge, criterion_node,
               optim_node_model, optim_edge_model,
               scheduler_node_model, scheduler_edge_model,
               device, use_edge_features, predict_node_types, use_conditioning,
               global_step, writer, base_path)


if __name__ == "__main__":
    main()
