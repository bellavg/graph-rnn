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
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP
from aig_dataset import AIGDataset


def train_mlp_step(graph_rnn, edge_mlp, data, criterion, optim_graph_rnn, optim_edge_mlp,
                   scheduler_graph_rnn, scheduler_mlp, device, use_edge_features):
    """ Train GraphRNN with MLP edge model. """
    graph_rnn.zero_grad()
    edge_mlp.zero_grad()

    s, lens = data['x'].float().to(device), data['len'].cpu()

    # Get truth table if available
    truth_table = data.get('y')
    if truth_table is not None:
        # Flatten truth table - assume shape is [batch, n_outputs, 2^n_inputs]
        # Reshape to [batch, n_outputs * 2^n_inputs]
        truth_table = truth_table.float().to(device)
        if len(truth_table.shape) > 2:
            truth_table = truth_table.reshape(truth_table.shape[0], -1)

    # If s does not have edge features, just add a dummy dimension 1
    # to the end
    if len(s.shape) == 3:
        s = s.unsqueeze(3)

    # Teacher forcing: We want the input to be one node offset from the target.
    # Therefore, we introduce an SOS token (the authors choose the value 1) to
    # the input tensor:
    #       x = [1,  S0, S1, S2, ...,     S(n-1), Sn]
    #       y = [S0, S1, S2, ..., S(n-1), Sn    ,  0]
    one_frame = torch.ones([s.shape[0], 1, s.shape[2], s.shape[3]], device=device)
    zero_frame = torch.zeros([s.shape[0], 1, s.shape[2], s.shape[3]], device=device)
    x = torch.cat((one_frame, s[:, :, :]), dim=1)
    y = torch.cat((s[:, :, :], zero_frame), dim=1)

    lens = lens + 1

    graph_rnn.reset_hidden()
    # Pass truth table to model if it has conditioning capability
    if hasattr(graph_rnn, 'use_conditioning') and graph_rnn.use_conditioning and truth_table is not None:
        hidden = graph_rnn(x, lens, truth_table)
    else:
        hidden = graph_rnn(x, lens)

    # Pass truth table to edge model if it has conditioning capability
    if hasattr(edge_mlp, 'use_conditioning') and edge_mlp.use_conditioning and truth_table is not None:
        y_pred = edge_mlp(hidden, return_logits=use_edge_features, truth_table=truth_table)
    else:
        y_pred = edge_mlp(hidden, return_logits=use_edge_features)

    y = pack_padded_sequence(y, lens, batch_first=True, enforce_sorted=False)
    y, _ = pad_packed_sequence(y, batch_first=True)

    # Edge features use Cross Entropy loss which wants the features in
    # the second dimension, so we have to swap them around
    if use_edge_features:
        # y_pred shape: [batch_size, seq_len+1, m, 3]
        # y shape: [batch_size, seq_len+1, m, 3] (contains 0.0/1.0 floats)

        # 1. Reshape predictions for CrossEntropyLoss: (Batch * Seq * M, Classes=3)
        num_classes = y_pred.shape[-1]  # Should be 3
        # Flatten all dimensions except the class dimension
        y_pred_reshaped = y_pred.reshape(-1, num_classes)

        # 2. Convert target y from one-hot float format to long integer class indices
        #    and reshape to (Batch * Seq * M)
        # Find the index (0, 1, or 2) of the '1.0' in the last dimension
        y_indices = torch.argmax(y, dim=-1)  # Shape: [batch_size, seq_len+1, m]
        # Flatten all dimensions
        y_reshaped = y_indices.reshape(-1)  # Shape: [batch_size * (seq_len+1) * m]

        # 3. Calculate loss (ensure target is long type)
        loss = criterion(y_pred_reshaped, y_reshaped.long())

    else:  # Original BCELoss path (if you were to use binary edges)
        # Note: BCELoss expects inputs and targets of the same shape,
        # containing probabilities or logits (depending on if using BCEWithLogitsLoss)
        loss = criterion(y_pred, y)

    loss.backward()
    optim_graph_rnn.step()
    optim_edge_mlp.step()
    scheduler_graph_rnn.step()
    scheduler_mlp.step()

    return loss.item()


def train_rnn_step(graph_rnn, edge_rnn, data, criterion, optim_graph_rnn, optim_edge_mlp,
                   scheduler_graph_rnn, scheduler_mlp, device, use_edge_features):
    """ Train GraphRNN with RNN edge model. """
    graph_rnn.zero_grad()
    edge_rnn.zero_grad()

    seq, lens = data['x'].float().to(device), data['len'].cpu()

    # Get truth table if available
    truth_table = data.get('y')
    if truth_table is not None:
        # Flatten truth table - assume shape is [batch, n_outputs, 2^n_inputs]
        # Reshape to [batch, n_outputs * 2^n_inputs]
        truth_table = truth_table.float().to(device)
        if len(truth_table.shape) > 2:
            truth_table = truth_table.reshape(truth_table.shape[0], -1)

    # If seq does not have edge features, just add a dummy dimension 1
    # to the end
    if len(seq.shape) == 3:
        seq = seq.unsqueeze(3)

    # Add SOS token to the node-level RNN input to prevent it from looking
    # into the future.
    one_frame = torch.ones([seq.shape[0], 1, seq.shape[2], seq.shape[3]], device=device)
    x_node_rnn = torch.cat((one_frame, seq[:, :-1, :]), dim=1)

    # Compute hidden graph-level representation with optional truth table
    graph_rnn.reset_hidden()
    if hasattr(graph_rnn, 'use_conditioning') and graph_rnn.use_conditioning and truth_table is not None:
        hidden = graph_rnn(x_node_rnn, lens, truth_table)
    else:
        hidden = graph_rnn(x_node_rnn, lens)

    # Process the packed sequence
    seq_packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False).data

    # Compute sequence lengths
    seq_packed_len = []
    m = graph_rnn.input_size
    for l in lens:
        for i in range(1, l + 1):
            seq_packed_len.append(min(i, m))
    seq_packed_len.sort()

    # Add SOS token to the edge-level RNN input
    one_frame = torch.ones([seq_packed.shape[0], 1, seq_packed.shape[2]], device=device)
    x_edge_rnn = torch.cat((one_frame, seq_packed[:, :-1, :]), dim=1)
    y_edge_rnn = seq_packed

    # Pack the hidden representation for the EdgeRNN
    hidden_packed = pack_padded_sequence(hidden, lens, batch_first=True, enforce_sorted=False).data
    edge_rnn.set_first_layer_hidden(hidden_packed)

    # For truth table conditioning in EdgeRNN, we need to create a version of the truth table
    # that matches the packed sequence structure
    edge_truth_table = None
    if truth_table is not None and hasattr(edge_rnn, 'use_conditioning') and edge_rnn.use_conditioning:
        # Create list to hold repeated truth tables
        repeated_tt = []

        for batch_idx, seq_len in enumerate(lens):
            # Repeat this sample's truth table for each node in its graph
            for _ in range(seq_len):
                repeated_tt.append(truth_table[batch_idx])

        # Stack into tensor
        if repeated_tt:
            edge_truth_table = torch.stack(repeated_tt, dim=0)

    # Compute edge probabilities with optional truth table
    if hasattr(edge_rnn, 'use_conditioning') and edge_rnn.use_conditioning and edge_truth_table is not None:
        y_edge_rnn_pred = edge_rnn(x_edge_rnn, seq_packed_len, return_logits=use_edge_features,
                                   truth_table=edge_truth_table)
    else:
        y_edge_rnn_pred = edge_rnn(x_edge_rnn, seq_packed_len, return_logits=use_edge_features)

    y_edge_rnn = pack_padded_sequence(y_edge_rnn, seq_packed_len, batch_first=True, enforce_sorted=False)
    y_edge_rnn, _ = pad_packed_sequence(y_edge_rnn, batch_first=True)

    # Edge features use Cross Entropy loss
    if use_edge_features:
        y_edge_rnn = torch.swapaxes(y_edge_rnn, 1, 2)
        y_edge_rnn = torch.argmax(y_edge_rnn, dim=1)  # One hot to class labels
        y_edge_rnn_pred = torch.swapaxes(y_edge_rnn_pred, 1, 2)

    loss = criterion(y_edge_rnn_pred, y_edge_rnn)
    loss.backward()
    optim_graph_rnn.step()
    optim_edge_mlp.step()
    scheduler_graph_rnn.step()
    scheduler_mlp.step()

    return loss.item()


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

    # Create models
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    if config['model']['edge_model'] == 'rnn':
        # Add tt_size to model parameters if truth table conditioning is enabled
        node_model = GraphLevelRNN(input_size=config['data']['m'],
                                   output_size=config['model']['EdgeRNN']['hidden_size'],
                                   tt_size=tt_size,
                                   **config['model']['GraphRNN']).to(device)

        edge_model = EdgeLevelRNN(tt_size=tt_size,
                                  **config['model']['EdgeRNN']).to(device)
        step_fn = train_rnn_step
    else:
        node_model = GraphLevelRNN(input_size=config['data']['m'],
                                   output_size=None,  # No output layer needed
                                   tt_size=tt_size,
                                   **config['model']['GraphRNN']).to(device)

        edge_model = EdgeLevelMLP(input_size=config['model']['GraphRNN']['hidden_size'],
                                  output_size=config['data']['m'],
                                  tt_size=tt_size,
                                  **config['model']['EdgeMLP']).to(device)
        step_fn = train_mlp_step

    # If we use directed graphs we need edge features, requiring
    # Cross Entropy Loss
    use_edge_features = 'edge_feature_len' in config['model']['GraphRNN'] \
                        and config['model']['GraphRNN']['edge_feature_len'] > 1
    if use_edge_features:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCELoss().to(device)

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
        criterion.load_state_dict(state["criterion"])

    dataset = AIGDataset(
        graph_file=config['data']['graph_file'],
        m=config['data']['m'],
        training=True,
        use_bfs=config['data']['use_bfs'],
        max_graphs=config['data'].get('max_graphs')
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

            loss = step_fn(node_model, edge_model, data, criterion, optim_node_model,
                           optim_edge_model, scheduler_node_model, scheduler_edge_model,
                           device, use_edge_features)
            loss_sum += loss

            # Tensorboard
            writer.add_scalar('loss', loss, global_step)

            if global_step % config['train']['print_iter'] == 0:
                running_time = time.time() - start_time
                time_per_iter = running_time / (global_step - start_step)
                eta = (config['train']['steps'] - global_step) * time_per_iter
                print("[{}] loss={} time_per_iter={:.4f}s eta={}"
                      .format(global_step,
                              loss_sum / config['train']['print_iter'],
                              time_per_iter,
                              datetime.timedelta(seconds=eta)))
                loss_sum = 0

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
                    "criterion": criterion.state_dict()
                }
                print("Saving checkpoint...")
                torch.save(state, os.path.join(base_path, config['train']['checkpoint_dir'],
                                               "checkpoint-{}.pth".format(global_step)))

    writer.close()