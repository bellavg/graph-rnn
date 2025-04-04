"""
AIG-Specific Graph Generation using Trained GraphRNN Model
"""

import argparse
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP
from aig_dataset import NODE_TYPES, EDGE_TYPES
from generate import *

# Node type color mapping for visualization
NODE_TYPE_COLORS = {
    NODE_TYPES['ZERO']: 'gray',
    NODE_TYPES['PI']: 'green',
    NODE_TYPES['AND']: 'blue',
    NODE_TYPES['PO']: 'red'
}


def generate_aig(num_nodes, node_model, edge_model, input_size, edge_gen_function, mode):
    """
    Generates an And-Inverter Graph with specific constraints

    :param num_nodes: Number of nodes to generate
    :param node_model: Graph-level RNN model
    :param edge_model: Edge-level model (MLP or RNN)
    :param input_size: Maximum number of previous nodes to connect
    :param edge_gen_function: Edge generation method
    :param mode: Generation mode (directed-multiclass)
    :return: Adjacency matrix and node types
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    node_model.eval()
    edge_model.eval()

    # Use softmax sampling for multi-class edge types
    sample_fun = lambda x: np.argmax(torch.softmax(x, dim=0).numpy())

    # Initialize adjacency vector and node types
    adj_vec = torch.ones([1, 1, input_size, node_model.edge_feature_len], device=device)
    list_adj_vecs = []
    node_types = []

    # Simple heuristic for node type generation
    def generate_node_type(current_nodes):
        """
        Simple node type generation heuristic
        - First few nodes are PIs
        - Middle nodes are AND gates
        - Last few nodes are POs
        """
        if len(current_nodes) < 2:
            return NODE_TYPES['PI']
        elif len(current_nodes) > num_nodes - 3:
            return NODE_TYPES['PO']
        else:
            return NODE_TYPES['AND']

    node_model.reset_hidden()

    for i in range(1, num_nodes):
        # Generate node type
        current_node_type = generate_node_type(node_types)
        node_types.append(current_node_type)

        # Generate graph state vector
        h = node_model(adj_vec)

        # Generate edges for this node
        adj_vec = edge_gen_function(
            edge_model,
            h,
            num_edges=min(i, input_size),
            adj_vec_size=input_size,
            sample_fun=sample_fun,
            attempts=1
        )

        # Process adjacency vector
        if mode == 'directed-multiclass':
            # Turn one-hot into class index
            one_hot = adj_vec[0, 0, :min(num_nodes, input_size), :].cpu().detach().int().numpy()
            class_vec = np.zeros([min(num_nodes, input_size)])
            class_vec[:i] = one_hot[:i].nonzero()[1]
            list_adj_vecs.append(class_vec)

        # Early stopping if no more edges
        if np.array(list_adj_vecs[-1] == 0).all():
            break

    # Convert to adjacency matrix
    adj = generate.m_seq_to_adj_mat(np.array(list_adj_vecs), m=input_size)
    adj = adj + adj.T
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.tril(adj)

    # Handle directed multiclass edge types
    if mode == 'directed-multiclass':
        adj = (adj % 2) + (adj // 2).T

    # Truncate node types to match final adjacency matrix
    node_types = node_types[:adj.shape[0]]

    return adj, node_types


def visualize_aig(adj_matrix, node_types, output_file='generated_aig.png'):
    """
    Visualize the generated AIG graph

    :param adj_matrix: Adjacency matrix
    :param node_types: List of node types
    :param output_file: File to save visualization
    """
    # Create networkx graph
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Set node colors based on types
    node_colors = [NODE_TYPE_COLORS.get(nt, 'gray') for nt in node_types]

    # Visualization
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=500,
        arrowsize=20,
        font_size=10,
        font_weight='bold'
    )

    # Add a color legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, marker='o', linestyle='None',
                   markersize=10, label=type_name)
        for type_name, color in NODE_TYPE_COLORS.items()
    ]
    plt.legend(handles=legend_elements, title='Node Types')

    plt.title('Generated AIG Graph')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return G


def load_aig_model_from_config(model_path):
    """
    Load AIG-specific models from checkpoint

    :param model_path: Path to model checkpoint
    :return: Loaded models and configuration details
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state = torch.load(model_path, map_location=device)
    config = state['config']

    input_size = config['data']['m']

    if config['model']['edge_model'] == 'rnn':
        node_model = GraphLevelRNN(
            input_size=config['data']['m'],
            output_size=config['model']['EdgeRNN']['hidden_size'],
            **config['model']['GraphRNN']
        ).to(device)
        edge_model = EdgeLevelRNN(**config['model']['EdgeRNN']).to(device)
        edge_gen_function = rnn_edge_gen
    else:
        node_model = GraphLevelRNN(
            input_size=config['data']['m'],
            output_size=None,
            **config['model']['GraphRNN']
        ).to(device)
        edge_model = EdgeLevelMLP(
            input_size=config['model']['GraphRNN']['hidden_size'],
            output_size=config['data']['m'],
            **config['model']['EdgeMLP']
        ).to(device)
        edge_gen_function = mlp_edge_gen

    node_model.load_state_dict(state['node_model'])
    edge_model.load_state_dict(state['edge_model'])

    mode = config['model']['mode'] if 'mode' in config['model'] else 'directed-multiclass'

    return node_model, edge_model, input_size, edge_gen_function, mode


def main():
    """
    Main script for generating AIGs
    """
    parser = argparse.ArgumentParser(description='AIG Graph Generation')
    parser.add_argument('model_path', help='Path to trained model checkpoint')
    parser.add_argument('-n', '--nodes', type=int, default=60,
                        help='Number of nodes to generate')
    parser.add_argument('-g', '--graphs', type=int, default=5,
                        help='Number of graphs to generate')
    parser.add_argument('-o', '--output_dir', default='generated_aigs',
                        help='Directory to save generated graphs')

    args = parser.parse_args()

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    node_model, edge_model, input_size, edge_gen_function, mode = load_aig_model_from_config(args.model_path)

    # Generate multiple graphs
    generated_graphs = []
    for i in range(args.graphs):
        adj_matrix, node_types = generate_aig(
            args.nodes,
            node_model,
            edge_model,
            input_size,
            edge_gen_function,
            mode
        )

        # Visualize and save graph
        output_file = os.path.join(args.output_dir, f'generated_aig_{i}.png')
        G = visualize_aig(adj_matrix, node_types, output_file)

        generated_graphs.append((G, adj_matrix, node_types))

    # Print some basic statistics
    print(f"Generated {args.graphs} AIGs with {args.nodes} nodes:")
    for i, (G, adj_matrix, node_types) in enumerate(generated_graphs):
        print(f"\nGraph {i}:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print("  Node Type Distribution:")
        type_counts = {
            'ZERO': node_types.count(NODE_TYPES['ZERO']),
            'PI': node_types.count(NODE_TYPES['PI']),
            'AND': node_types.count(NODE_TYPES['AND']),
            'PO': node_types.count(NODE_TYPES['PO'])
        }
        for type_name, count in type_counts.items():
            print(f"    {type_name}: {count}")


if __name__ == '__main__':
    main()