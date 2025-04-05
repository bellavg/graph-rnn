"""Evaluate performance of the graph generation model through visualization and metrics."""

import mmd
import data
import extension_data
import generate
import model
import graph_metrics
import mmd_stanford_impl
import orbit_stats
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import os
import yaml # For loading config from checkpoint state
from tqdm import tqdm
# AIG specific imports
from aig_dataset import AIGDataset, NODE_TYPES, EDGE_TYPES
# Use the AIG specific generation and loading functions
import aig_generate
import mmd # Keep if using MMD metrics
import graph_metrics # Keep standard metrics if desired
import mmd_stanford_impl # Keep if using MMD metrics
import orbit_stats # Keep if using orbit stats (ensure directed graph support)


# --- AIG Validity Calculation ---
def calculate_aig_validity_stats(graph_list):
    """
    Calculate the average AIG validity across a list of graphs.

    Args:
        graph_list: List of NetworkX DiGraph objects representing AIGs.

    Returns:
        float: Average overall validity (percentage of gates with correct in-degree).
        dict: Dictionary containing aggregated validity stats across all graphs.
    """
    total_and = 0
    total_po = 0
    total_valid_and = 0
    total_valid_po = 0
    num_graphs = len(graph_list)

    if num_graphs == 0:
        return 1.0, {'agg_overall_validity': 1.0} # Or 0.0? Assume perfect if no graphs.

    for G in graph_list:
        # Get node types from graph attributes
        node_types = [G.nodes[n].get('type', -1) for n in G.nodes()]

        and_gates = [n for i, n in enumerate(G.nodes()) if node_types[i] == NODE_TYPES['AND']]
        po_gates = [n for i, n in enumerate(G.nodes()) if node_types[i] == NODE_TYPES['PO']]

        total_and += len(and_gates)
        total_po += len(po_gates)

        for gate in and_gates:
            if G.in_degree(gate) == 2:
                total_valid_and += 1

        for gate in po_gates:
            if G.in_degree(gate) == 1:
                total_valid_po += 1

    total_gates_evaluated = total_and + total_po
    total_valid_gates = total_valid_and + total_valid_po

    # Calculate aggregate percentages
    agg_and_validity = (total_valid_and / total_and) if total_and > 0 else 1.0
    agg_po_validity = (total_valid_po / total_po) if total_po > 0 else 1.0
    agg_overall_validity = (total_valid_gates / total_gates_evaluated) if total_gates_evaluated > 0 else 1.0

    detailed_stats = {
        'num_graphs': num_graphs,
        'total_and_gates': total_and,
        'total_po_gates': total_po,
        'total_valid_and': total_valid_and,
        'total_valid_po': total_valid_po,
        'agg_and_validity': agg_and_validity,
        'agg_po_validity': agg_po_validity,
        'agg_overall_validity': agg_overall_validity
    }
    print(f"AIG Validity Stats: {detailed_stats}") # Print detailed stats
    return agg_overall_validity, detailed_stats


# --- Helper function for metrics on a list of graphs ---
def compute_single_list_metric_score(generated_graphs, metric_function):
    """Computes a metric that operates on the entire list of generated graphs."""
    # metric_function takes list of graphs, returns score (or score, stats_dict)
    result = metric_function(generated_graphs)
    if isinstance(result, tuple):
        return result[0] # Return only the primary score if tuple (score, stats)
    else:
        return result

# --- Function to generate graphs (modified) ---
def generate_new_graphs(test_dataset, models, config, device, log_interval=10):
    """
    Generates new AIGs corresponding to graphs in the test dataset.

    Args:
        test_dataset: An initialized AIGDataset (training=False).
        models: Tuple of loaded (node_model, edge_model).
        config: The loaded model configuration dictionary.
        device: torch.device ('cuda' or 'cpu').
        log_interval: Progress printing interval.

    Returns:
        List of generated NetworkX DiGraph objects.
    """
    node_model, edge_model = models
    input_size = config['data']['m']
    edge_gen_function = aig_generate.mlp_edge_gen if config['model']['edge_model'] == 'mlp' else aig_generate.rnn_edge_gen
    mode = config['model'].get('mode', 'directed-multiclass')
    use_conditioning = config['model'].get('truth_table_conditioning', False)

    print(f"Generating {len(test_dataset)} AIGs...")
    generated_graphs = []

    for i in tqdm(range(len(test_dataset))):
        data_item = test_dataset[i]
        # Determine target number of nodes for generation
        # Add 1 because 'len' from dataset is sequence length (num_nodes - 1)
        target_num_nodes = data_item['len'] + 1
        if target_num_nodes <= 0:
            print(f"Warning: Skipping test graph {i} with target_num_nodes <= 0")
            continue

        # Get target truth table if conditioning
        truth_table = None
        if use_conditioning and data_item.get('y') is not None:
            truth_table = data_item['y'].unsqueeze(0).to(device) # Add batch dim

        # Generate the AIG using the function from aig_generate
        # Assuming generate_aig now returns a NetworkX graph or None
        generated_graph = aig_generate.generate_aig(
            num_nodes=target_num_nodes,
            node_model=node_model,
            edge_model=edge_model,
            input_size=input_size,
            edge_gen_function=edge_gen_function,
            mode=mode,
            config=config, # Pass the full config
            truth_table=truth_table
        )

        if generated_graph is not None and generated_graph.number_of_nodes() > 0:
            generated_graphs.append(generated_graph)
        else:
             print(f"Warning: Failed to generate valid graph for test item {i}. Skipping.")
             # Optionally add a placeholder or handle this case
             # generated_graphs.append(nx.DiGraph()) # Add empty graph?

        # if (i + 1) % log_interval == 0:
        #     print(f"Generated {i+1}/{len(test_dataset)} graphs") # tqdm handles progress

    print(f"Done generating graphs. Successfully generated: {len(generated_graphs)}")
    return generated_graphs


def generated_graph_to_networkx(list_adj_vecs, directed=False):
    """
    Convert output of graph generation from model to networkx graph object.

    :param list_adj_vecs: list of torch tensors, each of which is adjacency vector for a node
    :return: networkx graph object
    """
    adj_matrix = np.array(list_adj_vecs)
    return nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph if directed else None)


def draw_generated_graph(list_adj_vecs, file_name="graph", directed=False):
    """
    Draw output of graph generation from model.

    :param list_adj_vecs: list of torch tensors, each of which is adjacency vector for a node
    :param file_name: the file name of the outputted graph drawing
    """
    graph = generated_graph_to_networkx(list_adj_vecs, directed)
    plt.figure()
    pos = nx.spring_layout(graph, k=1 / (np.sqrt(graph.number_of_nodes())), iterations=1000)
    nx.draw(graph, pos=pos)
    plt.savefig(file_name)


def compare_graphs_mmd(graph1, graph2, mmd_func):
    """
    Return MMD (maximum mean discrepancy) score between two graphs.

    :param graph1: networkx graph object of first graph
    :param graph2: networkx graph object of second graph
    :param mmd_func: MMD function to use
    :return: the MMD score between two graphs
    """
    adj_mat1 = nx.to_numpy_array(graph1)
    adj_mat2 = nx.to_numpy_array(graph2)
    return mmd_func(adj_mat1, adj_mat2)


def _diff_func(graph1, graph2, graph_metric_fn):
    """Applies a function to get a metric for each graph and returns the absolute
    value of the distance between them."""
    return abs(graph_metric_fn(graph1) - graph_metric_fn(graph2))


def compare_graphs_avg_degree(graph1, graph2):
    """
    Return difference between average degree of two networkx graphs.

    :param graph1: first networkx graph
    :param graph2: second networkx graph
    :return: absolute value of difference in average degree between two graphs
    """
    return _diff_func(graph1, graph2, graph_metrics.average_degree)


def compare_graphs_avg_clustering_coeff(graph1, graph2):
    """Return difference between avgerage clustering coefficients """
    avg1 = np.mean(np.array(list(nx.clustering(graph1).values())))
    avg2 = np.mean(np.array(list(nx.clustering(graph2).values())))
    return abs(avg1 - avg2)


def compare_graphs_avg_orbit_stats(graph1, graph2):
    """Return difference between avgerage clustering coefficients """
    avg1 = np.mean(np.array(get_orbit_stats([graph1])))
    avg2 = np.mean(np.array(get_orbit_stats([graph2])))
    return abs(avg1 - avg2)


def compare_graphs_avg_degree_centrality(graph1, graph2):
    """Return difference between average degree centrality of two networkx graphs."""
    return _diff_func(graph1, graph2, graph_metrics.average_degree_centrality)


def compare_graphs_avg_betweenness_centrality(graph1, graph2):
    """Return difference between average betweenness centrality of two networkx graphs."""
    return _diff_func(graph1, graph2, graph_metrics.average_betweenness_centrality)


def compare_graphs_avg_closeness_centrality(graph1, graph2):
    """Return difference between average closeness centrality of two networkx graphs."""
    return _diff_func(graph1, graph2, graph_metrics.average_closeness_centrality)


def compare_graphs_avg_eigenvector_centrality(graph1, graph2):
    """Return difference between average eigenvector centrality of two networkx graphs."""
    return _diff_func(graph1, graph2, graph_metrics.average_eigenvector_centrality)


def compare_graphs_transitivity(graph1, graph2):
    """Return difference between transitivity (triadic closure) of two networkx graphs."""
    return _diff_func(graph1, graph2, nx.transitivity)


def compare_graphs_density(graph1, graph2):
    """Return difference between average density of two networkx graphs."""
    return _diff_func(graph1, graph2, nx.density)


def _generate_graph_attribute_list(graph_list, fn):
    """Return a list of the result of applying the given function to each graph in the given list of graphs.

    :param graph_list: a list of networkx graphs
    :param fn: the function to apply to each list
    :return: a list of objects resulting from applying the function to a networkx graph
    """
    graph_attribute_list = []
    for graph in graph_list:
        graph_attribute_list.append(fn(graph))
    return graph_attribute_list


def _mmd_comparison_helper_func(graph_list1, graph_list2, mmd_func, metric_func):
    """Generates list of graph attributes using given metric function and runs MMD on the resulting list of lists."""
    list1 = _generate_graph_attribute_list(graph_list1, metric_func)
    list2 = _generate_graph_attribute_list(graph_list2, metric_func)
    return mmd_func(list1, list2)


def compare_graphs_mmd_degree(graph_list1, graph_list2, mmd_func):
    """Return MMD between distributions of degrees between two lists of networkx graphs."""
    return _mmd_comparison_helper_func(graph_list1, graph_list2, mmd_func, nx.degree_histogram)


def compare_graphs_mmd_clustering_coeff(graph_list1, graph_list2, mmd_func):
    """Return MMD between distributions of clustering coefficients between two lists of networkx graphs."""
    return _mmd_comparison_helper_func(graph_list1, graph_list2, mmd_func,
                                       graph_metrics.get_histogram_of_clustering_coeffs)


def get_orbit_stats(graph_list):
    """Given a list of graphs (networkx graph objects), return the orbit statistics, a list of lists.
    Based on Stanford code."""
    total_counts = []
    for graph in graph_list:
        orbit_counts = orbit_stats.orca(graph)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / graph.number_of_nodes()
        total_counts.append(orbit_counts_graph)
    return total_counts


def compare_graphs_mmd_orbit_stats(graph_list1, graph_list2, mmd_func):
    """Given two graph lists and an MMD function, return the MMD between the orbit statistics of the two graph lists."""
    total_counts1 = np.array(get_orbit_stats(graph_list1))
    total_counts2 = np.array(get_orbit_stats(graph_list2))
    return mmd_func(total_counts1, total_counts2)


def _generate_graph_attribute_list_dict(graph_list, fn):
    """Return a list of the result of applying the given function to each graph in the given list of graphs.
    Used for networkx graph metrics that are outputted as a dict.

    :param graph_list: a list of networkx graphs
    :param fn: the function to apply to each list
    :return: a list of objects resulting from applying the function to a networkx graph
    """
    graph_attribute_list = []
    for graph in graph_list:
        vals = np.array(list(fn(graph).values()))
        graph_attribute_list.append(vals)
    return graph_attribute_list


def _mmd_helper_func_dict(graph_list1, graph_list2, mmd_func, metric_func):
    """Generates list of graph attributes using given metric function (for graph metrics outputting a dict)
    and runs MMD on the resulting list of lists."""
    list1 = np.array(_generate_graph_attribute_list_dict(graph_list1, metric_func), dtype=object)
    list2 = np.array(_generate_graph_attribute_list_dict(graph_list2, metric_func), dtype=object)
    return mmd_func(list1, list2)


def compare_graphs_mmd_degree_centrality(graph_list1, graph_list2, mmd_func):
    """Return MMD between distributions of degree centrality between two lists of networkx graphs."""
    return _mmd_helper_func_dict(graph_list1, graph_list2, mmd_func, nx.degree_centrality)


def compare_graphs_mmd_betweenness_centrality(graph_list1, graph_list2, mmd_func):
    """Return MMD between distributions of betweenness centrality between two lists of networkx graphs."""
    return _mmd_helper_func_dict(graph_list1, graph_list2, mmd_func, nx.betweenness_centrality)


def compare_graphs_mmd_closeness_centrality(graph_list1, graph_list2, mmd_func):
    """Return MMD between distributions of closeness centrality between two lists of networkx graphs."""
    return _mmd_helper_func_dict(graph_list1, graph_list2, mmd_func, nx.closeness_centrality)


def generate_graph(models, num_nodes):
    """
    Use the trained model to generate a graph with a specified number of nodes.

    :param models: pytorch model objects to use
    :param num_nodes: number of nodes in the graph to be outputted
    :return: networkx object of the generated graph
    """
    node_model, edge_model, input_size, edge_gen_function, mode = models
    while True:
        adj_matrix = generate.generate(num_nodes, node_model, edge_model, input_size, edge_gen_function, mode)
        g = nx.to_networkx_graph(adj_matrix, create_using=nx.DiGraph if mode != "undirected" else None)
        if g.number_of_nodes() > 0:
            return g


def compute_average_metric_score(test_graphs, generated_graphs, metric_comparison_fn):
    """
    Compute average metric score between every graph in original testing dataset its corresponding graph generated by model.

    :param test_graphs: a list of NX graphs from the test dataset
    :param generated_graphs: a list of networkx graphs generated from trained model
    :param metric_comparison_fn: function that takes in two networkx graphs and outputs the comparison metric score
    :return: the average metric score between all the graphs
    """
    total_metric_score = 0
    count = 0

    for i, test_graph in enumerate(test_graphs):
        generated_graph = generated_graphs[i]
        metric_val = metric_comparison_fn(test_graph, generated_graph)

        total_metric_score += metric_val
        count += 1

    avg_metric_score = total_metric_score / count
    return avg_metric_score


def compute_average_metric_score_MMD(test_graphs, generated_graphs, metric_comparison_fn):
    """
    Compute average metric score across an entire distribution of graphs.
    The purpose of this function is to compute the MMD score between the distributions of a certain graph
    metric across two entire lists of graphs at once.

    :param test_graphs: a list of networkx graphs from the test dataset
    :param generated_graphs: a list of networkx graphs generated from trained model
    :param metric_comparison_fn: function that takes in two lists of networkx graphs and outputs the comparison metric score
    :return: the  metric score between all the graphs
    """
    return metric_comparison_fn(test_graphs, generated_graphs)


def run_all_metrics(metric_info, model_path, generator_name, f, small_dataset=False):
    """
    Run evaluation metrics for a specific model checkpoint.

    Args:
        metric_info: List of tuples (metric_name, metric_function, computation_type).
        model_path: Path to the model checkpoint file.
        generator_name: Name of the generator ('GraphRNN-AIG').
        f: File handle to write results.
        small_dataset: Whether to use a small subset for testing.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model state and config
    print(f"Loading model state from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        f.write(f"Error: Model file not found at {model_path}\n")
        return
    try:
        state = torch.load(model_path, map_location=device)
        config = state['config']
    except Exception as e:
        print(f"Error loading model state: {e}")
        f.write(f"Error loading model state: {e}\n")
        return

    # Load models using AIG-specific function (pass state directly)
    try:
        models = aig_generate.load_aig_model_from_config(state) # Assuming it takes state dict
        # If it takes path: models = aig_generate.load_aig_model_from_config(model_path)
    except Exception as e:
        print(f"Error loading models from state: {e}")
        f.write(f"Error loading models from state: {e}\n")
        return

    # Load test dataset based on config from the checkpoint
    try:
        graph_file_path = config['data']['graph_file']
        print(f"Loading test dataset from: {graph_file_path}")
        dataset = AIGDataset(
            graph_file=graph_file_path,
            m=config['data']['m'],
            training=False, # Use the test split
            use_bfs=config['data'].get('use_bfs', True),
            include_node_types=config['model'].get('predict_node_types', False),
            max_graphs=10 if small_dataset else config['data'].get('max_graphs') # Limit graphs if small_dataset
        )
        # Actual test graphs (NetworkX objects from AIGDataset.graphs)
        test_graphs_nx = dataset.graphs[dataset.start_idx : dataset.start_idx + dataset.length]

    except Exception as e:
        print(f"Error loading dataset: {e}")
        f.write(f"Error loading dataset: {e}\n")
        return

    if not test_graphs_nx:
        print("Error: No graphs found in the test dataset.")
        f.write("Error: No graphs found in the test dataset.\n")
        return

    # Generate new graphs corresponding to the test set
    # Pass only the models tuple needed by generate_new_graphs
    node_edge_models = (models[0], models[1]) # (node_model, edge_model)
    generated_graphs = generate_new_graphs(dataset, node_edge_models, config, device)

    if not generated_graphs:
        print("Error: Failed to generate any graphs.")
        f.write("Error: Failed to generate any graphs.\n")
        return

    # --- Compute Metrics ---
    f.write(f"\nResults for model: {os.path.basename(model_path)}\n")
    f.write(f"Dataset: {graph_file_path}\n")
    f.write(f"Generated {len(generated_graphs)} graphs.\n")

    for name, fn, computation_type in metric_info:
        try:
            if computation_type == 'list':
                 # Computes metric on the whole list of generated graphs
                 val = compute_single_list_metric_score(generated_graphs, fn)
            # elif computation_type == 'pairwise':
            #      # Computes metric for each test/generated pair and averages
            #      # Ensure test_graphs_nx aligns with generated_graphs if using pairwise
            #      aligned_test_graphs = test_graphs_nx[:len(generated_graphs)]
            #      val = compute_average_metric_score(aligned_test_graphs, generated_graphs, fn)
            elif computation_type == 'mmd':
                 # Computes MMD between test and generated distributions
                 # Ensure test_graphs_nx aligns with generated_graphs
                 aligned_test_graphs = test_graphs_nx[:len(generated_graphs)]
                 val = compute_average_metric_score_MMD(aligned_test_graphs, generated_graphs, fn)
            else:
                 print(f"Warning: Unknown computation type '{computation_type}' for metric '{name}'")
                 val = "N/A"

            print(f"{name}: {val}")
            f.write(f"{name}: {val}\n")
        except Exception as e:
            print(f"Error computing metric '{name}': {e}")
            f.write(f"Error computing metric '{name}': {e}\n")


def evaluate_all_models(model_info, metric_info, generator_name, f, small_dataset=False):
    """Runs evaluation for all specified models."""
    for i in model_info:
        # Tuple now might just contain model path, name is derived inside run_all_metrics
        model_file = i # Assuming model_info is just a list of paths
        print(f"\nEvaluating model: {model_file}")
        run_all_metrics(metric_info, model_file, generator_name, f, small_dataset)
        f.write("---------------\n")


def run_all_generators(generator_list, model_info, metric_info, f, small_dataset=False):
    """
    Run the evaluation suite for all the graph generators.

    :param generator_list: A list of all generation methods used to generate the graphs
    :param model_info: list of tuples containing names of the dataset, dataset name from data.py, and file path of trained model
    :param metric_info: list of tuples containing names of the metric and corresponding metric comparison functions
    :param small_dataset: if True, do not use the full dataset for running all the metrics.
                            Use only a small number of data points for testing.
    """
    for generator in generator_list:
        print("Using generator " + generator)
        f.write("Using generator " + generator + "\n")
        evaluate_all_models(model_info, metric_info, generator, f, small_dataset)
        f.write("===============================================\n")


if __name__ == '__main__':
    # Only GraphRNN-AIG generator makes sense now
    generator_list_vals = ["GraphRNN-AIG"] # Changed generator name

    # put info for all metrics to be run into metric_info
    mmd_stanford_fn_no_hist = lambda x, y: mmd_stanford_impl.compute_mmd(x, y, kernel=mmd_stanford_impl.gaussian_emd,
                                                                         is_hist=False)
    mmd_stanford_fn_is_hist = lambda x, y: mmd_stanford_impl.compute_mmd(x, y, kernel=mmd_stanford_impl.gaussian_emd,
                                                                         is_hist=True)
    mmd_stanford_fn_is_hist_clustering_settings = lambda x, y: mmd_stanford_impl.compute_mmd(x, y,
                                                                                             kernel=mmd_stanford_impl.gaussian_emd,
                                                                                             is_hist=True,
                                                                                             sigma=1.0 / 10,
                                                                                             distance_scaling=100)
    mmd_stanford_fn_orbit_settings = lambda x, y: mmd_stanford_impl.compute_mmd(x, y, kernel=mmd_stanford_impl.gaussian,
                                                                                is_hist=False, sigma=30.0)

    metric_info_vals = [
        # AIG Specific Metrics
        ("AIG Validity", calculate_aig_validity_stats, 'list'),
        ("MMD of degree distribution",
            lambda x, y: compare_graphs_mmd_degree(x, y, mmd_stanford_fn_is_hist)),
        ("MMD of clustering coefficient distribution",
            lambda x, y: compare_graphs_mmd_clustering_coeff(x, y, mmd_stanford_fn_is_hist_clustering_settings)),
        ("MMD of orbit stats distribution",
            lambda x, y: compare_graphs_mmd_orbit_stats(x, y, mmd_stanford_fn_orbit_settings)),

        ("MMD of degree centrality",
            lambda x, y: compare_graphs_mmd_degree_centrality(x, y, mmd_stanford_fn_no_hist)),
        ("MMD of betweenness centrality",
            lambda x, y: compare_graphs_mmd_betweenness_centrality(x, y, mmd_stanford_fn_no_hist)),
        ("MMD of closeness centrality",
            lambda x, y: compare_graphs_mmd_closeness_centrality(x, y, mmd_stanford_fn_no_hist)),

        ("Avg of degree distribution",
         lambda x, y: compare_graphs_avg_degree(x, y)),
        ("Avg of clustering coefficient distribution",
         lambda x, y: compare_graphs_avg_clustering_coeff(x, y)),
        ("Avg of orbit stats distribution",
         lambda x, y: compare_graphs_avg_orbit_stats(x, y)),

        ("Average Degree Difference", compare_graphs_avg_degree),
        ("Average Degree Centrality Difference", compare_graphs_avg_degree_centrality),
        ("Average Betweenness Centrality Difference", compare_graphs_avg_betweenness_centrality),
        ("Average Closeness Centrality Difference", compare_graphs_avg_closeness_centrality),
        ("Density Difference", compare_graphs_density),
        ("Transitivity (Triadic Closure) Difference", compare_graphs_transitivity)
    ]

    # put info for all models to be run into model_info_vals
    model_info_vals = [
        # Example: Replace with your actual checkpoint path(s)
        "runs/tn_checkpoints/checkpoint-10000.pth",
        # "path/to/another/checkpoint.pth",
    ]

    # --- Run Evaluation ---
    output_filename = "aig_eval_results.txt"
    print(f"Starting evaluation. Results will be saved to {output_filename}")
    with open(output_filename, "w") as f:
        f.write("AIG Generation Model Evaluation Results\n")
        f.write("=====================================\n")
        # Simplified loop as we only have one generator type now
        generator_name = generator_list_vals[0]
        print(f"Using generator {generator_name}")
        f.write(f"Using generator {generator_name}\n")
        evaluate_all_models(model_info_vals, metric_info_vals, generator_name, f,
                            small_dataset=False)  # Set small_dataset=True for quick test
        f.write("===============================================\n")

    print(f"Evaluation complete. Results saved to {output_filename}")