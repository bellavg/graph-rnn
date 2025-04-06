


"""
AIG Evaluation: All-in-one script for evaluating GraphRNN models trained to generate AIGs
"""

import sys
from tqdm import tqdm
from model import *
import json
from aig_generate import *
import networkx as nx
import matplotlib.pyplot as plt
# Required for graphviz layout
from networkx.drawing.nx_pydot import graphviz_layout

# Import required modules - you may need to adjust these imports based on your project structure
# AIG dataset includes NODE_TYPES, EDGE_TYPES constants
from aig_dataset import AIGDataset, NODE_TYPES

# For MMD metrics
try:
    import mmd
    import mmd_stanford_impl

    MMD_AVAILABLE = True
except ImportError:
    print("Warning: MMD modules not available. MMD metrics will be skipped.")
    MMD_AVAILABLE = False

# For graph metrics
try:
    import graph_metrics

    GRAPH_METRICS_AVAILABLE = True
except ImportError:
    print("Warning: graph_metrics module not available. Some graph metrics will be skipped.")
    GRAPH_METRICS_AVAILABLE = False

# For orbit statistics
try:
    import orbit_stats

    ORBIT_AVAILABLE = True
except ImportError:
    print("Warning: orbit_stats module not available. Orbit metrics will be skipped.")
    ORBIT_AVAILABLE = False


# ==========================================
# Helper Functions for Graph Generation
# ==========================================

def evaluate_multiple_models(model_paths, num_graphs=50, min_nodes=10, max_nodes=100,
                             output_dir='evaluation_results', use_conditioning=False,
                             predict_node_types=None, validity_threshold=0.0, seed=42,
                             test_dataset_path=None):
    """
    Evaluate multiple AIG model checkpoints.

    Args:
        model_paths: List of paths to model checkpoints
        num_graphs: Number of graphs to generate per model
        min_nodes: Minimum number of nodes per graph
        max_nodes: Maximum number of nodes per graph
        output_dir: Directory to save results
        use_conditioning: Whether to use truth table conditioning
        predict_node_types: Whether the model predicts node types (overrides config if not None)
        validity_threshold: Minimum validity threshold (0-1)
        seed: Random seed for reproducibility
        test_dataset_path: Optional path to test dataset for comparison

    Returns:
        List of summaries for each model
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate each model
    results = []

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n==== Evaluating model: {model_name} ====")

        # Create subdirectory for this model
        try:
            checkpoint_step = int(model_name.split("checkpoint-")[1].split(".")[0])
            model_dir = os.path.join(output_dir, f"step_{checkpoint_step}")
        except:
            model_dir = os.path.join(output_dir, model_name)

        os.makedirs(model_dir, exist_ok=True)

        # Run evaluation
        try:
            summary = evaluate_model(
                model_path=model_path,
                num_graphs=num_graphs,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                output_dir=model_dir,
                use_conditioning=use_conditioning,
                predict_node_types=predict_node_types,
                validity_threshold=validity_threshold,
                seed=seed,
                test_dataset_path=test_dataset_path
            )

            results.append(summary)
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            results.append({
                "model_path": model_path,
                "error": str(e)
            })

    # Save overall results
    if results:
        with open(os.path.join(output_dir, 'all_models_summary.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Print comparison of key metrics across models
        print("\n===== Model Comparison =====")
        print(f"{'Model':<30} {'Valid/Total':<15} {'Validity %':<12} {'Avg Nodes':<12} {'Avg Edges':<12}")
        print("-" * 85)

        for result in sorted(results, key=lambda x: x.get('checkpoint_step', 0)):
            if 'error' in result:
                model_name = os.path.basename(result['model_path'])
                print(f"{model_name:<30} Error: {result['error']}")
                continue

            model_name = os.path.basename(result['model_path'])
            valid = result['valid_graphs']
            total = result['generated_graphs']
            validity_pct = result['validity_rate'] * 100
            avg_nodes = result['avg_nodes']
            avg_edges = result['avg_edges']

            print(
                f"{model_name:<30} {f'{valid}/{total}':<15} {validity_pct:<12.2f} {avg_nodes:<12.2f} {avg_edges:<12.2f}")

    return results


def find_checkpoints_in_directory(model_dir, pattern="checkpoint-", extension=".pth"):
    """Find model checkpoints in a directory matching the pattern and extension."""
    checkpoints = []
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if pattern in file and file.endswith(extension):
                checkpoints.append(os.path.join(model_dir, file))

    # Sort by checkpoint step
    def get_step(path):
        try:
            return int(os.path.basename(path).split(pattern)[1].split(".")[0])
        except:
            return 0

    return sorted(checkpoints, key=get_step)


def create_random_truth_table(n_outputs=2, n_inputs=8, device='cpu'):
    """
    Create a random truth table for conditioning

    Args:
        n_outputs: Number of output bits
        n_inputs: Number of input bits
        device: Device to place tensor on

    Returns:
        Tuple of (flattened_tensor, original_table)
    """
    # Create random binary truth table - shape [n_outputs, 2^n_inputs]
    table = torch.randint(0, 2, (n_outputs, 2 ** n_inputs), dtype=torch.float)

    # Flatten to [1, n_outputs * 2^n_inputs] for conditioning
    flattened = table.reshape(1, -1).to(device)

    return flattened, table


def visualize_aig(G, output_file='generated_aig_layered.png', title=None):
    """
    Visualize the connected components of an AIG using a layered layout
    (requires Graphviz and pydot), excluding isolated nodes.

    Args:
        G: NetworkX DiGraph object representing the AIG.
        output_file: File to save visualization.
        title: Optional title for the plot.
    """
    if G is None or G.number_of_nodes() == 0:
        print("Cannot visualize empty or None graph.")
        return

    # --- START: Exclude isolated nodes ---
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        print(f"Visualization: Excluding {len(isolated_nodes)} isolated nodes: {isolated_nodes}")
        # Create a subgraph containing only the non-isolated nodes
        nodes_to_keep = [n for n in G.nodes() if n not in isolated_nodes]
        G_connected = G.subgraph(nodes_to_keep)
        if G_connected.number_of_nodes() == 0:
             print("Graph has no connected nodes to visualize.")
             return # Or handle as needed, e.g., save an empty plot
    else:
        G_connected = G # No isolates, use the original graph
    # --- END: Exclude isolated nodes ---

    # --- Use graphviz_layout for hierarchical structure ---
    try:
        pos = graphviz_layout(G_connected, prog="dot") # Use G_connected
    except ImportError:
        print("Warning: pydot or graphviz not found. Falling back to spring_layout.")
        pos = nx.spring_layout(G_connected, seed=42) # Use G_connected
    except Exception as e:
        print(f"Warning: graphviz_layout failed ({e}). Falling back to spring_layout.")
        pos = nx.spring_layout(G_connected, seed=42) # Use G_connected

    plt.figure(figsize=(12, 10)) # Adjust figure size if needed

    # Define node color and labels based on connectivity (using G_connected)
    node_colors = []
    node_labels = {}
    # Iterate through the nodes present in the connected subgraph
    for node in G_connected.nodes(): # Use G_connected
        in_deg = G_connected.in_degree(node) # Use G_connected
        out_deg = G_connected.out_degree(node) # Use G_connected

        # Note: Degree checks might be less meaningful if PIs/POs were isolated and removed
        if in_deg == 0:
            node_colors.append('lightgreen') # Input
            node_labels[node] = f"PI {node}"
        elif in_deg == 2:
            node_colors.append('lightblue') # AND
            node_labels[node] = f"AND {node}"
        elif in_deg == 1 and out_deg == 0:
            node_colors.append('salmon') # Output
            node_labels[node] = f"PO {node}"
        else:
            node_colors.append('lightgray') # Intermediate/Other
            node_labels[node] = f"{node}" # Just node number

    # Draw nodes using G_connected
    nx.draw_networkx_nodes(G_connected, pos, node_color=node_colors, node_size=400, alpha=0.9)

    # Draw edges with styles based on 'type' attribute (using G_connected)
    regular_edges = [(u, v) for u, v, d in G_connected.edges(data=True) if d.get('type') == 1]
    inverted_edges = [(u, v) for u, v, d in G_connected.edges(data=True) if d.get('type') == 2]
    other_edges = [(u, v) for u, v, d in G_connected.edges(data=True) if d.get('type') not in [1, 2]]

    nx.draw_networkx_edges(G_connected, pos, edgelist=regular_edges, width=1.0, edge_color='black', style='solid', arrows=True, arrowsize=10)
    nx.draw_networkx_edges(G_connected, pos, edgelist=inverted_edges, width=1.0, edge_color='red', style='dashed', arrows=True, arrowsize=10)
    if other_edges:
        print(f"Warning: Found edges with unexpected types in connected subgraph: {other_edges}")
        nx.draw_networkx_edges(G_connected, pos, edgelist=other_edges, width=0.5, edge_color='gray', style='dotted', arrows=True, arrowsize=10)

    # Draw labels using G_connected
    nx.draw_networkx_labels(G_connected, pos, labels=node_labels, font_size=8)

    # Add legend (copied from your provided code)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=8, label='Input (PI)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=8, label='AND Gate'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=8, label='Output (PO)'),
        plt.Line2D([0], [0], color='black', linestyle='solid', label='Regular Edge'),
        plt.Line2D([0], [0], color='red', linestyle='dashed', label='Inverted Edge')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize='small')

    plt.title(title if title else 'Generated AIG (Layered Layout, Connected Only)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Generated graph visualization saved to {output_file}")

    # Return the original G or the connected subgraph? Return G_connected for consistency.
    return G_connected


# ==========================================
# Evaluation Functions
# ==========================================

import networkx as nx # Make sure networkx is imported

def calculate_aig_structure_validity(G): # Removed node_types argument as base model doesn't predict them
    """
    Calculates the validity of an AIG based on stricter structural rules:
    1. Must be a Directed Acyclic Graph (DAG).
    2. Nodes must have in-degree 0 (PI), 1 (PO - only if out-degree is 0), or 2 (AND).
    3. Must contain at least one potential PI, PO, and AND gate.
    """
    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        return {'valid': False, 'reason': 'Empty graph', 'num_nodes': 0}

    # Rule 1: Must be a DAG
    if not nx.is_directed_acyclic_graph(G):
        return {'valid': False, 'reason': 'Graph contains cycles', 'num_nodes': num_nodes}

    # Rule 2: Check in-degrees of all nodes
    is_structurally_valid = True
    reason = "OK"
    potential_pis = 0
    potential_ands = 0
    potential_pos = 0

    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        if in_deg == 0:
            potential_pis += 1
            # Optional: Check if PIs have outgoing edges (they should)
            # if out_deg == 0:
            #     is_structurally_valid = False
            #     reason = f"Potential PI node {node} has no outgoing edges"
            #     break
        elif in_deg == 1:
            # Must be a PO, which should have no outgoing edges
            if out_deg == 0:
                potential_pos += 1
            else:
                # If in-degree is 1 but it has outputs, it's not a valid PO/structure
                is_structurally_valid = False
                reason = f"Node {node} has in-degree 1 but out-degree {out_deg} (expected 0 for PO)"
                break
        elif in_deg == 2:
            # Assumed to be an AND gate
            potential_ands += 1
        else:
            # Any other in-degree is invalid
            is_structurally_valid = False
            reason = f"Node {node} has invalid in-degree {in_deg} (must be 0, 1, or 2)"
            break

    # Rule 3: Check if we have at least one of each potential type
    has_min_components = (potential_pis > 0 and potential_ands > 0 and potential_pos > 0)
    if is_structurally_valid and not has_min_components:
        reason = "Graph lacks potential PIs, ANDs, or POs"

    final_validity = is_structurally_valid and has_min_components

    stats = {
        'num_nodes': num_nodes,
        'num_edges': G.number_of_edges(),
        'is_dag': True, # Already checked
        'num_potential_pis': potential_pis,
        'num_potential_ands': potential_ands,
        'num_potential_pos': potential_pos,
        'valid': final_validity,
        'reason': reason
    }
    return stats


def compute_graph_metrics(graphs, test_graphs=None):
    """
    Compute various graph metrics on a list of graphs.

    Args:
        graphs: List of NetworkX graphs to analyze
        test_graphs: Optional list of reference graphs for comparison

    Returns:
        dict: Dictionary containing metric results
    """
    results = {}

    # Skip if no graphs to evaluate
    if not graphs:
        return {"error": "No graphs to evaluate"}

    # Basic graph statistics
    n_nodes = [g.number_of_nodes() for g in graphs]
    n_edges = [g.number_of_edges() for g in graphs]

    results["avg_nodes"] = sum(n_nodes) / len(n_nodes)
    results["avg_edges"] = sum(n_edges) / len(n_edges)
    results["min_nodes"] = min(n_nodes)
    results["max_nodes"] = max(n_nodes)
    results["min_edges"] = min(n_edges)
    results["max_edges"] = max(n_edges)

    # Graph metrics from graph_metrics module
    if GRAPH_METRICS_AVAILABLE:
        try:
            avg_degrees = [graph_metrics.average_degree(g) for g in graphs]
            results["avg_degree"] = sum(avg_degrees) / len(avg_degrees)

            avg_degree_centralities = [graph_metrics.average_degree_centrality(g) for g in graphs]
            results["avg_degree_centrality"] = sum(avg_degree_centralities) / len(avg_degree_centralities)

            avg_betweenness_centralities = [graph_metrics.average_betweenness_centrality(g) for g in graphs]
            results["avg_betweenness_centrality"] = sum(avg_betweenness_centralities) / len(
                avg_betweenness_centralities)

            # Clustering coefficients - may not be as relevant for directed graphs but included
            try:
                avg_clustering = [np.mean(list(nx.clustering(g).values())) for g in graphs]
                results["avg_clustering"] = sum(avg_clustering) / len(avg_clustering)
            except:
                results["avg_clustering"] = "Error computing clustering coefficients"
        except Exception as e:
            results["graph_metrics_error"] = str(e)

    # Orbit statistics if available
    # if ORBIT_AVAILABLE:
    #     try:
    #         def get_orbit_stats(graph_list):
    #             total_counts = []
    #             for graph in graph_list:
    #                 try:
    #                     # Make graph undirected for orbit calculation
    #                     undirected_g = graph.to_undirected()
    #                     orbit_counts = orbit_stats.orca(undirected_g)
    #                     orbit_counts_graph = np.sum(orbit_counts, axis=0) / undirected_g.number_of_nodes()
    #                     total_counts.append(orbit_counts_graph)
    #                 except Exception as e:
    #                     print(f"Error computing orbit stats: {e}")
    #             return total_counts
    #
    #         orbit_results = get_orbit_stats(graphs)
    #         if orbit_results:
    #             orbit_means = np.mean(orbit_results, axis=0)
    #             results["orbit_means"] = orbit_means.tolist()
    #     except Exception as e:
    #         results["orbit_stats_error"] = str(e)

    # MMD metrics (if reference graphs are provided)
    if MMD_AVAILABLE and test_graphs and len(test_graphs) > 0:
        try:
            # Configuration for MMD
            mmd_fn_is_hist = lambda x, y: mmd_stanford_impl.compute_mmd(
                x, y, kernel=mmd_stanford_impl.gaussian_emd, is_hist=True
            )
            mmd_fn_no_hist = lambda x, y: mmd_stanford_impl.compute_mmd(
                x, y, kernel=mmd_stanford_impl.gaussian_emd, is_hist=False
            )

            # Degree distribution MMD
            degree_dist1 = [nx.degree_histogram(g) for g in test_graphs]
            degree_dist2 = [nx.degree_histogram(g) for g in graphs]
            results["mmd_degree"] = mmd_fn_is_hist(degree_dist1, degree_dist2)

            # Try computing other MMD metrics
            try:
                # Clustering coefficient MMD
                cc_hist1 = [graph_metrics.get_histogram_of_clustering_coeffs(g) for g in test_graphs]
                cc_hist2 = [graph_metrics.get_histogram_of_clustering_coeffs(g) for g in graphs]
                results["mmd_clustering"] = mmd_fn_is_hist(cc_hist1, cc_hist2)
            except:
                results["mmd_clustering"] = "Error computing clustering MMD"

            try:
                # Betweenness centrality MMD
                bc_vals1 = [[c for c in nx.betweenness_centrality(g).values()] for g in test_graphs]
                bc_vals2 = [[c for c in nx.betweenness_centrality(g).values()] for g in graphs]
                results["mmd_betweenness"] = mmd_fn_no_hist(bc_vals1, bc_vals2)
            except:
                results["mmd_betweenness"] = "Error computing betweenness MMD"
        except Exception as e:
            results["mmd_error"] = str(e)

    return results

# In src/aig_evaluate.py

# Make sure these are imported if not already
from model import GraphLevelRNN, EdgeLevelRNN, EdgeLevelMLP
from aig_dataset import AIGDataset # Needed to determine max_node_count
import os # Needed for os.path.exists

def load_aig_model_from_config(model_path):
    """
    Load AIG-specific models from checkpoint, determining correct input size.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    try:
        state = torch.load(model_path, map_location=device)
        config = state['config']
    except Exception as e:
        raise RuntimeError(f"Error loading model state from {model_path}: {e}")

    # --- START FIX for input_size ---
    use_bfs = config['data'].get('use_bfs', True) # Default to True if missing

    if use_bfs:
        # Use 'm' only if BFS mode is specified
        input_size = config['data'].get('m') # Use .get() for safety
        if input_size is None:
             raise ValueError("Config key 'data.m' is required when use_bfs is true.")
        print(f"INFO: Loading model for BFS mode. Input size (m): {input_size}")
    else:
        # Need max_node_count for TopSort. Determine from dataset.
        print("INFO: Config indicates TopSort mode. Determining max_node_count from dataset...")
        dataset_path = config['data'].get('graph_file')
        if not dataset_path or not os.path.exists(dataset_path):
             raise FileNotFoundError(f"Dataset file '{dataset_path}' specified in config not found.")
        try:
             # Load dataset temporarily just to get max_node_count
             # Note: This assumes AIGDataset can be initialized with m=None when training=False
             temp_dataset = AIGDataset(graph_file=dataset_path, m=None, training=False, use_bfs=False)
             max_node_count = temp_dataset.max_node_count
             if max_node_count <= 1: raise ValueError("Max node count from dataset <= 1")
             input_size = max_node_count - 1
             print(f"INFO: Loading model for TopSort mode. Input size (max_nodes-1): {input_size}")
        except Exception as e:
             raise RuntimeError(f"Could not determine max_node_count for TopSort mode from {dataset_path}: {e}")
    # --- END FIX for input_size ---

    # Assume edge_feature_len is correctly set in the config used for training
    edge_feature_len = config['model']['GraphRNN'].get('edge_feature_len', 3)
    if edge_feature_len != 3:
         print(f"Warning: Expected edge_feature_len=3 for AIGs, found {edge_feature_len} in config.")

    # Setup model args, forcing generation-specific settings
    node_model_args = config['model']['GraphRNN'].copy() # Use copy
    node_model_args['input_size'] = input_size # Use determined input_size
    node_model_args['edge_feature_len'] = edge_feature_len
    node_model_args['predict_node_types'] = False # Override for generation
    node_model_args['use_conditioning'] = False # Override for generation
    node_model_args['tt_size'] = None # Override for generation

    if config['model']['edge_model'] == 'rnn':
        edge_model_args = config['model']['EdgeRNN'].copy() # Use copy
        edge_model_args['edge_feature_len'] = edge_feature_len
        edge_model_args['use_conditioning'] = False # Override for generation
        edge_model_args['tt_size'] = None # Override for generation

        node_model = GraphLevelRNN(
            output_size=edge_model_args['hidden_size'], # RNN edge model needs output size
            **node_model_args # Pass modified args
        ).to(device)
        edge_model = EdgeLevelRNN(**edge_model_args).to(device) # Pass modified args
        # Import edge_gen_function from aig_generate
        from aig_generate import rnn_edge_gen
        edge_gen_function = rnn_edge_gen
        print("Using RNN edge model for generation.")
    else: # Assume MLP
        edge_model_args = config['model']['EdgeMLP'].copy() # Use copy
        edge_model_args['edge_feature_len'] = edge_feature_len
        edge_model_args['use_conditioning'] = False # Override for generation
        edge_model_args['tt_size'] = None # Override for generation
        # Set MLP output_size based on the *correct* input_size
        edge_model_args['output_size'] = input_size # Use determined input_size
        # Set input_size based on GraphRNN hidden size (this should be in config)
        edge_model_args['input_size'] = config['model']['GraphRNN']['hidden_size']


        node_model = GraphLevelRNN(
            output_size=None, # MLP edge model uses hidden state directly
            **node_model_args # Pass modified args
        ).to(device)
        edge_model = EdgeLevelMLP(**edge_model_args).to(device) # Pass modified args
        # Import edge_gen_function from aig_generate
        from aig_generate import mlp_edge_gen
        edge_gen_function = mlp_edge_gen
        print("Using MLP edge model for generation.")

    try:
        node_model.load_state_dict(state['node_model'])
        edge_model.load_state_dict(state['edge_model'])
        print("Model weights loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading model state dict: {e}")

    # Mode should be 'directed-multiclass' for AIGs
    mode = config['model'].get('mode', 'directed-multiclass')
    if mode != 'directed-multiclass':
        print(f"Warning: Expected mode 'directed-multiclass', found '{mode}' in config.")

    # Return necessary items for evaluation
    # Note: tt_size is returned as None because conditioning is forced off during generation
    #return node_model, edge_model, input_size, edge_gen_function, mode, None, config # Return config too
    return node_model, edge_model, input_size, edge_gen_function, mode, None, config, edge_feature_len


def evaluate_model(model_path, num_graphs=50, min_nodes=10, max_nodes=100,
                   output_dir='evaluation_results', use_conditioning=False,
                   predict_node_types=None, validity_threshold=0.0, seed=42,
                   test_dataset_path=None):
    """
    Generate and evaluate multiple AIGs using a trained model.

    Args:
        model_path: Path to the model checkpoint
        num_graphs: Number of graphs to generate
        min_nodes: Minimum number of nodes per graph
        max_nodes: Maximum number of nodes per graph
        output_dir: Directory to save results
        use_conditioning: Whether to use truth table conditioning
        predict_node_types: Whether the model predicts node types (overrides config if not None)
        validity_threshold: Minimum validity threshold (0-1)
        seed: Random seed for reproducibility
        test_dataset_path: Optional path to test dataset for comparison

    Returns:
        dict: Summary of evaluation results
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model_result = load_aig_model_from_config(model_path)
    if not model_result:
        return {"error": f"Failed to load model from {model_path}"}

    node_model, edge_model, input_size, edge_gen_function, mode, tt_size, config, edge_feature_len = model_result

    # Override config setting with explicit parameter if provided
    if predict_node_types is not None:
        config['model']['predict_node_types'] = predict_node_types
        print(f"Note: Overriding model config to set predict_node_types={predict_node_types}")

    # Use CUDA if available
    device = next(node_model.parameters()).device

    # Initialize statistics
    all_stats = []
    valid_count = 0
    total_count = 0

    # Load test dataset if provided
    test_graphs = []
    if test_dataset_path:
        try:
            print(f"Loading test dataset from {test_dataset_path}...")
            # --- START EDIT ---
            use_bfs_eval = config['data'].get('use_bfs', False)  # Get use_bfs from config, default False
            m_eval = config['data'].get('m') if use_bfs_eval else None  # Get m only if use_bfs is True

            test_dataset = AIGDataset(
                graph_file=test_dataset_path,
                m=m_eval,  # Pass m only if needed
                training=False,
                use_bfs=use_bfs_eval,  # Pass the correct use_bfs value
                include_node_types=config['model'].get('predict_node_types', False)  # Keep this
            )
            test_graphs = test_dataset.graphs[
                          test_dataset.start_idx: test_dataset.start_idx + test_dataset.length]
            print(f"Loaded {len(test_graphs)} test graphs for comparison (full test split)")
        except Exception as e:
            print(f"Error loading test dataset: {e}")

    # Prepare conditioning
    truth_table = None
    if use_conditioning and tt_size is not None:
        n_outputs = config.get('n_outputs', 2)
        n_inputs = config.get('n_inputs', 8)
        truth_table, _ = create_random_truth_table(n_outputs, n_inputs, device)
        print(f"Using random truth table for conditioning with shape: {truth_table.shape}")

    # Store generated NetworkX graphs
    generated_nx_graphs = []

    print(f"Generating {num_graphs} AIGs...")

    # Generate graphs
    for i in tqdm(range(num_graphs)):
        # Randomly choose number of nodes
        target_nodes = np.random.randint(min_nodes, max_nodes + 1)

        # Generate the AIG
        G = generate_aig_structure(  # Call the function from aig_generate.py
            num_nodes=target_nodes,
            node_model=node_model,
            edge_model=edge_model,
            input_size=input_size,  # Use the correctly loaded input_size
            edge_gen_function=edge_gen_function,
            mode=mode,
            edge_feature_len=edge_feature_len  # Add this missing argument
            # REMOVE incorrect config and truth_table arguments
        )

        # Skip if generation failed
        if G is None:  # Check if generate_aig_structure returned None
            continue

        total_count += 1

        # --- START FIX ---
        # We already have the graph G, no need to reconstruct
        # REMOVE: adj_matrix, node_types = result
        # REMOVE: G = nx.DiGraph() ... construction loop ...

        # Node types might not be generated by this function, pass None to validity check
        # unless your generate_aig_structure is modified to return them.
        node_types = None  # Assuming generate_aig_structure doesn't return types

        # Calculate validity directly on the generated graph G
        stats = calculate_aig_structure_validity(G, node_types=node_types)
        stats['graph_id'] = i
        # --- END FIX ---

        # Check if it meets threshold
        if stats['valid']:
            valid_count += 1
            generated_nx_graphs.append(G)  # Save the actual generated graph

            if valid_count <= 10:
                # Visualize using the function in aig_generate.py if preferred
                # Pass G directly if visualize_aig is adapted, or reconstruct adj_matrix if needed
                try:
                    # Option A: Use visualize_aig_structure from aig_generate
                    output_file = os.path.join(output_dir, f"valid_aig_{valid_count}.png")
                    visualize_aig_structure(G, output_file)  # Use the function from aig_generate

                    # Option B: Keep visualize_aig in aig_evaluate, but reconstruct matrix (less ideal)
                    # adj_matrix_vis = nx.to_numpy_array(G, nodelist=sorted(G.nodes())) # Example reconstruction
                    # visualize_aig(adj_matrix_vis, output_file, node_types=node_types)
                except Exception as vis_e:
                    print(f"Warning: Visualization failed for graph {i}: {vis_e}")

        all_stats.append(stats)

    # Calculate standard graph metrics on the generated graphs
    metric_results = {}
    if generated_nx_graphs:
        print("\nCalculating graph structure metrics...")
        try:
            metric_results = compute_graph_metrics(generated_nx_graphs, test_graphs)
        except Exception as e:
            print(f"Error computing graph metrics: {e}")
            metric_results["error"] = str(e)

    # Calculate overall statistics
    validity_rate = valid_count / total_count if total_count > 0 else 0
    in_degree_distribution = {k: sum(s['in_degree_counts'].get(k, 0) for s in all_stats) for k in [0, 1, 2, 'other']}
    out_degree_distribution = {k: sum(s['out_degree_counts'].get(k, 0) for s in all_stats) for k in [0, 'other']}

    avg_potential_inputs = sum(s['potential_inputs'] for s in all_stats) / len(all_stats) if all_stats else 0
    avg_potential_outputs = sum(s['potential_outputs'] for s in all_stats) / len(all_stats) if all_stats else 0
    avg_potential_ands = sum(s['potential_ands'] for s in all_stats) / len(all_stats) if all_stats else 0

    # Extract model info
    checkpoint_name = os.path.basename(model_path)
    try:
        checkpoint_step = int(checkpoint_name.split("checkpoint-")[1].split(".")[0])
    except:
        checkpoint_step = -1

    # Combine into summary
    summary = {
        'model_path': model_path,
        'checkpoint_step': checkpoint_step,
        'generated_graphs': total_count,
        'valid_graphs': valid_count,
        'validity_rate': validity_rate,
        'avg_nodes': sum(s['num_nodes'] for s in all_stats) / len(all_stats) if all_stats else 0,
        'avg_edges': sum(s['num_edges'] for s in all_stats) / len(all_stats) if all_stats else 0,
        'in_degree_distribution': in_degree_distribution,
        'out_degree_distribution': out_degree_distribution,
        'avg_potential_inputs': avg_potential_inputs,
        'avg_potential_outputs': avg_potential_outputs,
        'avg_potential_ands': avg_potential_ands,
        # Add advanced metrics
        'graph_metrics': metric_results
    }

    # Save all stats to JSON
    with open(os.path.join(output_dir, 'aig_evaluation_details.json'), 'w') as f:
        json.dump(all_stats, f, indent=2)

    # Save summary to JSON
    with open(os.path.join(output_dir, 'aig_evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Generated graphs: {total_count}")
    print(f"Valid graphs: {valid_count}")
    print(f"Validity rate: {validity_rate:.2%}")
    print(f"Average nodes: {summary['avg_nodes']:.2f}")
    print(f"Average edges: {summary['avg_edges']:.2f}")
    print("\nNode type distribution:")
    print(f"  Potential inputs (in-degree 0): {avg_potential_inputs:.2f}")
    print(f"  Potential AND gates (in-degree 2): {avg_potential_ands:.2f}")
    print(f"  Potential outputs (out-degree 0, in-degree 1): {avg_potential_outputs:.2f}")

    # Print graph metrics
    if metric_results:
        print("\n=== Graph Structure Metrics ===")
        for key, value in metric_results.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            elif isinstance(value, list) and len(value) < 5:
                print(f"{key}: {value}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate GraphRNN models for AIGs')
    parser.add_argument('--model_paths', nargs='+',
                        help='List of model checkpoint files to evaluate')
    parser.add_argument('--model_dir', type=str,
                        help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--num_graphs', type=int, default=500,
                        help='Number of graphs to generate per model')
    parser.add_argument('--min_nodes', type=int, default=10,
                        help='Minimum nodes per graph')
    parser.add_argument('--max_nodes', type=int, default=60,
                        help='Maximum nodes per graph')
    parser.add_argument('--condition', action='store_true',
                        help='Use truth table conditioning if model supports it')
    parser.add_argument('--node_types', action='store_true',
                        help='Whether the model predicts node types')
    parser.add_argument('--test_dataset', type=str, default=None,
                        help='Path to test dataset for comparison metrics')
    parser.add_argument('--specific_checkpoint', type=int, default=None,
                        help='Specific checkpoint step to evaluate (e.g., 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Get model checkpoints
    model_paths = []

    # Option 1: Directly specified model paths
    if args.model_paths:
        model_paths = args.model_paths

    # Option 2: Find all checkpoints in directory
    elif args.model_dir:
        if args.specific_checkpoint:
            # Look for specific checkpoint
            checkpoint_path = os.path.join(args.model_dir, f"checkpoint-{args.specific_checkpoint}.pth")
            if os.path.exists(checkpoint_path):
                model_paths = [checkpoint_path]
            else:
                print(f"Error: Specified checkpoint '{checkpoint_path}' not found.")
                return 1
        else:
            # Find all checkpoints
            model_paths = find_checkpoints_in_directory(args.model_dir)

    if not model_paths:
        print("Error: No model checkpoints specified or found.")
        return 1

    print(f"Found {len(model_paths)} model checkpoints to evaluate.")

    # Run evaluation
    evaluate_multiple_models(
        model_paths=model_paths,
        num_graphs=args.num_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        output_dir=args.output_dir,
        use_conditioning=args.condition,
        predict_node_types=args.node_types,
        seed=args.seed,
        test_dataset_path=args.test_dataset
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())  # !/usr/bin/env python