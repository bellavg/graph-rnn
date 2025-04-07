from aig_evaluate import calculate_paper_validity, calculate_extensive_validity, infer_node_types
# evaluate_aig_generation.py
# Generates AIGs using trained models and evaluates their validity and structure.

import argparse
import os
import glob
import torch
import networkx as nx
import re
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt # For visualization function

# --- Imports from project structure ---
# Ensure these paths are correct relative to where you run the script,
# or that the 'src' directory is in your PYTHONPATH.
try:
    from model import * # Import all model classes for setup_models
    from utils import setup_models
    # Import constants and helpers needed here and potentially by generate.py
    from aig_dataset import _calculate_levels, NUM_EDGE_FEATURES, EDGE_TYPES
    # Import generation functions
    from aig_generate import generate_aig, load_model_and_config, mlp_edge_gen_aig, rnn_edge_gen_aig
    # Import AIG-specific validity functions from the new file
    from aig_evaluate import calculate_paper_validity, calculate_extensive_validity, infer_node_types
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure evaluate_aig_generation.py is run from a location where")
    print("'model', 'utils', 'aig_dataset', 'generate', 'aig_evaluate' are importable.")
    exit(1)

# --- Visualization Function (Included here as requested) ---
def visualize_aig_structure(G, output_file='generated_aig_structure.png'):
    """Visualize the generated AIG structure, handling edge types."""
    if G is None or G.number_of_nodes() == 0:
        print(f"Skipping visualization for empty or None graph: {output_file}")
        return

    plt.figure(figsize=(14, 12))
    pos = None
    try:
        # Try a layout that works well for DAGs if pygraphviz is available
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except ImportError:
        print("Warning: pygraphviz not found. Using spring_layout (may be less clear). Install pygraphviz for better DAG layout.")
        pos = nx.spring_layout(G, seed=42)
    except Exception as e:
        print(f"Warning: graphviz layout failed ({e}). Using spring_layout.")
        pos = nx.spring_layout(G, seed=42) # Fallback

    # Infer node types for coloring (using the imported function)
    node_colors = []
    node_labels = {}
    # Use types stored during analysis if available, otherwise infer
    inferred_types = G.graph.get('_inferred_types_cleaned', infer_node_types(G))

    for node in G.nodes():
        node_type = inferred_types.get(node, "UNKNOWN")
        color_map = {"PI": 'lightgreen', "AND": 'lightblue', "PO": 'salmon', "UNKNOWN": 'lightgrey', "INVALID_FANIN": 'orange'}
        node_colors.append(color_map.get(node_type, 'lightgrey'))
        node_labels[node] = f"{node}\n({node_type})" # Label with node ID and type

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.9)

    # Draw edges with styles based on 'type' attribute
    regular_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == EDGE_TYPES["REGULAR"]]
    inverted_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == EDGE_TYPES["INVERTED"]]
    other_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') not in [EDGE_TYPES["REGULAR"], EDGE_TYPES["INVERTED"]]]

    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=1.5, edge_color='black', style='solid', arrowsize=20, node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=inverted_edges, width=1.5, edge_color='red', style='dashed', arrowsize=20, node_size=700)
    if other_edges:
        print(f"Warning: Found edges with unexpected types (not REGULAR/INVERTED) in {output_file}")
        nx.draw_networkx_edges(G, pos, edgelist=other_edges, width=1.0, edge_color='gray', style='dotted', arrowsize=20, node_size=700)

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    # Add legend for edge types and node colors
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=1.5, linestyle='solid', label=f'Regular Edge (type {EDGE_TYPES["REGULAR"]})'),
        plt.Line2D([0], [0], color='red', lw=1.5, linestyle='dashed', label=f'Inverted Edge (type {EDGE_TYPES["INVERTED"]})'),
        plt.scatter([], [], s=100, color='lightgreen', label='Inferred PI'),
        plt.scatter([], [], s=100, color='lightblue', label='Inferred AND'),
        plt.scatter([], [], s=100, color='salmon', label='Inferred PO'),
        plt.scatter([], [], s=100, color='orange', label='Invalid Fan-in'),
        plt.scatter([], [], s=100, color='lightgrey', label='Unknown/Other')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize='small') # Changed loc

    plt.title(f'Generated AIG Structure ({os.path.basename(output_file)})')
    plt.axis('off')
    plt.tight_layout()
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving visualization {output_file}: {e}")
    plt.close()

def get_checkpoint_step(filename):
    """Extracts the step number from a checkpoint filename."""
    match = re.search(r'checkpoint-(\d+)\.pth$', filename)
    if match:
        return int(match.group(1))
    return -1  # Return -1 or raise error if pattern doesn't match


# --- Main Script Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate and Evaluate AIGs from Trained Models")
    parser.add_argument('model_dir', type=str, help='Directory containing model checkpoints (.pth files)')
    parser.add_argument('-n', '--num_graphs', type=int, default=100, help='Number of AIGs to generate per model')
    parser.add_argument('--nodes_target', type=int, default=50, help='Target number of nodes for generated AIGs')
    parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature for generation (1.0=no change)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--output_csv', type=str, default='aig_evaluation_results.csv', help='Output CSV file name')
    parser.add_argument('--max_gen_steps', type=int, default=None, help='Optional max generation steps per graph (overrides default)')
    parser.add_argument('--patience', type=int, default=10, help='Patience for stopping if no real edges are added')

    # Visualization Args

    ### START: Add Checkpoint Limit Argument ###
    parser.add_argument('--num_checkpoints', type=int, default=None,
                        help='Evaluate only the latest N checkpoints (default: evaluate all found)')
    ### END: Add Checkpoint Limit Argument ###
    parser.add_argument('--save_plots', action='store_true', help='Save visualizations of the best valid generated graphs')
    parser.add_argument('--plot_dir', type=str, default='./aig_plots', help='Directory to save visualizations')
    parser.add_argument('--num_plots', type=int, default=5, help='Maximum number of best plots to save per model')
    parser.add_argument('--plot_sort_by', type=str, default='nodes', choices=['nodes', 'level'], help='Sort criteria for best plots ("nodes" or "level")')

    #Args potentially needed if info not in config (better to save in config)
    parser.add_argument('--force_max_nodes_train', type=int, default=None, help='Manually override max_node_count_train if not in config')
    parser.add_argument('--force_max_level_train', type=int, default=None, help='Manually override max_level_train if not in config')

    args = parser.parse_args()

    # --- Device Setup ---
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # --- Plot Directory ---
    if args.save_plots:
        os.makedirs(args.plot_dir, exist_ok=True)
        print(f"Will save up to {args.num_plots} best valid plots per model (sorted by {args.plot_sort_by}) to {args.plot_dir}")

    # --- Find Models ---
    # --- Find Models ---
    all_checkpoint_paths = sorted(glob.glob(os.path.join(args.model_dir, 'checkpoint-*.pth')))

    if not all_checkpoint_paths:
        print(f"Error: No 'checkpoint-*.pth' files found in {args.model_dir}")
        return 1

    ### START: Filter Checkpoints ###
    if args.num_checkpoints is not None and args.num_checkpoints > 0:
        # Sort by step number (descending) and take the top N
        print(f"Found {len(all_checkpoint_paths)} total checkpoints. Selecting latest {args.num_checkpoints}.")
        all_checkpoint_paths.sort(key=get_checkpoint_step, reverse=True)
        checkpoint_paths_to_evaluate = all_checkpoint_paths[:args.num_checkpoints]
        if not checkpoint_paths_to_evaluate:
            print("Warning: Filtering resulted in zero checkpoints to evaluate.")
    else:
        # Evaluate all found checkpoints if arg not provided or <= 0
        print(f"Found {len(all_checkpoint_paths)} total checkpoints. Evaluating all.")
        checkpoint_paths_to_evaluate = all_checkpoint_paths
    ### END: Filter Checkpoints ###

    if not checkpoint_paths_to_evaluate:
        print(f"No checkpoints selected for evaluation in {args.model_dir}")
        return 1

    print(f"Selected {len(checkpoint_paths_to_evaluate)} models/checkpoints to evaluate.")

    checkpoint_paths = checkpoint_paths_to_evaluate

    results_list = []

    # --- Model Loop ---
    for model_idx, model_path in enumerate(checkpoint_paths):
        model_basename = os.path.basename(model_path).replace(".pth", "")
        print(f"\n--- Evaluating Model [{model_idx+1}/{len(checkpoint_paths)}]: {model_basename} ---")
        start_time_model = time.time()

        # --- Load Model and Config ---
        try:
            model_state, loaded_config = load_model_and_config(model_path, device)
            print(f"  Config loaded successfully from checkpoint.")
        except (FileNotFoundError, ValueError, KeyError, Exception) as e:
            print(f"  Error loading model/config from {model_path}: {e}. Skipping.")
            continue

        # --- Determine Training Params (max_n, max_l) ---
        max_n_train = loaded_config.get('data', {}).get('max_node_count_train', None)
        max_l_train = loaded_config.get('data', {}).get('max_level_train', None)

        # Check if manual override is provided (optional)
        # if args.force_max_nodes_train: max_n_train = args.force_max_nodes_train
        # if args.force_max_level_train: max_l_train = args.force_max_level_train

        # --- Determine Training Params (max_n, max_l) ---
        ### START: Modify Logic ###
        max_n_train = loaded_config.get('data', {}).get('max_node_count_train', None)
        max_l_train = loaded_config.get('data', {}).get('max_level_train', None)

        # Apply overrides if provided and config values are missing
        config_source_msg = "from config"
        if max_n_train is None and args.force_max_nodes_train is not None:
            max_n_train = args.force_max_nodes_train
            config_source_msg += ", max_nodes overridden"
            print(f"  Using override max_nodes_train: {max_n_train}")
        if max_l_train is None and args.force_max_level_train is not None:
            max_l_train = args.force_max_level_train
            config_source_msg += ", max_level overridden"
            print(f"  Using override max_level_train: {max_l_train}")

        # Final check if values are determined
        if max_n_train is None or max_l_train is None:
            print(f"  Error: Could not determine max_node_count_train or max_level_train ({config_source_msg}).")
            print(f"  Config 'data': {loaded_config.get('data', {})}")
            print(f"  Override args: nodes={args.force_max_nodes_train}, level={args.force_max_level_train}")
            print("  Skipping this model.")
            continue  # Skip this model if values are still missing

        print(
            f"  Using training params for model setup: max_nodes={max_n_train}, max_level={max_l_train} ({config_source_msg})")
        ### END: Modify Logic ###

        # --- Setup Models ---
        try:
            # Pass the determined parameters to setup_models
            node_model_gen, edge_model_gen, _ = setup_models(loaded_config, device, max_n_train, max_l_train)
            print(f"  Node model type: {type(node_model_gen).__name__}")
            print(f"  Edge model type: {type(edge_model_gen).__name__}")
        except (ValueError, KeyError, TypeError, Exception) as e:
            print(f"  Error setting up models using config: {e}. Skipping.")
            continue

        # --- Load Weights ---
        try:
            node_model_gen.load_state_dict(model_state['node_model'])
            edge_model_gen.load_state_dict(model_state['edge_model'])
            print(f"  Model weights loaded successfully.")
        except KeyError as e:
            print(f"  Error: Checkpoint missing model state_dict key: {e}. Skipping.")
            continue
        except RuntimeError as e:
            print(f"  Error loading state_dict (model mismatch?): {e}. Skipping.")
            continue

        # --- Select Edge Function ---
        edge_model_type_config = loaded_config.get('model', {}).get('edge_model', 'mlp').lower()
        edge_func = None
        if edge_model_type_config == 'mlp':
            edge_func = mlp_edge_gen_aig
        # Check if the instantiated edge model class corresponds to RNN/LSTM types
        elif isinstance(edge_model_gen, (EdgeLevelRNN, EdgeLevelAttentionRNN, EdgeLevelLSTM, EdgeLevelAttentionLSTM)):
            edge_func = rnn_edge_gen_aig
        else:
            print(f"  Error: Could not determine edge function for model type '{type(edge_model_gen).__name__}' (config edge_model: '{edge_model_type_config}'). Skipping.")
            continue
        print(f"  Using edge generation function: {edge_func.__name__}")


        # --- Calculate effective_m ---
        # Ensure effective_m is at least 1
        effective_m_gen = max(1, max_n_train - 1)
        print(f"  Effective M for generation (max_n_train - 1): {effective_m_gen}")


        # --- Generation and Evaluation Loop for this Model ---
        all_graphs_data = [] # Stores dicts with graph data and results
        print(f"  Generating {args.num_graphs} graphs (target N={args.nodes_target}, temp={args.temp}, patience={args.patience})...")
        graphs_generated_count = 0
        with tqdm(total=args.num_graphs, desc=f"  Generating {model_basename}", leave=False) as pbar:
            for i in range(args.num_graphs):
                try:
                    gen_graph, gen_max_level = generate_aig(
                        num_nodes_target=args.nodes_target,
                        node_model=node_model_gen,
                        edge_model=edge_model_gen,
                        effective_m=effective_m_gen,
                        max_level_model=max_l_train, # Pass max level model trained with
                        edge_gen_fn=edge_func,
                        device=device,
                        temperature=args.temp,
                        max_steps=args.max_gen_steps,
                        eos_patience=args.patience
                    )
                    graphs_generated_count += 1
                except Exception as e:
                    print(f"\n  Error during generation for graph {i}: {e}")
                    # Optionally store error or skip
                    pbar.update(1)
                    continue # Skip analysis for this failed generation

                # --- Clean: Remove Isolated Nodes ---
                g_cleaned = gen_graph
                isolates = list(nx.isolates(gen_graph))
                if isolates:
                    nodes_to_keep = list(set(gen_graph.nodes()) - set(isolates))
                    g_cleaned = gen_graph.subgraph(nodes_to_keep).copy() if nodes_to_keep else nx.DiGraph()

                # --- Analyze Cleaned Graph ---
                cleaned_node_count = g_cleaned.number_of_nodes()
                cleaned_edge_count = g_cleaned.number_of_edges()

                # Calculate level, infer types, calculate validity on cleaned graph
                final_max_level_cleaned = -1
                inferred_types_cleaned = {}
                paper_val = 0.0
                extensive_val_details = {}
                is_extensively_valid = False

                if cleaned_node_count > 0:
                    try:
                        if nx.is_directed_acyclic_graph(g_cleaned):
                            _, final_max_level_cleaned = _calculate_levels(g_cleaned)
                        else: final_max_level_cleaned = -2 # Non-DAG

                        inferred_types_cleaned = infer_node_types(g_cleaned)
                        g_cleaned.graph['_inferred_types_cleaned'] = inferred_types_cleaned # Store for viz

                        # Calculate validity metrics using functions from aig_evaluate
                        paper_val = calculate_paper_validity(g_cleaned)
                        extensive_val_details = calculate_extensive_validity(g_cleaned, check_connectivity=True)
                        is_extensively_valid = extensive_val_details.get("overall_valid", False)

                    except Exception as e:
                         print(f"\n  Error during analysis for graph {i}: {e}")
                         final_max_level_cleaned = -3 # Analysis error
                         paper_val = np.nan # Mark metrics as NaN on error
                         is_extensively_valid = False


                all_graphs_data.append({
                    "graph": g_cleaned, # Store the cleaned graph object
                    "analysis_index": i,
                    "is_extensively_valid": is_extensively_valid,
                    "paper_validity": paper_val,
                    "node_count": cleaned_node_count,
                    "edge_count": cleaned_edge_count,
                    "max_level": final_max_level_cleaned, # Level of cleaned graph
                })
                pbar.update(1) # End of generation loop for one graph


        # --- Aggregate and Report Results for the Current Model ---
        num_analyzed = len(all_graphs_data)
        print(f"\n  Finished generation. Analyzing {num_analyzed} results...")

        if num_analyzed > 0:
            # Calculate overall averages (handle potential NaNs in paper validity)
            all_node_counts = [d["node_count"] for d in all_graphs_data]
            all_edge_counts = [d["edge_count"] for d in all_graphs_data]
            all_valid_levels = [d["max_level"] for d in all_graphs_data if d["max_level"] >= 0] # Levels for valid DAGs
            all_paper_validities = [d["paper_validity"] for d in all_graphs_data]
            all_extensive_valid = [d["is_extensively_valid"] for d in all_graphs_data]

            avg_nodes = np.mean(all_node_counts)
            avg_edges = np.mean(all_edge_counts)
            avg_max_level = np.mean(all_valid_levels) if all_valid_levels else 0.0
            max_max_level = np.max(all_valid_levels) if all_valid_levels else 0.0
            avg_paper_validity = np.nanmean(all_paper_validities) # Use nanmean
            percent_extensive_valid = np.mean(all_extensive_valid) * 100.0

            # Store aggregated results
            results_list.append({
                "model": model_basename,
                "num_generated": graphs_generated_count,
                "num_analyzed": num_analyzed,
                "target_nodes": args.nodes_target,
                "avg_nodes": f"{avg_nodes:.2f}",
                "avg_edges": f"{avg_edges:.2f}",
                "avg_paper_validity": f"{avg_paper_validity:.4f}",
                "percent_extensive_valid": f"{percent_extensive_valid:.2f}",
                "avg_max_level": f"{avg_max_level:.2f}",
                "max_max_level": f"{max_max_level:.0f}",
                "temperature": args.temp,
                "patience": args.patience,
            })

            # Print aggregated results
            print(f"    Avg Nodes (cleaned): {avg_nodes:.2f}")
            print(f"    Avg Edges (cleaned): {avg_edges:.2f}")
            print(f"    Avg Paper Validity: {avg_paper_validity:.4f}")
            print(f"    Extensively Valid: {percent_extensive_valid:.2f}% ({sum(all_extensive_valid)}/{num_analyzed})")
            print(f"    Avg Max Level (valid DAGs): {avg_max_level:.2f}")
            print(f"    Max Max Level (valid DAGs): {max_max_level}")

            # --- Select and Visualize Best Valid Graphs ---
            if args.save_plots and num_analyzed > 0:
                valid_graphs_data = [d for d in all_graphs_data if d["is_extensively_valid"]]
                print(f"  Found {len(valid_graphs_data)} extensively valid graphs for plotting.")

                if valid_graphs_data:
                    sort_key = "node_count" if args.plot_sort_by == "nodes" else "max_level"
                    valid_graphs_data_sorted = sorted(
                        valid_graphs_data,
                        key=lambda x: x.get(sort_key, 0),
                        reverse=True # Sort descending
                    )

                    num_plots_to_save = min(len(valid_graphs_data_sorted), args.num_plots)
                    print(f"  Saving plots for top {num_plots_to_save} valid graphs (sorted by {args.plot_sort_by})...")

                    for rank, graph_data in enumerate(valid_graphs_data_sorted[:num_plots_to_save]):
                        plot_filename = os.path.join(
                            args.plot_dir,
                            f"{model_basename}_valid_{args.plot_sort_by}_rank{rank+1}_idx{graph_data['analysis_index']}.png"
                        )
                        # Call the visualization function defined in this script
                        visualize_aig_structure(graph_data["graph"], plot_filename)
                    print(f"  Finished saving plots.")
                else:
                     print("  No extensively valid graphs found to plot.")


        else: # No graphs analyzed
            print("    No graphs were successfully generated or analyzed for this model.")
            # Add empty/error result row
            results_list.append({
                "model": model_basename, "num_generated": graphs_generated_count, "num_analyzed": 0,
                "target_nodes": args.nodes_target, "avg_nodes": "N/A", "avg_edges": "N/A",
                "avg_paper_validity": "N/A", "percent_extensive_valid": "N/A",
                "avg_max_level": "N/A", "max_max_level": "N/A",
                "temperature": args.temp, "patience": args.patience,
            })

        elapsed_time_model = time.time() - start_time_model
        print(f"  Model evaluation took {elapsed_time_model:.2f} seconds.")
        # --- End of loop for one model ---

    # --- Save Final Results ---
    if results_list:
        results_df = pd.DataFrame(results_list)
        # Define column order
        cols_order = [
             "model", "num_generated", "num_analyzed", "target_nodes", "temperature", "patience",
             "avg_nodes", "avg_edges", "avg_paper_validity", "percent_extensive_valid",
             "avg_max_level", "max_max_level"
        ]
        # Reorder/select columns, handling potential missing ones if errors occurred
        final_cols = [col for col in cols_order if col in results_df.columns]
        results_df = results_df[final_cols]

        try:
            results_df.to_csv(args.output_csv, index=False)
            print(f"\nEvaluation results saved to {args.output_csv}")
        except Exception as e:
            print(f"\nError saving results to CSV: {e}")
            print("\nResults DataFrame:")
            print(results_df.to_string())
    else:
        print("\nNo models were successfully evaluated.")

    return 0

if __name__ == "__main__":
    import sys
    # Set a higher recursion depth if deep graphs cause issues with NetworkX/layout (use with caution)
    # sys.setrecursionlimit(5000)
    sys.exit(main())