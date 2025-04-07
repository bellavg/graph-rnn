import networkx as nx
import matplotlib.pyplot as plt # <--- Add this import
from aig_dataset import *

def visualize_aig_structure(G, output_file='generated_aig_structure.png'):
    """Visualize the generated AIG structure, handling edge types."""
    if G is None or G.number_of_nodes() == 0:
        print(f"Skipping visualization for empty or None graph: {output_file}")
        return

    plt.figure(figsize=(14, 12)) # Increased size slightly
    try:
        # Try a layout that works well for DAGs
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except ImportError:
        print("Warning: pygraphviz not found. Using spring_layout for visualization (may be less clear for DAGs).")
        pos = nx.spring_layout(G, seed=42)
    except Exception as e:
        print(f"Warning: Layout failed ({e}). Using spring_layout.")
        pos = nx.spring_layout(G, seed=42)


    # Infer node types for coloring (optional, but nice)
    node_colors = []
    node_labels = {}
    inferred_types = G.graph.get('_inferred_types_cleaned', {}) # Get types if stored after cleaning
    if not inferred_types: # Fallback if not stored
         from evaluate import infer_node_types # Import if needed
         inferred_types = infer_node_types(G)

    for node in G.nodes():
        node_type = inferred_types.get(node, "UNKNOWN")
        color = 'lightgrey' # Default
        if node_type == "PI": color = 'lightgreen'
        elif node_type == "AND": color = 'lightblue'
        elif node_type == "PO": color = 'salmon'
        node_colors.append(color)
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
        plt.scatter([], [], s=100, color='lightgrey', label='Inferred Unknown/Other')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize='small')

    plt.title(f'Generated AIG Structure ({os.path.basename(output_file)})')
    plt.axis('off') # Turn off axis
    plt.tight_layout()
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        # print(f"Generated graph visualization saved to {output_file}")
    except Exception as e:
        print(f"Error saving visualization {output_file}: {e}")
    plt.close()
