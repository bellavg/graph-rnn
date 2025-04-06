import os
import pickle
import networkx as nx
from aigverse import read_aiger_into_aig, to_edge_list, simulate, simulate_nodes
import numpy as np
from typing import List, Tuple, Dict, Any

MAX_SIZE = 120
MAX_INPUTS = 8
MAX_TT_LENGTH = 2 ** MAX_INPUTS  # Maximum truth table length

# Directory where all AIG folders are stored
base_dir = './aiger'

# Define one-hot encodings for node types and edge labels
node_type_encoding = {
    "0": [0, 0, 0],
    "PI": [1, 0, 0],  # [One-hot encoding]
    "AND": [0, 1, 0],
    "PO": [0, 0, 1]
}

edge_label_encoding = {
    "INV": [1, 0],  # Inverted edge
    "REG": [0, 1]  # Regular edge
}


def save_all_graphs(all_graphs: List[nx.DiGraph], output_file: str) -> None:
    """
    Save all graphs to a single pickle file.

    Args:
        all_graphs: List of DiGraph objects to save
        output_file: Path to the output file
    """
    with open(output_file, "wb") as f:
        pickle.dump(all_graphs, f)
    print(f"Saved {len(all_graphs)} graphs to {output_file}")


def generate_binary_inputs(num_inputs: int) -> List[List[int]]:
    """
    Generates all possible binary input combinations for a given number of inputs.

    Args:
        num_inputs: Number of input variables

    Returns:
        List of all possible binary input combinations
    """
    return [[(i >> bit) & 1 for bit in range(num_inputs - 1, -1, -1)]
            for i in range(2 ** num_inputs)]


def get_padded_truth_table(tt_binary: str) -> List[int]:
    """
    Convert binary truth table string to list and pad with -1 to max length.

    Args:
        tt_binary: Binary string representation of truth table

    Returns:
        Padded truth table as a list of integers
    """
    # Convert binary string to list of integers
    tt_list = [int(bit) for bit in tt_binary]

    # Pad with -1 to ensure consistent length
    padding = [-1] * (MAX_TT_LENGTH - len(tt_list))
    return tt_list + padding


def get_nodes(aig: Any, G: nx.DiGraph, pad: bool = True) -> nx.DiGraph:
    input_patterns = list(zip(*generate_binary_inputs(aig.num_pis())))

    def format_tt(binary_str):
        return get_padded_truth_table(binary_str) if pad else [int(b) for b in binary_str]

    zero_tt = format_tt("0" * (2 ** aig.num_pis()))
    G.add_node(0, type=node_type_encoding["0"], feature=zero_tt)

    for pi in aig.pis():
        binary_inputs = "".join(map(str, list(input_patterns[pi - 1])))
        G.add_node(pi, type=node_type_encoding["PI"], feature=format_tt(binary_inputs))

    n_to_tt = simulate_nodes(aig)
    for gate in aig.gates():
        binary_truths = n_to_tt[gate].to_binary()
        G.add_node(gate, type=node_type_encoding["AND"], feature=format_tt(binary_truths))

    return G


def get_edges(aig: Any, G: nx.DiGraph) -> nx.DiGraph:
    """
    Add edges to the graph with one-hot encoded edge labels.

    Args:
        aig: AIG object
        G: NetworkX DiGraph

    Returns:
        Updated DiGraph with edges added
    """
    edges = to_edge_list(aig, inverted_weight=1, regular_weight=0)
    for e in edges:
        # Assign one-hot encoded edge labels
        onehot_label = np.array(
            edge_label_encoding["INV"] if e.weight == 1 else edge_label_encoding["REG"],
            dtype=np.float32
        )
        G.add_edge(e.source, e.target, type=onehot_label)
    return G


def get_outs(aig: Any, G: nx.DiGraph, size: int) -> nx.DiGraph:
    """
    Add output nodes and edges to the graph with one-hot encoded output nodes.

    Args:
        aig: AIG object
        G: NetworkX DiGraph
        size: Current size of the graph (number of nodes)

    Returns:
        Updated DiGraph with output nodes and edges added
    """
    tts = simulate(aig)

    for ind, po in enumerate(aig.pos()):
        binary_truths = tts[ind].to_binary()

        # Get out node
        new_out_node_id = size + ind
        G.add_node(new_out_node_id,
                   type=node_type_encoding["PO"],
                   feature=get_padded_truth_table(binary_truths))

        # Get out edge
        onehot_label = np.array(
            edge_label_encoding["INV"] if aig.is_complemented(po) else edge_label_encoding["REG"],
            dtype=np.float32
        )
        pre_node = aig.get_node(po)
        G.add_edge(pre_node, new_out_node_id, type=onehot_label)

    return G


def get_condition(aig: Any, graph_size: int, pad: bool = True) -> Tuple[List[List[int]], List[List[int]]]:
    def format_tt(binary_str):
        return get_padded_truth_table(binary_str) if pad else [int(b) for b in binary_str]

    zero_tt = format_tt("0" * (2 ** aig.num_pis()))
    condition_list = [zero_tt]
    full_condition_list = [zero_tt]

    input_patterns = list(zip(*generate_binary_inputs(aig.num_pis())))

    for pi in aig.pis():
        binary_inputs = "".join(map(str, list(input_patterns[pi - 1])))
        tt = format_tt(binary_inputs)
        condition_list.append(tt)
        full_condition_list.append(tt)

    condition_list += [[]] * aig.num_gates()
    n_to_tt = simulate_nodes(aig)

    for gate in aig.gates():
        binary_t = n_to_tt[gate].to_binary()
        tt = format_tt(binary_t)
        full_condition_list.append(tt)

    tts = simulate(aig)
    for ind, _ in enumerate(aig.pos()):
        binary_truths = tts[ind].to_binary()
        tt = format_tt(binary_truths)
        condition_list.append(tt)
        full_condition_list.append(tt)

    assert len(condition_list) == graph_size
    return condition_list, full_condition_list



def get_graph(aig: Any, graph_size: int, pad: bool = True) -> nx.DiGraph:
    condition, full_condition = get_condition(aig, graph_size, pad=pad)

    G = nx.DiGraph(
        inputs=aig.num_pis(),
        outputs=aig.num_pos(),
        tts=condition,
        full_tts=full_condition,
        output_tts=[[int(tt.get_bit(i)) for i in range(tt.num_bits())] for tt in simulate(aig)]
    )

    G = get_nodes(aig, G, pad=pad)
    G = get_edges(aig, G)
    G = get_outs(aig, G, aig.size())

    pos = aig.num_pos()
    assert G.number_of_nodes() == aig.size() + pos, f"Node count mismatch"

    return G


def main():
    """Main function to process filtered AIG files and create graph dataset."""
    all_graphs = []

    for filename in os.listdir(base_dir):
        if not filename.endswith('.aig'):
            continue

        file_path = os.path.join(base_dir, filename)

        try:
            aig = read_aiger_into_aig(file_path)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        # Only allow AIGs with 8 inputs and less than 8 outputs
        if aig.num_pis() != 8 and aig.num_pos() <=8:
            continue

        graph_size = aig.num_pis() + aig.num_pos() + aig.num_gates() + 1
        Graph = get_graph(aig, graph_size, pad=False)
        all_graphs.append(Graph)


    print("Filtered Dataset Size:", len(all_graphs))
    save_all_graphs(all_graphs, "./inputs8_outputs8less.pkl")


if __name__ == "__main__":
    main()