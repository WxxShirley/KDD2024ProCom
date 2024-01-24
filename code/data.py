"""Load Raw Communities Data"""
from torch_geometric.data import Data, Batch
import networkx as nx
import numpy as np
from torch_geometric.utils import k_hop_subgraph, subgraph
import torch
import utils


def load_data(dataset_name):
    r"""Load SNAP Community datasets and return `edges`, `communities`, and `feat` (if exists)"""
    communties = open(f"../data/{dataset_name}/{dataset_name}-1.90.cmty.txt")
    edges = open(f"../data/{dataset_name}/{dataset_name}-1.90.ungraph.txt")

    communties = [[int(i) for i in x.split()] for x in communties]
    edges = [[int(i) for i in e.split()] for e in edges]
    # Remove self-loop edges
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]

    nodes = {node for e in edges for node in e}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}

    edges = [[mapping[u], mapping[v]] for u, v in edges]
    communties = [[mapping[node] for node in com] for com in communties]

    num_node, num_edges, num_comm = len(nodes), len(edges), len(communties)
    print(f"[{dataset_name.upper()}] #Nodes {num_node}, #Edges {num_edges}, #Communities {num_comm}")

    # TODO: Load node features
    node_feats = None

    return num_node, num_edges, num_comm, nodes, edges, communties, node_feats


def feature_augmentation(nodes, edges, num_node, normalize=True, feat_type='AUG'):
    r"""Node feature augmentation `[deg(u), min(deg(N)), max(deg(N)), mean(deg(N)), std(deg(N))]`"""
    if feat_type == "ONE":
        return np.ones([num_node, 1], dtype=np.float32)

    g = nx.Graph(edges)
    g.add_nodes_from(nodes)

    node_degree = [g.degree[node] for node in range(num_node)]

    feat_matrix = np.zeros([num_node, 5], dtype=np.float32)
    # feature 1 - node degree
    feat_matrix[:, 0] = np.array(node_degree).squeeze()

    for node in range(num_node):
        if len(list(g.neighbors(node))) > 0:
            # other features
            neighbor_deg = feat_matrix[list(g.neighbors(node)), 0]
            feat_matrix[node, 1:] = neighbor_deg.min(), neighbor_deg.max(), neighbor_deg.mean(), neighbor_deg.std()

    if normalize:
        feat_matrix = (feat_matrix - feat_matrix.mean(0, keepdims=True)) / (feat_matrix.std(0, keepdims=True) + 1e-9)
    return feat_matrix, g


def prepare_data(dataset="amazon"):
    r"""Core Functions: load community-dataset, return `int` statistics, `PyG` data, and `List` communities"""
    num_node, num_edge, num_community, nodes, edges, communities, features = load_data(dataset)
    if features is None:
        features, nx_graph = feature_augmentation(nodes, edges, num_node)
    else:
        nx_graph = nx.Graph(edges)
        nx_graph.add_nodes_from(nodes)

    # this is important to convert into undirected graph
    converted_edges = [[v, u] for u, v in edges]
    graph_data = Data(x=torch.FloatTensor(features), edge_index=torch.LongTensor(np.array(edges+converted_edges)).transpose(0, 1))

    return num_node, num_edge, num_community, graph_data, nx_graph, communities


def prepare_pretrain_data(node_list, data: Data, max_size=25, num_hop=2, corrupt=0):
    r"""Prepare pre-training data
    node_list: nodes for extracting subgraphs
    data: `PyG.Data` format graph (network) data
    max_size: maximum subgraph size
    """
    batch, corrupt_batch = [], []

    num_nodes = data.x.size(0)

    for node in node_list:
        node_set, _, _, _ = k_hop_subgraph(node_idx=node, num_hops=num_hop, edge_index=data.edge_index, num_nodes=num_nodes)

        if len(node_set) > max_size:
            node_set = node_set[torch.randperm(node_set.shape[0])][:max_size]
            node_set = torch.unique(torch.cat([torch.LongTensor([node]), torch.flatten(node_set)]))

        node_list = node_set.detach().cpu().numpy().tolist()
        seed_idx = node_list.index(node)

        if seed_idx != 0:
            node_list[seed_idx], node_list[0] = node_list[0], node_list[seed_idx]

        # Hint: important!!!
        #  We must ensure all the first node is the target node
        assert node_list[0] == node
        # print(node, node_list)

        edge_index, _ = subgraph(node_list, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
        node_x = data.x[node_list] # node features
        g_data = Data(x=node_x, edge_index=edge_index)
        batch.append(g_data)

        if corrupt:
            corrupt_data = utils.generate_corrupt_graph_view(g_data)
            corrupt_batch.append(corrupt_data)

    batch = Batch().from_data_list(batch)
    if corrupt:
        corrupt_batch = Batch().from_data_list(corrupt_batch)

    return batch, corrupt_batch
