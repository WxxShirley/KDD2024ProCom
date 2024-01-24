import numpy as np
import random
import torch
import datetime
import time
import pytz
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


##########################################################
################# Corruption for Subgraph ################
##########################################################


def drop_nodes(graph_data, aug_ratio=0.1):
    r"""Contrastive model corruption: dropping nodes"""
    node_num, edge_num = graph_data.x.size(0), graph_data.edge_index.size(1)

    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)
    idx_drop = idx_perm[:drop_num]
    idx_non_drop = idx_perm[drop_num:]
    idx_non_drop.sort()

    idx_dict = {idx_non_drop[n]: n for n in list(range(idx_non_drop.shape[0]))}

    device = graph_data.edge_index.device

    edge_index = graph_data.edge_index.detach().cpu().numpy()

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                  (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]

    try:
        new_edge_index = torch.tensor(edge_index).transpose_(0, 1).to(device)
        new_x = graph_data.x[idx_non_drop]
        new_graph_data = Data(x=new_x, edge_index=new_edge_index)
    except:
        new_graph_data = graph_data
    return new_graph_data


def drop_edges(graph_data, aug_ratio=0.1):
    r"""Contrastive model corruption: permuting edges"""
    edge_num = graph_data.edge_index.size(1)
    permute_num = int(edge_num * aug_ratio)

    idx_delete = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
    new_edge_index = graph_data.edge_index[:, idx_delete]
    return Data(x=graph_data.x, edge_index=new_edge_index)


def generate_corrupt_graph_view(graph_data, aug_ratio=0.15):
    x = random.random()

    # we set drop_edge as default operation
    # 0.5 (for facebook)
    if x < 0.5:
        return drop_nodes(graph_data, aug_ratio)
    return drop_edges(graph_data, aug_ratio)


##########################################################
############## Prepare Data for Prompt Tuning ############
##########################################################


def generate_prompt_tuning_data(train_comm, graph_data, nx_graph, k=2):
    # step1, randomly pick a node from train_comm as `central node`
    degrees = [nx_graph.degree[node] for node in train_comm]
    sum_val = sum(degrees)
    degrees = [d / sum_val for d in degrees]

    central_node = np.random.choice(train_comm, 1, p=degrees).tolist()[0]

    # step2, generate central node's k-ego net
    k_ego_net, _, _, _ = k_hop_subgraph(central_node, num_hops=k, edge_index=graph_data.edge_index,
                                        num_nodes=graph_data.x.size(0))
    k_ego_net = k_ego_net.detach().cpu().numpy().tolist()

    # step3, generate labels (1 if node belongs to the community, otherwise 0)
    labels = [[int(node in train_comm)] for node in k_ego_net]

    if 0 not in labels:
        # (optional) if all 1 , we have to randomly sample nodes to constitue the negative samples
        # TODO: the question is, how do we choose suitable negative samples
        random_negatives = np.random.choice(graph_data.x.size(0), len(k_ego_net)).tolist()
        random_negatives = [node for node in random_negatives if node not in k_ego_net]  # remove positive samples
        k_ego_net += random_negatives
        labels += [[0]] * len(random_negatives)

    return [central_node] * len(k_ego_net), k_ego_net, torch.FloatTensor(labels)


def pred_community_analysis(pred_comms):
    lengths = [len(com) for com in pred_comms]
    avg_length = np.mean(np.array(lengths))
    print(f"Predicted communitys #{len(pred_comms)}, avg size {avg_length:.4f}")
