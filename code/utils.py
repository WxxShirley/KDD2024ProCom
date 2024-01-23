import numpy as np
import random
import torch
import datetime
import time
import pytz
from torch_geometric.utils import k_hop_subgraph


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


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
