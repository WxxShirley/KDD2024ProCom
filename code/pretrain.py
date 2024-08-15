import time
import math
import torch
import torch.nn as nn
import random
from model import GNNEncoder
import torch.optim as optim
import data
import numpy as np
from torch_geometric.utils import subgraph
from torch_geometric.data import Data, Batch


class PreTrain(nn.Module):
    def __init__(self, dataset, gnn_type="GCN", input_dim=None, hidden_dim=None, num_layers=2,
                 device=torch.device("cuda:0")):
        r"""
        Perform pre-train on graph neural networks
        :param gnn_type: GNN encoder type (TransformerConv, GCN, GAT)
        :param input_dim:
        :param hidden_dim:
        """
        super(PreTrain, self).__init__()

        self.dataset = dataset
        self.device = device

        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.num_layer = num_layers

        self.gnn = GNNEncoder(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              output_dim=hidden_dim,
                              n_layer=num_layers,
                              gnn_type=gnn_type)
        self.gnn.to(device)

    def generate_all_candidate_community_emb(self, model, graph_data, batch_size=128, k=2, max_size=20):
        model.eval()
        node_num = graph_data.x.size(0)

        node_list = np.arange(0, node_num, 1)
        z = torch.Tensor(node_num, self.hidden_dim).to(self.device)
        group_nb = math.ceil(node_num / batch_size)

        for i in range(group_nb):
            maxx = min(node_num, (i + 1) * batch_size)
            minn = i * batch_size

            batch, _ = data.prepare_pretrain_data(node_list[minn:maxx].tolist(), data=graph_data, max_size=max_size,
                                                  num_hop=k)
            batch = batch.to(self.device)
            _, comms_emb = model(batch.x, batch.edge_index, batch.batch)
            z[minn:maxx] = comms_emb
            print(f"***Generate nodes embedding from idx {minn} to {maxx}")
        return z

    def generate_all_node_emb(self, model, graph_data):
        model.eval()

        # return all nodes embedding
        node_emb = model(graph_data.x, graph_data.edge_index)
        return node_emb

    def generate_target_community_emb(self, model, comms, graph_data):
        batch = []
        num_nodes = graph_data.x.size(0)
        model.eval()
        for community in comms:
            edge_index, _ = subgraph(community, graph_data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
            node_x = graph_data.x[community]  # node features
            g_data = Data(x=node_x, edge_index=edge_index)
            batch.append(g_data)
        batch = Batch().from_data_list(batch).to(self.device)
        # print(batch)

        _, comms_emb = model(batch.x, batch.edge_index, batch.batch)
        del batch
        return comms_emb

    def contrastive_loss(self, x1, x2):
        r"""Compute contrastive InfoNCE loss"""
        # hyperparameter: temperature
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        loss = - torch.log(loss).mean()
        return loss

    def train_subg(self, model, batch, optimizer, corrupt_batch=None, node_scale=1, subg_scale=0.1):
        model.train()
        batch = batch.to(self.device)

        z, summary = model(batch.x, batch.edge_index, batch.batch)
        optimizer.zero_grad()

        if node_scale:
            loss = self.contrastive_loss(z, summary)
        else:
            loss = 0.0

        if subg_scale and corrupt_batch:
            corrupt_batch = corrupt_batch.to(self.device)
            _, corrupt_summary = model(corrupt_batch.x, corrupt_batch.edge_index, corrupt_batch.batch)

            subg_loss = self.contrastive_loss(summary, corrupt_summary)

            loss += subg_scale * subg_loss
        loss.backward()
        optimizer.step()

        return float(loss.detach().cpu().item())

    def train(self, graph_data, batch_size=128, lr=1e-3, decay=0.00001, epochs=100, subg_max_size=20, num_hop=1,
              node_scale=1, subg_scale=0.1):
        optimizer = optim.Adam(self.gnn.parameters(), lr=lr, weight_decay=decay)

        num_nodes = graph_data.x.size(0)

        for epoch in range(1, epochs + 1):
            st = time.time()
            node_list = random.sample(range(num_nodes), batch_size)
            # Prepare data
            batch_data, corrupt_batch_data = data.prepare_pretrain_data(node_list, data=graph_data,
                                                                        max_size=subg_max_size, num_hop=num_hop,
                                                                        corrupt=subg_scale)

            train_loss = self.train_subg(self.gnn, batch_data, optimizer, corrupt_batch=corrupt_batch_data,
                                         node_scale=node_scale, subg_scale=subg_scale)
            print(
                "***epoch: {:04d} | train_loss: {:.5f} | cost time {:.3}s".format(epoch, train_loss, time.time() - st))
