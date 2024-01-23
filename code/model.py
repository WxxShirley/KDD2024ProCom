import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, SAGEConv, global_add_pool, GINConv


def build_conv(conv_type: str):
    """Return basic Graph Convolutional Layer based on specific type `conv_type`"""
    if conv_type == "GCN":
        return GCNConv
    elif conv_type == "GIN":
        return lambda i, h: GINConv(
            nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h))
        )
    elif conv_type == "GAT":
        return GATConv
    elif conv_type == "TransformerConv":
        return TransformerConv
    elif conv_type == "SAGE":
        return SAGEConv
    else:
        raise KeyError("[Model] GNN_TYPE can only be GAT, GCN, SAGE, GIN, and TransformerConv")


class GNNEncoder(torch.nn.Module):
    r"""Graph Neural Networks for node/graph encoding, Customized Settings include Dimension, Layer, and GNN-Type"""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=2, gnn_type="GCN"):
        super().__init__()
        conv = build_conv(gnn_type)

        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act = torch.nn.LeakyReLU()

        self.pool = global_add_pool

        if n_layer < 1:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(n_layer))
        elif n_layer == 1:
            self.conv_layers = torch.nn.ModuleList([conv(input_dim, hidden_dim)])
        elif n_layer == 2:
            self.conv_layers = torch.nn.ModuleList([conv(input_dim, hidden_dim), conv(hidden_dim, output_dim)])
        else:
            layers = [conv(input_dim, hidden_dim)]
            for _ in range(n_layer - 2):
                layers.append(conv(hidden_dim, hidden_dim))
            layers.append(conv(hidden_dim, output_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

    def forward(self, x, edge_index, batch=None):
        for graph_conv in self.conv_layers[0:-1]:
            x = graph_conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, training=self.training)

        node_emb = self.conv_layers[-1](x, edge_index)

        if batch is None:
            # input is a whole graph, return all nodes' embeddings
            return node_emb

        # compute center node embedding
        device = batch.device
        ones = torch.ones_like(batch).to(device)
        nodes_per_graph = global_add_pool(ones, batch)
        #  `cum_num`: the index of each centric node (recall we assign them in the first position) in the whole batch
        cum_num = torch.cat((torch.LongTensor([0]).to(device), torch.cumsum(nodes_per_graph, dim=0)[:-1]))

        graph_emb = self.pool(node_emb, batch)

        # return both centric nodes' and subgraphs' embeddings
        return node_emb[cum_num], graph_emb


class PromptLinearNet(nn.Module):
    def __init__(self, hidden_dim, threshold=0.1) -> None:
        super().__init__()

        self.predictor = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1),
                                       nn.Sigmoid())
        self.threshold = threshold

    def forward(self, ego_nodes, central_nodes):
        pred_logits = self.predictor(torch.cat([ego_nodes, central_nodes], dim=1))
        return pred_logits

    def make_prediction(self, ego_nodes, central_nodes):
        pred_logits = self.predictor(torch.cat([ego_nodes, central_nodes], dim=1)).squeeze(1)

        pos = torch.where(pred_logits >= self.threshold, 1.0, 0.0)
        # print(pos)
        # print(pred_logits, pos)
        return pos.nonzero().t().squeeze(0).detach().cpu().numpy().tolist()
