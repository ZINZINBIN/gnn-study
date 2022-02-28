import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, EdgeConv, ChebConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.nn.pool.graclus import graclus
from torch_scatter import scatter_max
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.nn import max_pool, max_pool_x, avg_pool
from torch_geometric.utils.undirected import to_undirected
from typing import Union, Tuple, Optional
from pytorch_model_summary import summary
from src.layer import *

atom_feats = [7, 5, 4, 4, 2, 2, 4, 3, 8]
mol_feats = 22

class FeatureEmbedding(nn.Module):
    def __init__(self, feature_lens, max_norm = 1.0):
        super(FeatureEmbedding, self).__init__()
        self.feature_lens = feature_lens
        self.emb_layers = nn.ModuleList()
        self.max_norm = max_norm

        for size in feature_lens[:-1]:
            emb_layer = nn.Embedding(size, size, max_norm = max_norm)
            emb_layer.load_state_dict({'weight': torch.eye(size)})
            self.emb_layers.append(emb_layer)

    def forward(self, x):
        output = []
        for i, layer in enumerate(self.emb_layers):
            output.append(layer(x[:, i].long()))

        output.append(x[:, -self.feature_lens[-1]:])
        output = torch.cat(output, 1)
        return output

class GConvNet(nn.Module):
    def __init__(self, hidden, alpha = 0.01, embedd_max_norm = 1.0):
        super(GConvNet, self).__init__()
        torch.manual_seed(42)
        self.hidden = hidden
        self.alpha = alpha
        self.embedd_max_norm = embedd_max_norm

        self.embedd = FeatureEmbedding(feature_lens = atom_feats, max_norm=embedd_max_norm)
        self.init_graph = GCNLayer(sum(atom_feats), hidden, alpha)
        self.head = GCNLayer(hidden, hidden, alpha)
        self.body = GCNLayer(hidden, hidden, alpha)
        self.tail = GCNLayer(hidden, hidden, alpha)

        self.mlp = nn.ModuleList(
            [
                nn.Linear(4 * self.hidden, self.hidden),
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, 1),
            ]
        )
       
    def forward(self, inputs):
        x, edge_idx, batch_idx = inputs.x,  inputs.edge_index, inputs.batch
        x = self.embedd(x)
        x = self.init_graph(x, edge_idx)
        x1 = self.head(x, edge_idx)
        x2 = self.body(x1, edge_idx)
        x3 = torch.add(x, x2)
        x = self.tail(x3, edge_idx)
        x = torch.cat((x,x1,x2,x3), dim = 1)
        x = global_max_pool(x, batch_idx)
        x = torch.flatten(x, start_dim = 1)
        for layer in self.mlp:
            x = layer(x)
        return x.squeeze(1)

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))
        
class HGNN(nn.Module):
    def __init__(self, num_heads : int = 4, hidden : int = 128, p = 0.5, alpha = 0.01, embedd_max_norm = 1.0):
        super(HGNN, self).__init__()
        self.num_heads = num_heads
        self.hidden = hidden
        self.p = p
        self.alpha = alpha

        self.embedd_max_norm = embedd_max_norm

        self.embedd = FeatureEmbedding(
            feature_lens=atom_feats, max_norm=embedd_max_norm)

        self.init_graph = GATLayer(sum(atom_feats), hidden, num_heads, alpha, p)
        self.head = GATLayer(hidden*num_heads, hidden, num_heads, alpha=alpha, p = p)
        self.body = GATLayer(hidden*num_heads, hidden, num_heads, alpha=alpha, p = p)
        self.tail = GATLayer(hidden*num_heads, hidden, num_heads, alpha=alpha, p = p)

        self.mlp = nn.ModuleList(
            [
                nn.Linear(4 * self.hidden * self.num_heads, self.hidden * self.num_heads),
                nn.BatchNorm1d(self.hidden * self.num_heads),
                nn.ReLU(),
                nn.Linear(self.hidden * self.num_heads, 1),
            ]
        )

    def forward(self, inputs):
        x, edge_idx, batch_idx, edge_attr = inputs.x,  inputs.edge_index, inputs.batch, inputs.edge_attr
        edge_attr = None
        x = self.embedd(x)
        x = self.init_graph(x, edge_idx, edge_attr)
        x1 = self.head(x, edge_idx, edge_attr)
        x2 = self.body(x1, edge_idx, edge_attr)
        x3 = torch.add(x, x2)
        x = self.tail(x3, edge_idx, edge_attr)
        x = torch.cat((x,x1,x2,x3), dim = 1)
        x = global_max_pool(x, batch_idx)
        x = torch.flatten(x, start_dim = 1)
        for layer in self.mlp:
            x = layer(x)
        return x.squeeze(1)

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))


class ChebNet(nn.Module):
    def __init__(self, k : int, hidden : int, alpha = 0.01, embedd_max_norm = 1.0, p = 0.5):
        super(ChebNet, self).__init__()
        self.k = k
        self.hidden = hidden
        self.alpha = alpha
        self.p = p

        self.embedd_max_norm = embedd_max_norm

        self.embedd = FeatureEmbedding(
            feature_lens=atom_feats, max_norm=embedd_max_norm)

        self.init_graph = ChebConvLayer(sum(atom_feats), k, hidden, alpha)
        self.head = ChebConvLayer(hidden, k, hidden, alpha)
        self.body = ChebConvLayer(hidden, k, hidden, alpha)
        self.tail = ChebConvLayer(hidden, k, hidden, alpha)

        self.dropout_head = nn.Dropout(p)
        self.dropout_tail = nn.Dropout(p)

        self.mlp = nn.ModuleList(
            [
                nn.Linear(4 * self.hidden, self.hidden),
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, 1),
            ]
        )
    
    def forward(self, inputs):
        x, edge_idx, batch_idx = inputs.x,  inputs.edge_index, inputs.batch
        x = self.embedd(x)
        x = self.init_graph(x, edge_idx)
        x1 = self.head(x, edge_idx)
        x1 = self.dropout_head(x1)
        x2 = self.body(x1, edge_idx)
        x3 = torch.add(x, x2)
        x = self.tail(x3, edge_idx)
        x = self.dropout_tail(x)
        x = torch.cat((x,x1,x2,x3), dim = 1)
        x = global_max_pool(x, batch_idx)
        x = torch.flatten(x, start_dim = 1)
        for layer in self.mlp:
            x = layer(x)
        return x.squeeze(1)

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))


class GMMNet(nn.Module):
    def __init__(self, dim : int, hidden : int, kernel_size : int, separate_gaussians : bool = False, aggr : str = 'mean', alpha = 0.01, p = 0.5, embedd_max_norm = 1.0):
        super(GMMNet, self).__init__()
        self.dim = dim
        self.hidden = hidden
        self.p = p
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.separate_gaussians = separate_gaussians
        self.embedd_max_norm = embedd_max_norm

        self.dropout_head = nn.Dropout(p)
        self.dropout_tail = nn.Dropout(p)

        self.embedd = FeatureEmbedding(feature_lens=atom_feats, max_norm=embedd_max_norm)

        self.init_graph = GMMConvLayer(sum(atom_feats), dim, hidden, kernel_size, separate_gaussians, alpha, aggr)
        self.head = GMMConvLayer(hidden, dim, hidden, kernel_size, separate_gaussians, alpha, aggr)
        self.body = GMMConvLayer(hidden, dim, hidden, kernel_size, separate_gaussians, alpha, aggr)
        self.tail = GMMConvLayer(hidden, dim, hidden, kernel_size, separate_gaussians, alpha, aggr)

        self.mlp = nn.ModuleList(
            [
                nn.Linear(4 * self.hidden, self.hidden),
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, 1),
            ]
        )
    
    def forward(self, inputs):
        x, edge_idx, batch_idx, edge_attr = inputs.x,  inputs.edge_index, inputs.batch, inputs.edge_attr
        x = self.embedd(x)
        x = self.init_graph(x, edge_idx, edge_attr)
        x1 = self.head(x, edge_idx, edge_attr)
        x1 = self.dropout_head(x1)
        x2 = self.body(x1, edge_idx, edge_attr)
        x3 = torch.add(x, x2)
        x = self.tail(x3, edge_idx, edge_attr)
        x = self.dropout_tail(x)
        x = torch.cat((x,x1,x2,x3), dim = 1)
        x = global_max_pool(x, batch_idx)
        x = torch.flatten(x, start_dim = 1)
        for layer in self.mlp:
            x = layer(x)
        return x.squeeze(1)

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))


# C-SGEL : Molecule Property Prediction Based on Spatial Graph Embedding(https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.9b00410?rand=oin4mnup)
# convolutional spatial graph embedding network : learn features from molecular graphs