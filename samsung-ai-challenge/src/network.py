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


class Network(nn.Module):
    def __init__(self, model="GCN", *args, **kwargs):
        super(Network, self).__init__()
        self.model = model
        self.args = args
        self.atom_feat_emb = FeatureEmbedding(feature_lens=atom_feats, max_norm=args["embedd_max_norm"])

        self.init_graph = self.build_layer(
            model=model, 
            in_features=sum(atom_feats), 
            out_features = args["hidden"],
            alpha = args["alpha"],
            **kwargs
        )

    def forward(self, inputs):
        x, edge_idx, batch_idx, edge_attr = inputs.x,  inputs.edge_index, inputs.batch, inputs.edge_attr
        x = self.atom_feat_emb(x)
        x = self.init_graph(x, edge_idx)
        
        return x
                
    def build_layer(self, model="GCN", *args):
        if model == "GCN":
            return GCNLayer(
                in_features = args["in_features"],
                out_features = args["out_features"],
                alpha = args["alpha"]
            )
        elif model == "GAT":
            return GATLayer(
                num_features=args["in_features"],
                hidden = args["out_features"],
                num_head = args["num_head"],
                alpha = args["alpha"],
                p = args["p"]
            )

        elif model == "Chebnet":
            return ChebConvLayer(
                n_dims_in = args["in_features"],
                k = args["k"],
                n_dims_out=args["out_features"],
                alpha = args["alpha"]
            )
        
        elif model == "GMM":
            return GMMConvLayer(
                n_dims_in = args["in_features"],
                dim = args["dim"],
                n_dims_out = args["out_features"],
                kernel_size = args["kernel_size"],
                separate_gaussians=False,
                alpha = args["alhpa"],
                aggr = args["str"]
            )
        elif model == "GraphConv":
            return GraphConvLayer(
                in_features = args["in_features"],
                out_features = args["out_features"],
                alpha = args["alpha"],
                aggr = args["aggr"]
            )
        else:
            return GCNLayer(
                in_features=args["in_features"],
                out_features=args["out_features"],
                alpha=args["alpha"]
            )

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


class GraphConvNet(nn.Module):
    def __init__(self, hidden, alpha=0.01, embedd_max_norm=1.0, aggr = "max"):
        super(GraphConvNet, self).__init__()
        torch.manual_seed(42)
        self.hidden = hidden
        self.alpha = alpha
        self.embedd_max_norm = embedd_max_norm
        self.aggr = aggr

        if aggr == "max":
            self.pooling = global_max_pool
        elif aggr == "mean":
            self.pooling = global_mean_pool
        else:
            self.pooling = global_add_pool

        self.embedd = FeatureEmbedding(
            feature_lens=atom_feats, max_norm=embedd_max_norm)
        self.init_graph = GraphConvLayer(sum(atom_feats), hidden, alpha)
        self.head = GraphConvLayer(hidden, hidden, alpha)
        self.body = GraphConvLayer(hidden, hidden, alpha)
        self.tail = GraphConvLayer(hidden, hidden, alpha)

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
        edge_attr = inputs.edge_attr

        x = self.embedd(x)
        x = self.init_graph(x, edge_idx, edge_attr)
        x1 = self.head(x, edge_idx, edge_attr)
        x2 = self.body(x1, edge_idx, edge_attr)
        x3 = torch.add(x, x2)
        x = self.tail(x3, edge_idx, edge_attr)
        x = torch.cat((x, x1, x2, x3), dim=1)
        x = self.pooling(x, batch_idx)
        x = torch.flatten(x, start_dim=1)
        for layer in self.mlp:
            x = layer(x)
        return x.squeeze(1)

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth=None,
              show_parent_layers=True, show_input=True))


# C-SGEL : Molecule Property Prediction Based on Spatial Graph Embedding(https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.9b00410?rand=oin4mnup)
# convolutional spatial graph embedding network : learn features from molecular graphs