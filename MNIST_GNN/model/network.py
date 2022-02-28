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
from model.layer import *
from utility.utils import *
from typing import Union, Tuple, Optional
from pytorch_model_summary import summary

class Network(nn.Module):
    def __init__(self, num_features, embedd_size, num_classes, alpha = 0.01, p  = 0.5):
        super(Network, self).__init__()
        torch.manual_seed(42)
        self.num_features = num_features
        self.embedd_size = embedd_size
        self.p = p
        self.init_gconv = GCNLayer(num_features, embedd_size, p, alpha)
        self.gconv1 = GCNLayer(embedd_size, embedd_size, p, alpha)
        self.gconv2 = GCNLayer(embedd_size, embedd_size, p, alpha)
        self.gconv3 = GCNLayer(embedd_size, embedd_size, p, alpha)
        self.gconv4 = GCNLayer(embedd_size, embedd_size, p, alpha)
        self.linear = torch.nn.Linear(embedd_size * 2, num_classes)
        self.alpha = alpha
       
    def forward(self, inputs):
        x, edge_idx, batch_idx = inputs.x, inputs.edge_index, inputs.batch
        h = self.init_gconv(x, edge_idx)
        h = self.gconv1(h, edge_idx)
        h = self.gconv2(h, edge_idx)
        h = self.gconv3(h, edge_idx)
        h = self.gconv4(h, edge_idx)
        h = torch.cat(
            [global_max_pool(h, batch_idx), global_mean_pool(h,batch_idx)], dim = 1
        )
        outputs = self.linear(h)
        return outputs

class GCN(nn.Module):
    def __init__(self, num_features, embedd_size, num_classes, activation = "tanh", alpha = 0.01):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.num_features = num_features
        self.embedd_size = embedd_size
        self.init_gconv = GCNConv(num_features, embedd_size)
        self.gconv1 = GCNConv(embedd_size, embedd_size)
        self.gconv2 = GCNConv(embedd_size, embedd_size)
        self.gconv3 = GCNConv(embedd_size, embedd_size)
        self.linear = torch.nn.Linear(embedd_size * 2, num_classes)
        self.activation = activation
        self.alpha = alpha
        
        if self.activation == "tanh":
            self.act = F.tanh()
        else:
            self.act = torch.nn.LeakyReLU(alpha  = self.alpha)
       
    def forward(self, inputs):
        x, edge_idx, batch_idx = inputs.x, inputs.edge_index, inputs.batch
        h = self.init_gconv(x, edge_idx)
        h = self.act(h)
        h = self.gconv1(h, edge_idx)
        h = self.act(h)
        h = self.gconv2(h, edge_idx)
        h = self.act(h)
        h = self.gconv3(h, edge_idx)
        h = self.act(h)

        h = torch.cat(
            [global_max_pool(h, batch_idx), global_mean_pool(h,batch_idx)], dim = 1
        )
        outputs = self.linear(h)
        return outputs


class GATNetworkMNIST(nn.Module):
    def __init__(self, num_features, num_classes, num_heads = [2,2,2], embedd_size_per_layer = [128,128,128], mlp_layer = [128,128]):
        super(GATNetworkMNIST, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.layer_heads = [1] + num_heads
        self.embedd_size_per_layer = [num_features] + embedd_size_per_layer
        self.MLP_layer_sizes = [self.layer_heads[-1]*self.embedd_size_per_layer[-1]] + mlp_layer + [num_classes]
        
        self.GAT_layers = nn.ModuleList(
              [
                GATLayerMultiHead(d_in*heads_in,d_out,heads_out)
                for d_in,d_out,heads_in,heads_out in zip(
                    self.embedd_size_per_layer[:-1],
                    self.embedd_size_per_layer[1:],
                    self.layer_heads[:-1],
                    self.layer_heads[1:],
                )
              ]
        )
        self.MLP_layers = nn.ModuleList(
              [
                nn.Linear(d_in,d_out)
                for d_in,d_out in zip(self.MLP_layer_sizes[:-1],self.MLP_layer_sizes[1:])
              ]
        )

    def forward(self, inputs):
        x, edge_idx, adj_t, batch_idx = inputs.x, inputs.edge_index, inputs.adj_t, inputs.batch

        for l in self.GAT_layers:
            x = l(x,adj_t)

        for layer in zip(self.MLP_layers):
            x = torch.relu(layer(x))

        return x

# torch_geometric version
# version 1 : custom
class GAT(nn.Module):
    def __init__(self, num_classes : int, num_features : int, num_heads = [], hidden = [], p = 0.5, alpha = 0.01):
        super(GAT, self).__init__()
        self.num_heads = [1] + num_heads
        self.num_features = num_features
        self.hidden = [num_features] + hidden
        self.p = p
        self.alpha = alpha

        self.GAT_modules = nn.ModuleList(
            [
                GATLayer(h_in * n_head_in, h_out, n_head_out, alpha = alpha, p = p) 
                for h_in, n_head_in, h_out, n_head_out in zip(self.hidden[0:-1],self.num_heads[0:-1],self.hidden[1:],self.num_heads[1:])
            ]
        )
        self.linear = nn.Linear(2 * num_heads[-1] * hidden[-1], num_classes)

    def forward(self, inputs):
        x, edge_idx, batch_idx = inputs.x,  inputs.edge_index, inputs.batch

        for layer in self.GAT_modules:
            x = layer(x,edge_idx)

        x = torch.cat(
            [global_max_pool(x, batch_idx), global_mean_pool(x, batch_idx)], dim = 1
        )
        x = self.linear(x)
        return x

# version 2 : Superpixel Image Classification with Graph Attention Networks
# type : Region Adjacency Graph
class GAT_MNIST(nn.Module):
    def __init__(self, num_classes: int, num_features: int, n_head : int, hidden : Tuple[int, int, int], mlp_hidden = 128, p=0.5):
        super(GAT_MNIST, self).__init__()
        self.n_head = n_head
        self.num_classes = num_classes
        self.num_features = num_features
        self.hidden = hidden
        self.p = p
        self.mlp_hidden = mlp_hidden

        self.GAT_modules = nn.ModuleList(
            [
                GATConv(num_features, hidden[0], n_head, True, dropout = p),
                GATConv(hidden[0] * n_head, hidden[1], n_head, True, dropout = p),
                GATConv(hidden[1] * n_head, hidden[2], n_head, True, dropout = p),
            ]
        )

        self.batch_modules = nn.ModuleList(
            [
                nn.BatchNorm1d(n_head * hidden[0]),
                nn.BatchNorm1d(n_head * hidden[1]),
                nn.BatchNorm1d(n_head * hidden[2]),
            ]
        )

        self.mlp = nn.Sequential(
            *[
                nn.Dropout(p = p),
                nn.Linear(2 * hidden[2] * n_head, mlp_hidden),
                nn.BatchNorm1d(mlp_hidden),
                nn.ReLU(),
                nn.Dropout(p=p),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.BatchNorm1d(mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, num_classes),
            ]
        )

    def forward(self, inputs):
        x, edge_idx, batch_idx = inputs.x,  inputs.edge_index, inputs.batch

        for gat_layer, batch_layer in zip(self.GAT_modules, self.batch_modules):
            x = gat_layer(x, edge_idx)
            x = batch_layer(x)

        x = torch.cat(
            [global_max_pool(x, batch_idx), global_mean_pool(x, batch_idx)], dim=1
        )
        x = self.mlp(x)
        return x

class GConvNet(nn.Module):
    def __init__(self, num_features, num_classes, hidden, alpha = 0.01, p = 0.5):
        super(GConvNet, self).__init__()
        torch.manual_seed(42)
        self.num_classes = num_classes
        self.num_features = num_features
        self.hidden = hidden
        self.p = p
        self.alpha = alpha

        self.init_graph = GCNLayer(num_features, hidden, p, alpha)
        self.head = GCNLayer(hidden, hidden, p, alpha)
        self.body = GCNLayer(hidden, hidden, p, alpha)
        self.tail = GCNLayer(hidden, hidden, p, alpha)

        self.mlp = nn.ModuleList(
            [
                nn.Linear(4 * self.hidden, self.hidden),
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.num_classes),
            ]
        )
       
    def forward(self, inputs):
        x, edge_idx, batch_idx = inputs.x,  inputs.edge_index, inputs.batch
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
        return x

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))
        


class HGNN(nn.Module):
    def __init__(self, num_classes : int, num_features : int, num_heads : int, hidden : int, p = 0.5, alpha = 0.01):
        super(HGNN, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_heads = num_heads
        self.hidden = hidden
        self.p = p
        self.alpha = alpha

        self.init_graph = GATLayer(num_features, hidden, num_heads, alpha, p)
        self.head = GATLayer(hidden*num_heads, hidden, num_heads, alpha=alpha, p = p)
        self.body = GATLayer(hidden*num_heads, hidden, num_heads, alpha=alpha, p = p)
        self.tail = GATLayer(hidden*num_heads, hidden, num_heads, alpha=alpha, p = p)

        self.mlp = nn.ModuleList(
            [
                nn.Linear(4 * self.hidden * self.num_heads, self.hidden * self.num_heads),
                nn.BatchNorm1d(self.hidden * self.num_heads),
                nn.ReLU(),
                nn.Linear(self.hidden * self.num_heads, self.num_classes),
            ]
        )

    def forward(self, inputs):
        x, edge_idx, batch_idx = inputs.x,  inputs.edge_index, inputs.batch
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
        return x

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))

class ChebNet(nn.Module):
    def __init__(self,  num_features : int, num_classes : int, k : int, hidden : int,  alpha = 0.01, p = 0.5):
        super(ChebNet, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.k = k
        self.hidden = hidden
        self.p = p
        self.alpha = alpha

        self.init_graph = ChebConvLayer(num_features, k, hidden, p, alpha)
        self.head = ChebConvLayer(hidden, k, hidden, p, alpha)
        self.body = ChebConvLayer(hidden, k, hidden, p, alpha)
        self.tail = ChebConvLayer(hidden, k, hidden, p, alpha)

        self.mlp = nn.ModuleList(
            [
                nn.Linear(4 * self.hidden, self.hidden),
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.num_classes),
            ]
        )
    
    def forward(self, inputs):
        x, edge_idx, batch_idx = inputs.x,  inputs.edge_index, inputs.batch
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
        return x

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))

class GMMNet(nn.Module):
    def __init__(self,  num_features : int, num_classes : int, dim : int, hidden : int, kernel_size : int, separate_gaussians : bool = False, aggr : str = 'mean', alpha = 0.01, p = 0.5):
        super(GMMNet, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.dim = dim
        self.hidden = hidden
        self.p = p
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.separate_gaussians = separate_gaussians

        self.init_graph = GMMConvLayer(num_features, dim, hidden, kernel_size, separate_gaussians, p, alpha, aggr)
        self.head = GMMConvLayer(hidden, dim, hidden, kernel_size, separate_gaussians, p, alpha, aggr)
        self.body = GMMConvLayer(hidden, dim, hidden, kernel_size, separate_gaussians, p, alpha, aggr)
        self.tail = GMMConvLayer(hidden, dim, hidden, kernel_size, separate_gaussians, p, alpha, aggr)

        self.mlp = nn.ModuleList(
            [
                nn.Linear(4 * self.hidden, self.hidden),
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.num_classes),
            ]
        )
    
    def forward(self, inputs):
        x, edge_idx, batch_idx, edge_attr = inputs.x,  inputs.edge_index, inputs.batch, inputs.edge_attr
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
        return x

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))


class DynamicReductionNetwork(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 1, k=16, p = 0.5, aggr='add', norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
        super(DynamicReductionNetwork, self).__init__()
        self.datanorm = nn.Parameter(norm)
        self.k = k

        start_width = 2 * hidden_dim
        middle_width = 3 * hidden_dim // 2

        self.start_width = start_width
        self.middle_width = middle_width
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggr = aggr

        self.inputnet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),
            # nn.BatchNorm1d(hidden_dim//2),
            nn.ELU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
        )

        convnn1 = nn.Sequential(
            nn.Linear(start_width, middle_width),
            nn.BatchNorm1d(middle_width),
            nn.ELU(),
            nn.Linear(middle_width, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
        )

        convnn2 = nn.Sequential(
            nn.Linear(start_width, middle_width),
            nn.BatchNorm1d(middle_width),
            nn.ELU(),
            nn.Linear(middle_width, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
        )

        self.edgeconv1 = EdgeConv(nn = convnn1, aggr = aggr)
        self.edgeconv2 = EdgeConv(nn = convnn2, aggr = aggr)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ELU(),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, inputs):
        inputs.x = self.datanorm * inputs.x
        x, edge_idx, batch_idx = inputs.x,  inputs.edge_index, inputs.batch
        x = self.inputnet(x)

        # to_undirected : edge_idx -> undirected graph
        # knn_graph : compute graph edge to the nearest k points
        # input : feature node, k(num of neighbor), batch(batch vector), loop(self-loop, boolean), flow(source_to_target or target_to_source)
        edge_idx = to_undirected(knn_graph(x, self.k, batch_idx, loop = False, flow = self.edgeconv1.flow))
        x = self.edgeconv1(x, edge_idx)
        weight = normalized_cut_2d(edge_idx, x)

        # A greedy clustering algorithm from the "Weighted Graph Cuts without Eigenvectors: A Multilevel Approach"
        cluster = graclus(edge_idx, weight, x.size(0))
        
        inputs.x = x
        inputs.edge_index = edge_idx
        inputs.edge_attr = None

        inputs = max_pool(cluster, inputs)

        inputs.edge_index = to_undirected(knn_graph(inputs.x, self.k, inputs.batch, loop = False, flow = self.edgeconv2.flow))
        inputs.x = self.edgeconv2(inputs.x, inputs.edge_index)

        weight = normalized_cut_2d(inputs.edge_index, inputs.x)
        cluster = graclus(inputs.edge_index, weight, inputs.x.size(0))
        
        x, batch_idx = max_pool_x(cluster, inputs.x, inputs.batch)

        x = global_max_pool(x, batch_idx)

        # output = self.output(x)
        output = self.output(x).squeeze(-1)

        return output

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))