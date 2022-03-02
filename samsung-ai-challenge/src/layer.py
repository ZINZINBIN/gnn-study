import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import data_parallel
import torch_geometric
from torch_geometric.nn import GCNConv,GATConv, ChebConv, GMMConv, GATv2Conv, SAGEConv, GraphConv
from torch_scatter import scatter_max
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool

class GCNLayer(nn.Module):
    def __init__(self, in_features : int, out_features : int, alpha : float)->None:
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.gconv = GCNConv(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.act = nn.LeakyReLU(alpha)
        
    def forward(self, x, edge_idx = None):
        h = self.gconv(x, edge_idx)
        h = self.norm(h)
        h = self.act(h)
        return h

class GATLayer(nn.Module):
    def __init__(self, num_features, hidden, num_head, alpha = 0.01, p = 0.5):
        super(GATLayer, self).__init__()
        self.num_features = num_features
        self.hidden = hidden
        self.num_head = num_head
        self.alpha = alpha

        self.gat = GATConv(num_features, hidden, num_head, dropout = p, edge_dim = 2)
        self.norm  = nn.BatchNorm1d(num_head * hidden)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x, edge_idx  = None, edge_attr = None):
        x = self.gat(x, edge_idx, edge_attr)
        x = self.norm(x)
        x = self.act(x)
        return x

class ChebConvLayer(nn.Module):
    def __init__(self, n_dims_in : int, k : int, n_dims_out : int, alpha = 0.01):
        super(ChebConvLayer, self).__init__()
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out
        self.k = k
        self.alpha = alpha

        self.cheb = ChebConv(n_dims_in, n_dims_out, k, normalization='sym')
        self.norm = nn.BatchNorm1d(n_dims_out)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x, edge_idx = None, edge_attr = None):
        x = self.cheb(x,edge_idx, edge_attr)
        x = self.norm(x)
        x = self.act(x)
        return x
    
class GMMConvLayer(nn.Module):
    def __init__(self, n_dims_in : int, dim : int, n_dims_out : int, kernel_size : int, separate_gaussians : bool = False, alpha = 0.01, aggr : str = 'mean'):
        super(GMMConvLayer, self).__init__()
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out
        self.dim = dim
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.separate_gaussians = separate_gaussians
        self.aggr = aggr

        self.GMMConv = GMMConv(n_dims_in, n_dims_out, dim, kernel_size, separate_gaussians=separate_gaussians, aggr = aggr)
        self.norm = nn.BatchNorm1d(n_dims_out)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x, edge_idx = None, edge_attr = None):
        x = self.GMMConv(x,edge_idx,edge_attr)
        x = self.norm(x)
        x = self.act(x)
        return x

class SAGEConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, alpha: float):
        super(SAGEConvLayer, self).__init__()
        self.n_dims_in = in_features
        self.n_dims_out = out_features
        self.alpha = alpha

        self.GMMConv = SAGEConv(in_features, out_features)
        self.norm = nn.BatchNorm1d(self.n_dims_out)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x, edge_idx = None, edge_attr = None):
        x = self.GMMConv(x,edge_idx)
        x = self.norm(x)
        x = self.act(x)
        return x     


class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, alpha: float, aggr : str = "add"):
        super(GraphConvLayer, self).__init__()
        self.n_dims_in = in_features
        self.n_dims_out = out_features
        self.alpha = alpha
        self.aggr = aggr

        self.GraphConv = GraphConv(in_features, out_features, aggr = aggr)
        self.norm = nn.BatchNorm1d(self.n_dims_out)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x, edge_idx=None, edge_attr=None):
        x = self.GraphConv(x, edge_idx, edge_attr)
        x = self.norm(x)
        x = self.act(x)
        return x
