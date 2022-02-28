import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import data_parallel
import torch_geometric
from torch_geometric.nn import GCNConv,GATConv, ChebConv, GMMConv
from torch_scatter import scatter_max
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool

'''
# layer list
- GCNLayer(torch_geometric)
- GATLayer(torch_geometric)
- GATLayerAdj
- GATLayerEdgeAverage
- GATLayerEdgeSoftmax
- GATLayerMultiHead
'''
class GCNLayer(nn.Module):
    def __init__(self, in_features : int, out_features : int, p : float,  alpha : float)->None:
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p= p
        self.alpha = alpha
        self.gconv = GCNConv(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(p)
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
        self.p = p

        self.gat = GATConv(num_features, hidden, num_head, dropout = p)
        self.norm  = nn.BatchNorm1d(num_head * hidden)
        self.act = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(p)

    def forward(self, x, edge_idx  = None):
        x = self.gat(x, edge_idx)
        x = self.norm(x)
        x = self.act(x)
        return x

class ChebConvLayer(nn.Module):
    def __init__(self, n_dims_in : int, k : int, n_dims_out : int, p = 0.5, alpha = 0.01):
        super(ChebConvLayer, self).__init__()
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out
        self.k = k
        self.p = p
        self.alpha = alpha

        self.cheb = ChebConv(n_dims_in, n_dims_out, k, normalization='sym')
        self.norm = nn.BatchNorm1d(n_dims_out)
        self.act = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(p)

    def forward(self, x, edge_idx = None):
        x = self.cheb(x,edge_idx)
        x = self.norm(x)
        x = self.act(x)
        return x
    
class GMMConvLayer(nn.Module):
    def __init__(self, n_dims_in : int, dim : int, n_dims_out : int, kernel_size : int, separate_gaussians : bool = False, p = 0.5, alpha = 0.01, aggr : str = 'mean'):
        super(GMMConvLayer, self).__init__()
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out
        self.dim = dim
        self.p = p
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.separate_gaussians = separate_gaussians
        self.aggr = aggr

        self.GMMConv = GMMConv(n_dims_in, n_dims_out, dim, kernel_size, separate_gaussians=separate_gaussians, aggr = aggr)
        self.norm = nn.BatchNorm1d(n_dims_out)
        self.act = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(p)

    def forward(self, x, edge_idx = None, edge_attr = None):
        x = self.GMMConv(x,edge_idx,edge_attr)
        x = self.norm(x)
        x = self.act(x)
        return x


class GATLayerAdj(nn.Module):
    def __init__(self, di, do, eps = 1e-6):
        super(GATLayerAdj, self).__init__()
        self.di = di
        self.do = do
        self.eps = eps

        self.f = nn.Linear(2 * di, do)
        self.w = nn.Linear(2 * di, 1)
        self._init_weights()

    def forward(self, x, adj):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Msrc -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        """
        N = x.size()[0]
        hsrc = x.unsqueeze(0).expand(N, -1, -1) # 1, N, i -> N, N, i
        htgt = x.unsqueeze(1).expand(-1, N, -1) # N, 1, i -> N, N, i

        h = torch.concat([hsrc, htgt], dim = 2) # N, N, 2i

        a = self.w(h) # N, N, 1
        a_sqz = a.squeeze(2) # N, N

        a_zero = -1e16*torch.ones_like(a_sqz) # N,N
        a_mask = torch.where(adj>0,a_sqz,a_zero) # N,N -> adj >0 then a_mask_ij = a_sqz_ij or a_mask_ij = a_zero_ij
        a_att = F.softmax(a_mask,dim=1) # N,N
        
        y = self.act(self.f(h)) # N,N,do
        y_att = a_att.unsqueeze(-1)*y # (N,N,1) * (N,N,do)
        o = y_att.sum(dim=1).squeeze() # (N,1,do) => (N,do)
        return o

    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)

class GATLayerEdgeAverage(nn.Module):
    def __init__(self, di, do, eps = 1e-6):
        super(GATLayerEdgeAverage, self).__init__()
        self.di = di
        self.do = do
        self.eps = eps
        self.f = nn.Linear(2*di,do)
        self.w = nn.Linear(2*di,1)
        self._init_weights()
        self.eps = eps

    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self,x,adj,src,tgt,Msrc,Mtgt):
        hsrc = x[src] # E,i
        htgt = x[tgt] # E,i
        h = torch.cat([hsrc,htgt],dim=1) # E,2i
        y = torch.relu(self.f(h)) # E,o
        a = self.w(h) # E,1
        a_sum = torch.mm(Mtgt, a) + self.eps # N,E x E,1 = N,1
        o = torch.mm(Mtgt,y * a) / a_sum # N,1
        assert not torch.isnan(o).any()

        return o

class GATLayerEdgeSoftmax(nn.Module):
    def __init__(self, di, do, eps=1e-6):
        super(GATLayerEdgeSoftmax,self).__init__()
        self.f = nn.Linear(2*di,do)
        self.w = nn.Linear(2*di,1)
        self._init_weights()
        self.eps = eps
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)
    
    def forward(self,x,adj,src,tgt,Msrc,Mtgt):
        hsrc = x[src] # E,i
        htgt = x[tgt] # E,i
        h = torch.cat([hsrc,htgt],dim=1) # E,2i
        y = torch.relu(self.f(h)) # E,o
        a = self.w(h) # E,1
        assert not torch.isnan(a).any()
        a_base, _ = torch.max(a,0,keepdim=True)#[0] + self.eps
        assert not torch.isnan(a_base).any()
        a_norm = a-a_base
        assert not torch.isnan(a_norm).any()
        a_exp = torch.exp(a_norm)
        assert not torch.isnan(a_exp).any()
        a_sum = torch.mm(Mtgt,a_exp) + self.eps # N,E x E,1 = N,1
        assert not torch.isnan(a_sum).any()
        o = torch.mm(Mtgt,y * a_exp) / a_sum # N,1
        assert not torch.isnan(o).any()
        return o

class GATLayerMultiHead(nn.Module):
    def __init__(self, d_in, d_out, num_heads):
        super(GATLayerMultiHead, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads

        self.GAT_heads = nn.ModuleList(
              [
                GATLayerAdj(d_in,d_out)
                for _ in range(num_heads)
              ]
        )
    
    def forward(self,x,adj):
        '''
        [GATLayerEdgeSoftmax : (E, d_out)] -> (E, d_out * num_heads)
        '''
        return torch.cat([l(x,adj) for l in self.GAT_heads], dim=1)

