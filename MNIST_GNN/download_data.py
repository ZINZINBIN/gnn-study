import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataListLoader, DataLoader

dataset = MNISTSuperpixels('dataset', True, pre_transform=T.ToSparseTensor(remove_edge_index=False, attr = 'edge_weight'), transform = T.Cartesian())
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

sample = next(iter(data_loader))

print("sample data x:", sample.x.size())
print("sample data edge idx:", sample.edge_index.size())
print("sample data edge attr:", sample.edge_attr.size())