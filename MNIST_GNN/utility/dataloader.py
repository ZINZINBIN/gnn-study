import torch
import torch_geometric
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataListLoader, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

def generate_dataset(valid_ratio = 0.2):
    dataset = MNISTSuperpixels(
        'dataset', True, pre_transform=T.ToSparseTensor(remove_edge_index=False, attr = 'edge_weight'))
    test_dataset = MNISTSuperpixels(
        'dataset', False, pre_transform=T.ToSparseTensor(remove_edge_index=False, attr = 'edge_weight'))

    total_len = dataset.len()
    train_size = int(total_len * (1 - valid_ratio))
    train_indices = [idx for idx in range(0, train_size)]
    valid_indices = [idx for idx in range(train_size, total_len)]

    train_dataset = dataset.index_select(train_indices)
    valid_dataset = dataset.index_select(valid_indices)

    return train_dataset, valid_dataset, test_dataset

def generate_loader(batch_size = 128, valid_ratio = 0.2, add_attr = False):

    if add_attr:
        dataset = MNISTSuperpixels('dataset', True, pre_transform=T.ToSparseTensor(remove_edge_index=False, attr = 'edge_weight'), transform = T.Cartesian())
        test_dataset = MNISTSuperpixels('dataset', False, pre_transform=T.ToSparseTensor(remove_edge_index=False, attr = 'edge_weight'), transform = T.Cartesian()) 
    else:
        dataset = MNISTSuperpixels('dataset', True, pre_transform=T.ToSparseTensor(remove_edge_index=False, attr = 'edge_weight'))
        test_dataset = MNISTSuperpixels('dataset', False, pre_transform=T.ToSparseTensor(remove_edge_index=False, attr = 'edge_weight')) 


    total_len = dataset.len()
    train_size = int(total_len  * (1 - valid_ratio))

    total_indices = [idx for idx in range(0, train_size)]
    train_indices, valid_indices = train_test_split(total_indices, test_size = valid_ratio, random_state = 42, shuffle = True)

    train_dataset = dataset.index_select(train_indices)
    valid_dataset = dataset.index_select(valid_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def generate_parallel_loader(batch_size=128, valid_ratio=0.2, is_testset_parallel = False):
    dataset = MNISTSuperpixels(
        'dataset', True, pre_transform=T.ToSparseTensor(remove_edge_index=False, attr = 'edge_weight'))
    test_dataset = MNISTSuperpixels(
        'dataset', False, pre_transform=T.ToSparseTensor(remove_edge_index=False, attr = 'edge_weight'))

    total_len = dataset.len()
    train_size = int(total_len * (1 - valid_ratio))
    train_indices = [idx for idx in range(0, train_size)]
    valid_indices = [idx for idx in range(train_size, total_len)]

    train_dataset = dataset.index_select(train_indices)
    valid_dataset = dataset.index_select(valid_indices)

    train_loader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    if is_testset_parallel:
        test_loader = DataListLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader
