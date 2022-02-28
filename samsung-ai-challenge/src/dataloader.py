import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Dataset, InMemoryDataset
from typing import Union, List, Tuple

class CustomDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform = None):
        super().__init__(root, transform)
        self.data_list = data_list
        self.data, self.slices = torch.load(self.process_paths[0])

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'
    
    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])