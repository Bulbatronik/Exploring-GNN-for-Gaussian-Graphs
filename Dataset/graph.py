"""Ref: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets"""
import numpy as np
from scipy.sparse import coo_matrix

import torch
from torch_geometric.io import fs
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data


class DatasetGraph(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.load(self.processed_paths[0])
        print(f"Number of samples in the dataset: {len(self.data_list)}")

    @property
    def processed_file_names(self):
        return 'data.pt'

    @classmethod
    def save(cls, data_list, path) -> None:
        r"""Saves a list of data objects to the file path :obj:`path`."""
        data, slices = cls.collate(data_list)
        fs.torch_save((data.to_dict(), slices, data.__class__), path)


    def process(self):
        self.save(self.data_list, self.processed_paths[0])
    
    def to_loader(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
        
        

def to_data_list(A, x, y, device):
    """Converts input data into a list of :class:`Data` objects."""
    data_list = []
    for idx in range(len(A)):
        edge_index = coo_matrix(A[idx])
        sample = Data(x=torch.tensor(x[idx], dtype=torch.float32).to(device), 
                    edge_index=torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.int64).to(device), 
                    edge_weight=torch.tensor(edge_index.data, dtype=torch.float32).to(device), 
                    y=torch.tensor(y[idx], dtype=torch.float32).to(device))
        data_list.append(sample)
    return data_list
    