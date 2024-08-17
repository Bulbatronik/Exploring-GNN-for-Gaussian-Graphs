import os
import sys
import copy
import random

from pathlib import Path

import numpy as np
import pandas as pd

import networkx as nx

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, ModuleList

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.io import fs

from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, GraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from scipy.sparse import coo_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from .base_gnn import BaseModel

class GIN(BaseModel):
    """GIN"""
    def __init__(self, dim_in, dim_h, num_layers, dropout=False):
        super().__init__(num_layers, dropout)
      
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(GINConv(
            Sequential(Linear(dim_in, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU())))
            else:
                self.layers.append(GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU())))
        self.layers = ModuleList(self.layers)
        
        self.classifier = Linear(self.num_layers*dim_h, self.num_layers*dim_h)
        self.output = Linear(self.num_layers*dim_h, 1)

    def forward(self, x, edge_index, edge_weight, batch):
        # Node embeddings 
        h = []
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index) # edge_weight is not supported
            h.append(x)
        
        # Graph-level readout
        for i in range(self.num_layers):
            h[i] = global_mean_pool(h[i], batch)
        
        # Concatenate graph embeddings
        x = torch.cat(h, dim=1)
        
        # Classifier
        x = self.classifier(x)
        if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        x = self.output(x)
        
        return torch.sigmoid(x)