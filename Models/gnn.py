import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import to_dense_adj

from .base_gnn import BaseModel


class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = Linear(dim_in, dim_out, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.linear(x)
        adjacency = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)[0]
        adjacency += torch.eye(len(adjacency)).to(edge_index.device) # add self-loop
        x = torch.sparse.mm(adjacency, x)
        return x
    
    
class VanillaGNN(BaseModel):
    """Vanilla Graph Neural Network"""
    def __init__(self, dim_in, dim_h, num_layers, dropout=False):
        super().__init__(num_layers, dropout)
        
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(VanillaGNNLayer(dim_in, dim_h))
            else:
                self.layers.append(VanillaGNNLayer(dim_h, dim_h))
        self.layers = ModuleList(self.layers)
        
        self.classifier = Linear(dim_h, dim_h)
        self.output = Linear(dim_h, 1)
        
    def forward(self, x, edge_index, edge_weight, batch):
        # Node embeddings 
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index, edge_weight)
            x = torch.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        
        # Graph-level readout
        x = global_add_pool(x, batch) #global_add_pool, global_max_pool
        
        # Classifier
        x = self.classifier(x)
        if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        x = self.output(x)
        
        return torch.sigmoid(x)