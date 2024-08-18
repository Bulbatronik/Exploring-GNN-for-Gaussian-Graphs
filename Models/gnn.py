import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.utils import to_dense_adj
from .base_gnn import BaseModel


class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = Linear(dim_in, dim_out, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.linear(x) # X*W.T
        adjacency = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)[0]
        # add self-loop
        adjacency += torch.eye(len(adjacency)).to(edge_index.device) # A_hat 
        
        # A_hat.T*X*W.T
        x = adjacency.T @ x
        
        return x
    
    def __str__(self):
        return f"VanillaGNNLayer(dim_in={self.dim_in}, dim_out={self.dim_out})"
    
    
class VanillaGNN(BaseModel):
    """Vanilla GNN"""
    def __init__(self, dim_in, dim_h, num_layers, dropout=False, pool_type='add'):
        super().__init__(num_layers, dropout, pool_type)
        
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
        x = self.pool(x, batch) 
        
        # Classifier
        x = self.classifier(x)
        if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        x = self.output(x)
        
        return torch.sigmoid(x)