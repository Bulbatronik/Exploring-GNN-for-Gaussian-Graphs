import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from .base_gnn import BaseModel


class GCN(BaseModel):
    """GCN"""
    def __init__(self, dim_in, dim_h, num_layers, dropout=False, pool_type='add', manual=False):
        super().__init__(num_layers, dropout, pool_type)
      
        for i in range(self.num_layers):
            if i == 0:
                if manual:
                    self.layers.append(VanillaGNNLayer(dim_in, dim_h))
                else:
                    self.layers.append(GCNConv(dim_in, dim_h)) 
            else:
                if manual:
                    self.layers.append(VanillaGNNLayer(dim_h, dim_h))
                else:   
                    self.layers.append(GCNConv(dim_h, dim_h))
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
    
    
####################### VanillaGCN #####################
class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = Linear(dim_in, dim_out, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        # H = D_hat^-1*A_hat.T*D_hat^-1*X*W.T 
        x = self.linear(x) # X*W.T
        
        A = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)[0] 
        Atilde = A + torch.eye(A.shape[0]).to(edge_index.device)
        
        Dtilde = torch.diag(1 / torch.sqrt(torch.sum(Atilde, dim=0))).to(edge_index.device)
        adj_norm = Dtilde @ Atilde.T @ Dtilde
        
        x = adj_norm @ x
        
        return x 
        
    def __str__(self):
        return f"VanillaGCNLayer(dim_in={self.dim_in}, dim_out={self.dim_out})"