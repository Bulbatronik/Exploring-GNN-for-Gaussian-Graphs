import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_scatter import scatter_sum
from torch_geometric.utils import to_dense_adj, add_self_loops
from .base_gnn import BaseModel


class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.W = Linear(dim_in, dim_out, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        """ Straightforward implementation
        # H = A_hat.T*X*W.T
        x = self.W(x) # X*W.T
        
        A = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)[0]
        Atilde = A + torch.eye(A.shape[0]).to(edge_index.device)
        
        x = Atilde.T @ x
        
        return x
        """
        # Linear transformation
        x = self.W(x)
        
        # Select the neighbors
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight) 
        x_j = x[edge_index[1]]
        
        # Scale features by edge weights
        if edge_weight is not None:
            x_j = x_j * edge_weight[edge_index[1]].view(-1, 1)  

        # Aggregate the information
        h_prime = scatter_sum(x_j, index=edge_index[0], dim=0)
        
        return h_prime 
    
    def __str__(self):
        return f"VanillaGNNLayer({self.dim_in}, {self.dim_out})"
    
    
class GNN(BaseModel):
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
        x = torch.relu(x)
        x = self.output(x)
        
        return torch.sigmoid(x)