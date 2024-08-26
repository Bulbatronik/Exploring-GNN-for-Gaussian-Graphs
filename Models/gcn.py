import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_scatter import scatter_sum
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, add_self_loops
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
        self.W = Linear(dim_in, dim_out, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        """ Straightforward implementation
        # H = D_hat^-1*A_hat.T*D_hat^-1*X*W.T 
        
        x = self.W(x) # X*W.T
        
        A = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)[0] 
        Atilde = A + torch.eye(A.shape[0]).to(edge_index.device)
        
        Dtilde = torch.diag(1 / torch.sqrt(torch.sum(Atilde, dim=0))).to(edge_index.device)
        adj_norm = Dtilde @ Atilde.T @ Dtilde
        
        x = adj_norm @ x
        
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

        # Compute degree d
        d = scatter_sum(torch.ones_like(edge_index[0]), index=edge_index[0], dim=0)
        
        # Compute convolution weights
        d_i = torch.index_select(d, index=edge_index[0], dim=0)
        d_j = torch.index_select(d, index=edge_index[1], dim=0)
        
        # Avoid division by zero
        d_i[d_i == 0] = 1e-10
        d_j[d_j == 0] = 1e-10
        
        w = torch.unsqueeze(1 / torch.sqrt(d_i * d_j), -1)

        # Aggregate the information
        h_prime = scatter_sum(x_j*w, index=edge_index[0], dim=0)
    
        return h_prime
    
    def __str__(self):
        return f"VanillaGCNLayer({self.dim_in}, {self.dim_out})"