import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, LeakyReLU
from torch_scatter import scatter_sum
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax
from .base_gnn import BaseModel


class GAT(BaseModel):
    """GAT"""
    def __init__(self, dim_in, dim_h, num_layers, dropout=False, pool_type='add', manual=False, num_heads=3):
        super().__init__(num_layers, dropout, pool_type)
        for i in range(self.num_layers):
            if self.num_layers == 1: # only one layer -> 1 head
                if manual:
                    self.layers.append(VanillaGATLayer(dim_in, dim_h, heads=1))
                else:
                    self.layers.append(GATv2Conv(dim_in, dim_h, heads=1))
            elif i == 0: # first layer -> dim_in
                if manual:
                    self.layers.append(VanillaGATLayer(dim_in, dim_h, heads=num_heads, concat=True))
                else:
                    self.layers.append(GATv2Conv(dim_in, dim_h, heads=num_heads))
            elif i == self.num_layers - 1: # last layer -> 1 head
                if manual:
                    self.layers.append(VanillaGATLayer(dim_h*num_heads, dim_h, heads=1))
                else:
                    self.layers.append(GATv2Conv(dim_h*num_heads, dim_h, heads=1))
            else: # middle layer
                if manual:
                    self.layers.append(VanillaGATLayer(dim_h*num_heads, dim_h, heads=num_heads, concat=True))
                else:
                    self.layers.append(GATv2Conv(dim_h*num_heads, dim_h, heads=num_heads))
        self.layers = ModuleList(self.layers)
        
        self.classifier = Linear(dim_h, dim_h)
        self.output = Linear(dim_h, 1)
        
    def forward(self, x, edge_index, edge_weight, batch):
        # Node embeddings 
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
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


####################### VanillaGAT #####################  
class VanillaGATLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, heads, concat=False):
        super().__init__()
        self.heads = heads
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.concat = concat

        self.attention_heads = ModuleList([
            GATHead(dim_in, dim_out) for _ in range(heads)
        ])

    def forward(self, x, edge_index, edge_weight=None):
        # Apply each attention head
        head_outputs = [att_head(x, edge_index) for att_head in self.attention_heads]

        if self.concat:
            # Concatenate all head outputs along the features
            return torch.cat(head_outputs, dim=1)
        else:
            # Average the head outputs along the heads
            return torch.mean(torch.stack(head_outputs), dim=0)
    
    def __repr__(self):
        return f"VanillaGATLayer({self.dim_in}, {self.dim_out}, heads={self.heads}, concat={self.concat})"
    

class GATHead(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = Linear(in_features, out_features, bias=False)
        self.a = Linear(2*out_features, 1, bias=False)
        self.leakyrelu = LeakyReLU(0.2)

    
    def forward(self, x, edge_index, edge_weight=None):
        """ Straightforward implementation
        # H = A_hat.T*W_alpha*X*W.T 
        
        x = self.W(x) # X*W.T
        
        A = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight)[0] 
        Atilde = A + torch.eye(A.shape[0]).to(edge_index.device)
        
        
        a_input = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        e = self.leakyrelu(self.a(a_input))
        
        E = torch.zeros_like(A.shape)
        E[edge_index[0], edge_index[1]] = e[0]
        
        W_alpha = F.softmax(E, dim=1)
        
        x = A.T @ W_alpha @ x
        
        return x 
        """ 
        # Linear transformation
        h = self.W(x)

        # Compute attention coefficients
        x_i, x_j = h[edge_index[0]], h[edge_index[1]]
        a_input = torch.cat( [x_i, x_j], dim=1)
        e = self.leakyrelu(self.a(a_input))

        # Compute attention weights using softmax
        attention = softmax(e, edge_index[0])
        
        # Apply attention weights and ggregate the information
        h_prime = scatter_sum(attention * x_j, index=edge_index[0], dim=0)
        
        return h_prime