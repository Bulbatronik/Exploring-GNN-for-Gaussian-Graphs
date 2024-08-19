import torch
import torch.nn.functional as F
from torch.nn import Softmax
from torch.nn import Linear, ModuleList, LeakyReLU
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax, to_dense_adj
from .base_gnn import BaseModel


class GAT(BaseModel):
    """GAT"""
    def __init__(self, dim_in, dim_h, num_layers, dropout=False, pool_type='add', manual=False, num_heads=1):
        super().__init__(num_layers, dropout, pool_type)
        num_heads = 3
        for i in range(self.num_layers):
            if self.num_layers == 1: # only one layer -> 1 head
                self.layers.append(GATv2Conv(dim_in, dim_h, heads=1))
            elif i == 0: # first layer -> dim_in
                self.layers.append(GATv2Conv(dim_in, dim_h, heads=num_heads))
            elif i == self.num_layers - 1: # last layer -> 1 head
                self.layers.append(GATv2Conv(dim_h*num_heads, dim_h, heads=1))
            else: # middle layer
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
        x = self.classifier(x)
        x = self.output(x)
        
        return torch.sigmoid(x)


####################### VanillaGAT #####################  
class VanillaGATLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, heads=1, concat=True, negative_slope=0.2, dropout=0.0, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.heads = heads
        self.concat = concat
        
        self.lin = Linear(dim_in, heads * dim_out, bias=False)
        self.att_lin = Linear(2 * dim_out, heads, bias=False) 
        
    
    def forward(self, x, edge_index):
        # Step 1: Linear Transformation
        x = self.lin(x)
        # Reshape x to [num_nodes, heads, out_channels]
        x = x.view(-1, self.heads, self.out_channels)

        # Step 2: Compute the attention scores
        s, d = edge_index
        x_i, x_j  = x[s], x[d]  # Source and target node embeddings [num_edges, heads, out_channels]
          
        # Concatenate the embeddings of connected nodes
        x_ij = torch.cat([x_i, x_j], dim=-1)  # [num_edges, heads, 2 * out_channels]
        
        # Apply the linear layer to compute attention scores
        e_ij = F.leaky_relu(self.att_lin(x_ij), 0.2)  # [num_edges, heads]

        # Step 3: Normalize the attention scores
        alpha = softmax(e_ij, s, num_nodes=x.size(0))

        # Step 5: Compute the aggregated node features
        out = alpha.unsqueeze(-1) * x_j  # [num_edges, heads, out_channels]
        out = torch.zeros_like(x_i).scatter_add_(0, s.unsqueeze(-1).expand_as(out), out)

        # Step 6: Concatenate or average the head outputs
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)  # [num_nodes, heads * out_channels]
        else:
            out = out.mean(dim=1)  # [num_nodes, out_channels]

        return out

"""
class GATLayer(Module):
  def __init__(self, in_channels, out_channels):
    #One layer of Graph Attention Network (GAT)
    #
    #Args:
    #    in_channels: (int) - input dimension
    #    out_channels: (int) - output dimension
    super().__init__()
    self.a_i = Linear(out_channels,1, bias=False)
    self.a_j = Linear(out_channels,1, bias=False)

    self.linear = Linear(in_channels, out_channels)
    self.negative_slope = 0.2

  def forward(self, x, edge_index):
    #Args:
    #    x: (n, in_dim) - initial node features
    #    edge_index: (2, e) - list of edge indices
    #
    #Returns:
    #    out: (n, out_dim) - updated node features
    x = self.linear(x)
    # select all the source nodes for each edge
    x_j = torch.index_select(x, index=edge_index[0], dim=0)
    # select all the destination nodes for each edge
    x_i = torch.index_select(x, index=edge_index[1], dim=0)

    # ============ ANSWER ==============
    # Implement the equation above to compute the
    # node update corresonding to GAT
    #
    e = F.leaky_relu(self.a_i(x_i) + self.a_j(x_j), self.negative_slope)
    alpha = softmax(e, edge_index[0])
    out = scatter_sum(alpha * x_j, index=edge_index[1], dim=0)
    #
    # ===========================================

    return out
"""