import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GATv2Conv
from .base_gnn import BaseModel


class GAT(BaseModel):
    """GAT"""
    def __init__(self, dim_in, dim_h, num_layers, dropout=False, pool_type='add', num_heads=1):
        super().__init__(num_layers, dropout, pool_type)
        num_heads = 3
        print(self.num_layers)
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
    
    