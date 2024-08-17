import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList

from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool



from .base_gnn import BaseModel

class GCONV(BaseModel):
    """GraphConv"""
    def __init__(self, dim_in, dim_h, num_layers, dropout=False):
        super().__init__(num_layers, dropout)
      
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(GraphConv(dim_in, dim_h))
            else:
                self.layers.append(GraphConv(dim_h, dim_h))
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
    
