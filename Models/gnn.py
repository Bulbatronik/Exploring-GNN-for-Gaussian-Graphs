# TODO 1. Normalize links 2. Try node embeddings 3. Convert data for MLP to pytorch dataset
from torch_geometric.utils import to_dense_adj

class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = Linear(dim_in, dim_out, bias=False)

    def forward(self, x, edge_index):
        x = self.linear(x)
        adjacency = to_dense_adj(edge_index)[0]
        adjacency += torch.eye(len(adjacency)).to(device) # add self-loop
        x = torch.sparse.mm(adjacency, x)
        return x
    
    
class VanillaGNN(torch.nn.Module):
    # TODO: Include edge_weight
    """Vanilla Graph Neural Network"""
    def __init__(self, dim_in, dim_h):
        super().__init__()
        self.gnn1 = VanillaGNNLayer(dim_in, dim_h)
        self.gnn2 = VanillaGNNLayer(dim_h, dim_h)
        
        self.lin1 = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, 1)

    def forward(self, x, edge_index, edge_weight, batch):
        # Node embeddings 
        h = self.gnn1(x, edge_index)
        h = torch.relu(h)
        h = self.gnn2(h, edge_index)
        h = torch.relu(h)
        
        h = global_add_pool(h, batch)
        
        # Classifier
        h = self.lin1(h)
       
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return torch.sigmoid(h)

    def fit(self, train_loader, val_loader, epochs, verbose=False):
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001) # weight_decay=5e-4)


        self.train()
        for epoch in range(epochs+1): 
            train_loss, train_acc = 0.0, 0.0
            
            # Train on batches
            for data in train_loader:
                optimizer.zero_grad()
                out_train = self(data.x, data.edge_index, data.edge_weight, data.batch).view(-1)
                loss = criterion(out_train, data.y)
                
                train_loss += loss / len(train_loader)
                train_acc += accuracy(out_train>=0.5, data.y) / len(train_loader)
                
                loss.backward()
                optimizer.step()

            if(epoch % 20 == 0) and verbose:
                val_loss, val_acc = self.test(val_loader)
                print(f'Epoch {epoch:>3} | Train Loss: {train_loss:.3f} | Train Acc:'
                    f' {train_acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | '
                    f'Val Acc: {val_acc*100:.2f}%')

    @torch.no_grad()      
    def test(self, loader):
        criterion = torch.nn.BCELoss()
        self.eval()
        loss, acc = 0.0, 0.0
        for data in loader:
            out = self(data.x, data.edge_index, data.edge_weight, data.batch).view(-1)
            loss += criterion(out, data.y) / len(loader)
            acc += accuracy(out>=0.5, data.y) / len(loader)
        return loss, acc