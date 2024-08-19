import torch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from .evaluation import accuracy


class BaseModel(torch.nn.Module):
    def __init__(self, num_layers, dropout, pool_type):
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = []
        self.dropout = dropout
        self.pool_type = pool_type

    def fit(self, train_loader, val_loader, epochs, verbose=False):
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001) # weight_decay=5e-4)

        # store the results
        history = {'train_loss': [], 'train_acc': [], 
                 'val_loss': [], 'val_acc': []}
        
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
                val_loss, val_acc = self.test(val_loader)
                
                loss.backward()
                optimizer.step()

            # Store the results in the dictionary
            history['train_loss'].append(train_loss.item())
            history['train_acc'].append(train_acc.item())
            history['val_loss'].append(val_loss.item())
            history['val_acc'].append(val_acc.item())
                
            if(epoch % 20 == 0) and verbose:
                val_loss, val_acc = self.test(val_loader)
                print(f'Epoch {epoch:>3} | Train Loss: {train_loss:.3f} | Train Acc:'
                    f' {train_acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | '
                    f'Val Acc: {val_acc*100:.2f}%')
        return history
    
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
    
    def pool(self, x, batch):
        if self.pool_type == 'add':
            return global_add_pool(x, batch)
        elif self.pool_type == 'mean':
            return global_mean_pool(x, batch)
        elif self.pool_type == 'max':
            return global_max_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pool type: {self.pool_type}. Valid options are 'add', 'mean', or 'max'.")
    
    def __repr__(self):
        layers = ''
        for i in range(self.num_layers):
            layers += str(self.layers[i]) + '\n'
        layers += f'Pooling: {self.pool_type}' + '\n'
        layers += str(self.classifier) + '\n'
        layers += str(self.output) + '\n'
        return layers