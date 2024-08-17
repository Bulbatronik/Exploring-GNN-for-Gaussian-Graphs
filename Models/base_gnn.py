import torch

from .utils import accuracy

class BaseModel(torch.nn.Module):
    """Multilayer Perceptron"""
    def __init__(self, num_layers, dropout):
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = []
        self.dropout = dropout
        #self.linear1 = Linear(dim_in, dim_h)
        #self.linear2 = Linear(dim_h, 1)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        pass
        #h = self.linear1(x)
        #h = torch.relu(h)
        #h = F.dropout(h, p=0.5, training=self.training)
        #h = self.linear2(h)
        #return torch.sigmoid(h)

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
    
    
    
    def __repr__(self):
        layers = ''
        for i in range(self.num_layers):
            layers += str(self.layers[i]) + '\n'
        layers += str(self.classifier) + '\n'
        layers += str(self.output) + '\n'
        return layers