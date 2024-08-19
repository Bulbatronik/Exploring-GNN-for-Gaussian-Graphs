import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from .evaluation import accuracy


# Create MLP mode
class MLP(torch.nn.Module):
    """Multilayer Perceptron"""
    def __init__(self, dim_in, dim_h, num_layers, dropout=False):
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = []
        self.dropout = dropout
        
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(Linear(dim_in, dim_h))
            else:
                self.layers.append(Linear(dim_h, dim_h))
        
        # To resolve device issues
        self.linear = ModuleList(self.layers)
        
        # Classifier
        self.classifier = Linear(dim_h, dim_h)
        self.output = Linear(dim_h, 1)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = torch.relu(x)
            if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        
        # Classifier
        x = self.classifier(x)
        if self.dropout:
                x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        x = self.output(x)
        
        return torch.sigmoid(x)

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
            for x_batch, y_batch in train_loader: 
                optimizer.zero_grad()
                out_train = self(x_batch)
                loss = criterion(out_train, y_batch)
                
                train_loss += loss / len(train_loader)
                train_acc += accuracy(out_train>=0.5, y_batch) / len(train_loader)
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
        for x_batch, y_batch in loader:
            out = self(x_batch)
            loss += criterion(out, y_batch) / len(loader)
            acc += accuracy(out>=0.5, y_batch) / len(loader)
        return loss, acc