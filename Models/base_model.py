import torch

from sklearn.metrics import accuracy_score

class BaseModel(torch.nn.Module):
    """Multilayer Perceptron"""
    def __init__(self, dim_in, dim_h):
        super().__init__()
        pass
        #self.linear1 = Linear(dim_in, dim_h)
        #self.linear2 = Linear(dim_h, 1)

    def forward(self, x):
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
            for x_batch, y_batch in train_loader: 
                optimizer.zero_grad()
                out_train = self(x_batch)
                loss = criterion(out_train, y_batch)
                
                train_loss += loss / len(train_loader)
                train_acc += accuracy(out_train>=0.5, y_batch) / len(train_loader)
                
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
        for x_batch, y_batch in loader:
            out = self(x_batch)
            loss += criterion(out, y_batch) / len(loader)
            acc += accuracy(out>=0.5, y_batch) / len(loader)
        return loss, acc