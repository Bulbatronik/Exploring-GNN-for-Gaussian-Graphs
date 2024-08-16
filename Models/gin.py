class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_in, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dim_in, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        #self.conv3 = GINConv(
        #    Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
        #               Linear(dim_h, dim_h), ReLU()))
        
        self.lin1 = Linear(dim_h*2, dim_h*2)#*3
        self.lin2 = Linear(dim_h*2, 1)

    def forward(self, x, edge_index, edge_weight, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        #h3 = self.conv2(h2, edge_index)
        
        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        #h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2), dim=1)
        
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