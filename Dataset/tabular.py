import torch 
from torch.utils.data import Dataset, DataLoader

class DatasetTabular(Dataset):
    def __init__(self, x, y, device='cpu'):
        self.x = torch.tensor(x.reshape((x.shape[0], -1)), dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        print(f"Number of samples in the dataset: {len(self.x)}")
              
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index,: ], self.y[index]
    
    def to_loader(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)