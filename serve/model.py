import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self, hidden_dim, dropout =0.3):
        
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(427, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        
        self.fc4 = nn.Linear(hidden_dim, 32)
        
        self.fc5 = nn.Linear(32, 3)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        print("Inside Model")
        print(x.shape)
        out = self.dropout(F.relu(self.fc1(x)))
        
        out = self.dropout(F.relu(self.fc2(out)))
        
        out = self.dropout(F.relu(self.fc3(out)))
        
        out = self.dropout(F.relu(self.fc4(out)))
        
        out = self.fc5(out)
        
        return out
