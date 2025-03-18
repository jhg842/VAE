import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, in_dim ,hidden_dim, z_dim):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.out = nn.Linear(hidden_dim, in_dim)
        
    def forward(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        z = F.sigmoid(self.out(z))
        
        return z
    
def build_decoder(args):
    
    return Decoder(args.in_dim, args.hidden_dim, args.z_dim)
        