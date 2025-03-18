import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,in_dim, hidden_dim, z_dim):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_var = nn.Linear(hidden_dim, z_dim)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        z = reparameterize(mu, log_var)
        
        return z, mu, log_var
    
def reparameterize(mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    
    return mu + eps*std

def build_encoder(args):
    
    return Encoder(args.in_dim, args.hidden_dim, args.z_dim)