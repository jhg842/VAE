import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import build_encoder
from .decoder import build_decoder

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        bs, c, h, w = x.size()
        x = x.view(bs, -1)
        z, mu, log_var = self.encoder(x)
        x = self.decoder(z)
        x = x.view(bs, c, h, w)
        
        return x, mu, log_var
    
    
class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, recon_x, target, mu, log_var):
        reconst_loss = F.binary_cross_entropy(recon_x, target, reduction='sum')
        kl_div = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1)
        
        return reconst_loss + kl_div
    
# def criterion(output, target, mu, log_var):
#     reconst_loss = F.binary_cross_entropy(output, target, reduction='sum')
#     kl_div = -0.5 * torch.sum(mu.pow(2) + log_var.exp() - logvar - 1)
    
#     return reconst_loss + kl_div

def build(args):
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    
    model = VAE(encoder, decoder)

    criterion = Criterion()
    
    
    return model, criterion
