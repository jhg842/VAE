import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)