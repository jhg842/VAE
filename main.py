import argparse
import os
import random
import time

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler


from models import build_model



def get_args_parser():
    parser = argparse.ArgumentParser('Set generative model', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    parser.add_argument('--in_dim', default=3*256*256, type=int)
    parser.add_argument('--hidden_dim', default=2048, type=int)
    parser.add_argument('--z_dim', default=3*256*256, type=int)
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    return parser
    
def main(args):
    
    device = torch.device(args.device)
    model, criterion = build_model(args)
    model.to(device)
    criterion.to(device)
    loss = criterion(output, target, mu, logvar)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('VAE', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)