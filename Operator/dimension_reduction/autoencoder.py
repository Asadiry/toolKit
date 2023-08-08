import os

import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=in_dim),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    
