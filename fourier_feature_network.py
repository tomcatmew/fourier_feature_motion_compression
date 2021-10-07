import torch
import torch.nn as nn
import numpy as np
import math

class MLP(nn.Module):
    def __init__(self, depth=4, mapping_size=512, hidden_size=256):
        super().__init__()
        layers = []
        layers.append(nn.Linear(mapping_size, hidden_size))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_size, 127))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
        # return torch.sigmoid(self.layers(x))


def fourier_map(x, B):
    xp = torch.matmul(2 * math.pi * x, B)
    return torch.cat([torch.sin(xp), torch.cos(xp)], dim=-1)