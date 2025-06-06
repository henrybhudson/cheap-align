import torch
import torch.nn as nn

class CheapAlign(nn.Module):
        """
        Single 3x3 conv that learns to imitate the SD-VAE decoder.
        """
        def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
                nn.init.xavier_uniform_(self.conv.weight)
                nn.init.zeros_(self.conv.bias)
                
        def forward(self, x):
                return torch.relu(self.conv(x))