import torch
import torch.nn as nn

class OneLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.one_layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(32*32*3, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        logits = self.one_layer_stack(x)
        return logits
