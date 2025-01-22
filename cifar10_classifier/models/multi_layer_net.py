import torch.nn as nn

class MultiLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(32*32*3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        logits = self.multi_layer_stack(x)
        return logits