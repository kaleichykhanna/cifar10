import torch.nn as nn

class OneLayerCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.one_layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 10),
        )
    def forward(self, x):
        logits = self.one_layer_stack(x)
        return logits
