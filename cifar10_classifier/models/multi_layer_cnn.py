import torch
import torch.nn as nn

class MultiLayerCnn(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.multi_layer_cnn_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(), 

            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),

            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.multi_layer_cnn_stack(x)
