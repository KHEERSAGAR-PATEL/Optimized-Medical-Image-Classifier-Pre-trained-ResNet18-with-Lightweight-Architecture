import torch
import torch.nn as nn

class LightweightNetwork(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_layers):
        super(LightweightNetwork, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.input_dim = input_channels

        self.layers = nn.ModuleList([
            nn.Linear(self.input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 2)  # Assuming binary classification

    def forward(self, x):
        x = self.pool(x).view(x.size(0), -1)  
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)  
        return x
