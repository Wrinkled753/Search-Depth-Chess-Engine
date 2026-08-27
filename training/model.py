import torch
import torch.nn as nn

class ChessEvalNet(nn.Module):
    """
    A simple feedforward neural network for chess position evaluation.
    Takes a 769-bit feature vector (64 squares x 12 pieces + 1 turn bit) 
    and outputs a single scalar value [-1, 1].
    """
    def __init__(self, hidden_size=256, num_hidden_layers=2):
        super(ChessEvalNet, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(769, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            
        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Tanh()) # Binds output between -1 and 1
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
