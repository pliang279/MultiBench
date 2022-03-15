"""Implements a Transformer with EarlyFusion."""
import torch
import torch.nn.functional as F
from torch import nn


class EarlyFusionTransformer(nn.Module):
    """Implements a Transformer with Early Fusion."""
    
    embed_dim = 9

    def __init__(self, n_features):
        """Initialize EarlyFusionTransformer Object.

        Args:
            n_features (int): Number of features in input.
        
        """
        super().__init__()

        self.conv = nn.Conv1d(n_features, self.embed_dim,
                              kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=3)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.linear = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        """Apply EarlyFusion with a Transformer Encoder to input.

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Layer Output
        """
        x = self.conv(x.permute([0, 2, 1]))
        x = x.permute([2, 0, 1])
        x = self.transformer(x)[-1]
        return self.linear(x)
