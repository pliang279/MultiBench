"""Defines some helper nn.module instances."""
import torch
from torch import nn


class Sequential2(nn.Module):
    """Implements a simpler version of sequential that handles inputs with 2 arguments."""
    
    def __init__(self, a, b):
        """Instatiate Sequential2 object.

        Args:
            a (nn.Module): First module to sequence
            b (nn.Module): Second module
        """
        super(Sequential2, self).__init__()
        self.model = nn.Sequential(a, b)

    def forward(self, x):
        """Apply Sequential2 modules to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return self.model(x)
