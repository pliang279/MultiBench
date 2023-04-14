import pytest
import torch 
import numpy as np

@pytest.fixture
def set_seeds():
    """Set seeds for reproducibility."""
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)