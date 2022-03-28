from robustness import *
from robustness.audio_robust import add_audio_noise
import torch

def test_audio():
    """Test audio module."""
    from robustness.audio_robust import add_audio_noise
    test = torch.zeros((4,2,20))
    idf = add_audio_noise(test, noise_level=0, noises=None)
    assert idf.shape == (4,2,20)

def test_tabular():
    """Test tabular module."""
    from robustness.tabular_robust import add_tabular_noise
    test = torch.zeros((4,1,20))
    idf = add_tabular_noise(test)
    assert idf.shape == (4,1,20)
