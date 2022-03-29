from robustness import *
from robustness.audio_robust import add_audio_noise
import torch
import numpy as np

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
    
def test_tabular():
    """Test vision module."""
    from robustness.visual_robust import add_visual_noise
    test = np.zeros((3,128,128))
    idf = add_visual_noise(test, noise_level=1)
    assert idf[0].shape == (128,128)
    

def test_text():
    """Test text module."""
    from robustness.text_robust import add_text_noise
    test = ["Hello", "Darkness","My old", "Friend."]
    idf = add_text_noise(test, noise_level=1)
    assert len(idf) in [1,2,3,4]
    