from robustness import *
from robustness.audio_robust import add_audio_noise
from robustness.visual_robust import colorize, WB, periodic
import torch
import numpy as np
from PIL import Image

def test_audio():
    """Test audio module."""
    np.random.seed(0)
    from robustness.audio_robust import add_audio_noise
    test = torch.zeros((4,2,20))
    idf = add_audio_noise(test, noise_level=0, noises=None)
    assert idf.shape == (4,2,20)

def test_tabular():
    """Test tabular module."""
    np.random.seed(0)
    from robustness.tabular_robust import add_tabular_noise
    test = torch.zeros((4,1,20))
    idf = add_tabular_noise(test)
    assert idf.shape == (4,1,20)
    
def test_visual():
    """Test vision module."""
    np.random.seed(0)
    from robustness.visual_robust import add_visual_noise
    test = np.zeros((3,128,128))
    idf = add_visual_noise(test, noise_level=1)
    assert idf[0].shape == (128,128)
    idf = add_visual_noise(test, noise_level=0)
    assert idf[0].shape == (128,128)

    imarray = np.random.rand(100,100,3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    im2 = colorize(im,1.0)
    assert im2.size == (100,100)
    im2 = WB(im,1.0)
    assert im2.size == (100,100)    

def test_text():
    """Test text module."""
    np.random.seed(0)
    from robustness.text_robust import add_text_noise
    test = ["Hello", "Darkness","My old", "Friend."]
    idf = add_text_noise(test, noise_level=1)
    assert len(idf) in [1,2,3,4]
    
def test_ts():
    np.random.seed(0)
    from robustness.timeseries_robust import add_timeseries_noise
    test = [[ np.array([1.0,2.0,3.0]),np.array([4.0,5.0,6.0])]]
    idf = add_timeseries_noise(test, noise_level=1)
    assert len(idf) in [1,2]
    idf = add_timeseries_noise(test, noise_level=0)
    assert len(idf) in [1,2]
