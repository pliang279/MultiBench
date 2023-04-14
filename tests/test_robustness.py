import os
import sys
sys.path.append(os.getcwd())
from robustness.audio_robust import *
from robustness.visual_robust import *
from robustness.tabular_robust import *
from robustness.text_robust import *
from robustness.timeseries_robust import *
from tests.common import *
import torch
import numpy as np
from PIL import Image

def test_audio(set_seeds):
    """Test audio module."""
    test = torch.rand((1,20)) + 0.1
    idf = add_audio_noise(test, noise_level=1, noises=[ additive_white_gaussian_noise])
    assert np.isclose(np.linalg.norm(idf[0]), 6.85951)  
    test = torch.rand((1,20)) + 0.1
    idf = add_audio_noise(test, noise_level=1, noises=[ audio_random_dropout])
    assert np.isclose(np.linalg.norm(idf[0]), 0)  
    test = torch.rand((1,20)) + 0.1
    idf = add_audio_noise(test, noise_level=1, noises=[ audio_structured_dropout])
    assert np.isclose(np.linalg.norm(idf[0]), 0)  
    test = torch.rand((1,20)) + 0.1
    idf = add_audio_noise(test, noise_level=1, noises=None)
    assert idf.shape == test.shape 

def test_tabular(set_seeds):
    """Test tabular module."""
    test = torch.rand((4,2,20))
    idf = add_tabular_noise(test, noise_level=1)
    assert np.isclose(np.linalg.norm(idf), 0)
    test = torch.rand((4,2,20))
    idf = add_tabular_noise(test, noise_level=0)
    assert np.isclose(idf, test).all()
    
def test_visual(set_seeds):
    """Test vision module."""
    from PIL import Image
    np.random.seed(0)
    test = np.random.random((3,128,128))
    idf = add_visual_noise(test, noise_level=1)
    assert np.isclose(np.linalg.norm(idf),25996.547)
    assert idf[0].shape == (128,128)
    test = np.random.random((3,128,128))
    idf = add_visual_noise(test, noise_level=0)
    for i in range(len(test)):
        img = Image.fromarray(test[i])
        img = img.convert('RGB').convert(img.mode)
        assert np.isclose(idf[i], img).all()

    imarray = np.random.rand(100,100,3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    im2 = colorize(im,1.0)
    assert im2.size == (100,100)
    im2 = WB(im,1.0)
    assert im2.size == (100,100)    

def test_text(set_seeds):
    """Test text module."""
    text = ['Assistance', 'imprudence', 'yet', 'sentiments', 'unpleasant', 'expression', 'met', 'surrounded', 'not', 'Saw', 'vicinity', 'judgment', 'remember', 'finished', 'men', 'throwing.']
    idf_swap = add_text_noise(text[:3], noise_level=1, rand_mid=False, typo=False, sticky=False, omit=False)
    idf_rand_mid = add_text_noise(text[3:6], noise_level=1, swap=False, typo=False, sticky=False, omit=False)
    idf_typo = add_text_noise(text[6:10], noise_level=1, swap=False, rand_mid=False, sticky=False, omit=False)
    idf_sticky = add_text_noise(text[10:12], noise_level=1, swap=False, rand_mid=False, typo=False, omit=False)
    idf_omit = add_text_noise(text[12:], noise_level=1, swap=False, rand_mid=False, typo=False, sticky=False)
    idf = np.concatenate([idf_swap, idf_rand_mid, idf_typo, idf_sticky, idf_omit])
    target = ["assistnace", "imprduence", "yet", "stnmeteins", "usneaanplt", "esorpixesn", "met", "surrounsed", "not", "saw", "vicinitty", "judggment", "rememer", "finised", "men", "thrwing ."]
    for i in range(len(idf)):
        assert idf[i] == target[i]
    
def test_ts(set_seeds):
    np.random.seed(0)
    test = [[ np.array([1.0,2.0,3.0]),np.array([4.0,5.0,6.0])]]
    idf = add_timeseries_noise(test, noise_level=1)
    assert np.isclose(idf, np.zeros((2,3))).all()
    test = [[ np.array([1.0,2.0,3.0]),np.array([4.0,5.0,6.0])]]
    idf = add_timeseries_noise(test, noise_level=0)
    assert np.isclose(idf, test).all()
