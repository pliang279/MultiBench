from fusions.MVAE import *
from unimodals.MVAE import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_a():
    poe = ProductOfExperts((2,2,2))
    assert poe(torch.ones(10,2,2).to(device),torch.zeros(10,2,2).to(device))[0].shape == (2,2)
    poe2 = ProductOfExperts_Zipped((2,2,2))
    assert poe2(torch.ones(10,2,2,2).to(device))[0].shape == (2,2)


def test_b():
    
    mlp = MLPEncoder(10,2,1)
    assert mlp(torch.ones(10,10))[0].shape == (10,1)