
import unimodals.common_models as cm
import torch

def test_LSTM():
    """Test LSTM shape."""
    test = torch.zeros([32, 50, 35])
    model = cm.LSTM(35,2, linear_layer_outdim=2, dropout=True)
    assert model(test).shape == (32,2)



def test_VGG():
    """Test VGG shape."""
    test = torch.zeros([32, 3, 128, 128])
    model = cm.VGG(35)
    assert model(test)[0].shape == (32,512)

def test_ResNetLSTM():
    """Test ResNetLSTM shape."""
    test = torch.zeros([1,3, 150, 112, 112])
    model = cm.ResNetLSTMEnc(10, dropout=True)
    assert model(test).shape == (1,10)