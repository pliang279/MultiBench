
import unimodals.common_models as cm
import unimodals.robotics.encoders as robenc
import unimodals.robotics.decoders as robdec
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

def test_robotics_encoders():
    enc = robenc.ActionEncoder(10)
    assert enc(torch.zeros(32,10)).shape == (32,32)
    assert enc(None) is None
    enc = robenc.DepthEncoder(10,0.2)
    assert enc(torch.zeros(32,1,128,128))[0].shape == (32,20,1)
    enc = robenc.ImageEncoder(10,0.2)
    assert enc(torch.zeros(32,128,128,3))[0].shape == (32,20,1)
    enc = robenc.ForceEncoder(10,0.2)
    assert enc(torch.zeros(32,6,32))[0].shape == (20,1)
    enc = robenc.ProprioEncoder(10,0.2)
    assert enc(torch.zeros(32,8))[0].shape == (20,1)


    
    