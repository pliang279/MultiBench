
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

def test_res3d2():
    from functools import reduce
    def get_param_count(model):
        return sum(reduce(lambda a,b: a*b, x.size()) for x in model.parameters() if x.requires_grad)
    from unimodals.res3d import generate_model, ResNet, BasicBlock, _get_inplanes
    assert get_param_count(generate_model(18)) == 33421164
    assert get_param_count(generate_model(34)) == 63747500
    model = generate_model(50)
    assert model(torch.zeros((1,3,3,128,128))).shape == (1,400)
    assert get_param_count(generate_model(50)) == 47035382
    assert get_param_count(generate_model(101)) == 86101110
    assert get_param_count(generate_model(152)) == 118277110
    assert get_param_count(generate_model(200)) == 127485942
    assert get_param_count(ResNet(BasicBlock, [1, 1, 1, 1], _get_inplanes(),shortcut_type='A')) == 14434220
    model = ResNet(BasicBlock, [1, 1, 1, 1], _get_inplanes(),shortcut_type='A')
    assert model(torch.zeros((1,3,3,128,128))).shape == (1,400)
    
    