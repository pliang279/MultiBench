from utils.aux_models import *

def test_CellBlock():
    """Test if CellBlock preserves dimensions."""
    class args:
        planes = 3
        drop_path = 0.1
        kernel_size = 3
    for i in range(10):
        cb = CellBlock(i,i,args)
        input_t = torch.zeros((1,3,128,128))
        out = cb(input_t, input_t)
        assert out.shape == (1,3,128,128)

def test_Id():
    """"""
    id = Identity()
    assert id(torch.zeros((3,3))).shape == (3,3)


def test_3():
    """"""
    fn = Tensor1DLateralPadding(1)
    assert fn(torch.zeros((3,3))).shape == (3,4)
    fn = ChannelPadding(1)
    assert fn(torch.zeros((1,3,3,4))).shape == (1,4,3,4)
    fn = GlobalPooling2D()
    assert fn(torch.zeros((1,3,4,4))).shape == (1,3)
    fn = GlobalPooling1D()
    assert fn(torch.zeros((1,3,4,4))).shape == (1,3,4)
    fn = Maxout(4,4,4)
    assert fn(torch.zeros(1,3,4,4)).shape == (1,3,4,4)
    fn = AlphaScalarMultiplication(4,4)
    assert fn(torch.zeros(1,3,4,4),torch.zeros(1,3,4,4))[0].shape == (1,3,4,4)
    fn = AlphaVectorMultiplication(4)
    assert fn(torch.zeros(1,3,4,4)).shape == (1,3,4,4)
    fn = FactorizedReduction(4,4)
    assert fn(torch.zeros(1,4,4,4)).shape == (1,4,2,2)

def test_Cell():
    pass