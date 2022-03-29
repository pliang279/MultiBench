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

def test_CellBlock():
    """Test if CellBlock preserves dimensions."""
    id = Identity()
    assert id(torch.zeros((3,3))).shape == (3,3)