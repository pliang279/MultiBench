from fusions.common_fusions import *

def test_common():
    """."""
    fusion = Concat()
    assert fusion([torch.zeros((1,2)) for _ in range(2)]).shape == (1,4)
    fusion = ConcatEarly()
    assert fusion([torch.zeros((1,2,2)) for _ in range(2)]).shape == (1,2,4) 
    fusion = Stack()
    assert fusion([torch.zeros((1,2,2)) for _ in range(2)]).shape == (1,4,2)
    
    fusion = ConcatWithLinear(2,2)
    assert fusion([torch.zeros((1,2,2)) for _ in range(2)]).shape == (1,4,2)
    
    fusion = TensorFusion()
    assert fusion([torch.zeros((1,2,2)) for _ in range(2)]).shape == (1,2,9)
    fusion = LowRankTensorFusion((10,10),2,1)
    assert fusion([torch.zeros((10,10)) for _ in range(2)]).shape == (10,2)
    