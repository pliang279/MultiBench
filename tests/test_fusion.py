from fusions.common_fusions import *

def test_common():
    """."""
    fusion = Concat()
    assert fusion([torch.zeros((1,2)) for _ in range(2)]).shape == (1,4)
    fusion = ConcatEarly()
    assert fusion([torch.zeros((1,2,2)) for _ in range(2)]).shape == (1,2,4) 
    fusion = Stack()
    assert fusion([torch.zeros((1,2,2)) for _ in range(2)]).shape == (1,2,4)
    
    fusion = ConcatWithLinear(2,2)
    assert fusion([torch.zeros((1,2,2)) for _ in range(2)]).shape == (1,2,4)