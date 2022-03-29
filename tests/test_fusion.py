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
    
def test_sl6():
  from unimodals.common_models import LeNet, MLP, Constant
  import torch
  from fusions.common_fusions import MultiplicativeInteractions2Modal
  data = [torch.zeros((64,1,28,28)),torch.zeros((64,1,112,112)),torch.cat((torch.ones((32,)),torch.zeros((32,))),dim=0).long()]
    
  channels = 1
  encoders = [LeNet(1, channels, 3), LeNet(1, channels, 5)]
  head = MLP(channels*40, 100, 10)

  # fusion=Concat().cuda()
  fusion = MultiplicativeInteractions2Modal(
      [channels*8, channels*32], channels*40, 'matrix')
  
  out = fusion([encoders[0](data[0]), encoders[1](data[1])])
  assert out.shape == (64,40)

  fusion = MultiplicativeInteractions2Modal(
      [channels*8, channels*32], [channels,channels], 'matrix3D')
  
  out = fusion([encoders[0](data[0]), encoders[1](data[1])])
  assert out.shape == (64,1,1)

  fusion = MultiplicativeInteractions2Modal(
      [channels*8, channels*32], channels*40, 'vector')
  
  out = fusion([encoders[0](data[0]), encoders[1](data[1])])
  assert out.shape == (64,32)

  fusion = MultiplicativeInteractions2Modal(
      [channels*8, channels*32], channels*40, 'scalar', grad_clip=(0.1,0.2))
  
  out = fusion([encoders[0](data[0]), encoders[1](data[1])])
  assert out.shape == (64,32)

  fusion = MultiplicativeInteractions3Modal(
      [channels*8, channels*32, channels*8], channels*40)
  
  out = fusion([encoders[0](data[0]), encoders[1](data[1]),encoders[0](data[0])])
  assert out.shape == (64,64,40)

  fusion = NLgate(24, 30, 10, None, (10, 300), (10, 300))
  out = fusion([torch.zeros((24,30)),torch.zeros((30,10))])
  assert out.shape == (30,720)
