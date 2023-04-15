import os
import sys
sys.path.append(os.getcwd())
from fusions.common_fusions import *
from tests.common import *
from unimodals.common_models import LeNet, MLP
from fusions.mult import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_concat(set_seeds):
    """Test concat."""
    fusion = Concat()
    output = fusion([torch.randn((1,2)) for _ in range(2)])
    assert output.shape == (1,4)
    assert np.isclose(torch.norm(output).item(), 2.7442679405212402)
    assert count_parameters(fusion) == 0 

def test_concat_early(set_seeds):
    """Test concat early."""
    fusion = ConcatEarly()
    output = fusion([torch.randn((1,2,2)) for _ in range(2)])
    assert output.shape == (1,2,4) 
    assert np.isclose(torch.norm(output).item(), 3.395326614379883)
    assert count_parameters(fusion) == 0 

def test_stack(set_seeds):
    """Test stack."""
    fusion = Stack()
    output = fusion([torch.randn((1,2,2)) for _ in range(2)])
    assert output.shape == (1,4,2)
    assert np.isclose(torch.norm(output).item(), 3.395326614379883)
    assert count_parameters(fusion) == 0 
    
def test_concat_linear(set_seeds):
    """Test concat linear."""
    fusion = ConcatWithLinear(2,2)
    output = fusion([torch.randn((1,2,2)) for _ in range(2)])
    assert output.shape == (1,4,2)
    assert np.isclose(torch.norm(output).item(), 0.9164801836013794)
    assert count_parameters(fusion) == 6 
    
def test_tensor_fusion(set_seeds):
    """Test tensor fusion."""
    fusion = TensorFusion()
    inputs = [torch.randn((1,2,2)) for _ in range(2)]
    output = fusion(inputs)
    assert output.shape == (1,2,9)
    assert np.isclose(torch.norm(output).item(), 5.0617828369140625)
    output = fusion([inputs[0]])
    assert output.shape == (1,2,2)
    assert np.isclose(torch.norm(output).item(), 2.7442679405212402)
    assert count_parameters(fusion) == 0 
    try:
        output = fusion([torch.randn((1,2,2)) for _ in range(3)])
    except Exception as e:
        assert isinstance(e, AssertionError)

def test_low_rank_tensor_fusion(set_seeds):
    """Test low rank tensor fusion."""
    fusion = LowRankTensorFusion((10,10),2,1)
    output = fusion([torch.randn((10,10)).to(device) for _ in range(2)])
    assert output.shape == (10,2)
    assert np.isclose(torch.norm(output).item(), 0.6151207089424133)
    assert count_parameters(fusion) == 3 
    fusion = LowRankTensorFusion((10,10),2,1,flatten=False)
    output = fusion([torch.randn((10,10)).to(device) for _ in range(2)])
    assert output.shape == (10,2)
    assert np.isclose(torch.norm(output).item(), 8.852971076965332)
    assert count_parameters(fusion) == 3
    
def test_multiplicative_interaction_models(set_seeds):
    """Test multiplicative interaction models."""
    data = [torch.randn((64,1,28,28)),torch.randn((64,1,112,112)),torch.cat((torch.ones((32,)),torch.randn((32,))),dim=0).long()]
    channels = 1
    encoders = [LeNet(1, channels, 3), LeNet(1, channels, 5)]

    # fusion=Concat().cuda()
    fusion = MultiplicativeInteractions2Modal(
        [channels*8, channels*32], channels*40, 'matrix')
    out = fusion([encoders[0](data[0]), encoders[1](data[1])])
    assert out.shape == (64,40)
    assert np.isclose(torch.norm(out).item(), 89.92830657958984)
    assert count_parameters(fusion) == 11880


    fusion = MultiplicativeInteractions2Modal(
        [channels*8, channels*32], [channels,channels], 'matrix3D')
    out = fusion([encoders[0](data[0]), encoders[1](data[1])])
    assert out.shape == (64,1,1)
    assert np.isclose(torch.norm(out).item(), 42.0806884765625)
    assert count_parameters(fusion) == 297

    fusion = MultiplicativeInteractions2Modal(
        [channels*8, channels*32], channels*40, 'vector')
    out = fusion([encoders[0](data[0]), encoders[1](data[1])])
    assert out.shape == (64,32)
    assert np.isclose(torch.norm(out).item(), 98.5400619506836)
    assert count_parameters(fusion) == 576

    fusion = MultiplicativeInteractions2Modal(
        [channels*8, channels*32], channels*40, 'scalar', grad_clip=(0.1,0.2))
    out = fusion([encoders[0](data[0]), encoders[1](data[1])])
    assert out.shape == (64,32)
    assert np.isclose(torch.norm(out).item(), 88.90837097167969)
    try:
        fusion([encoders[0](data[0]), encoders[1](data[1]), encoders[1](data[1])])
    except Exception as e:
        assert isinstance(e, AssertionError)
    assert count_parameters(fusion) == 18

    fusion = MultiplicativeInteractions2Modal(
        [channels*8, channels*32], channels*40, 'scalar', grad_clip=(0.1,0.2), flip=True, flatten=True, clip=(-0.5, 0.5))
    out = fusion([encoders[1](data[1]), encoders[0](data[0])])
    assert out.shape == (64,32)
    assert np.isclose(torch.norm(out).item(), 55.64186477661133)
    assert count_parameters(fusion) == 18

    fusion = MultiplicativeInteractions2Modal(
        [channels*8], channels*40, 'scalar', grad_clip=(0.1,0.2))
    out = fusion([encoders[0](data[0])])
    assert np.isclose(torch.norm(out).item(), 27.642284393310547)
    assert count_parameters(fusion) == 18

    fusion = MultiplicativeInteractions3Modal(
        [channels*8, channels*32, channels*8], channels*40)

    out = fusion([encoders[0](data[0]), encoders[1](data[1]),encoders[0](data[0])])
    assert out.shape == (64,64,40)
    assert np.isclose(torch.norm(out).item(), 1081.623291015625)
    assert count_parameters(fusion) == 106920

    fusion = MultiplicativeInteractions3Modal(
        [channels*8, channels*32, channels*8], channels*40, task='affect')
    out = fusion([encoders[0](data[0]), encoders[1](data[1]),encoders[0](data[0])])
    assert np.isclose(torch.norm(out).item(), 124.64234924316406)
    assert count_parameters(fusion) == 106920


def test_nl_gate(set_seeds):
  fusion = NLgate(30, 10, 30, (10, 300), (10, 300), (10, 300))
  out = fusion([torch.randn((30,10)),torch.randn((30,10))])
  assert out.shape == (30,300)
  assert np.isclose(torch.norm(out).item(), 56.99603271484375)
  assert count_parameters(fusion) == 9900
  fusion = NLgate(30, 10, 30, None, None, None)
  out = fusion([torch.randn((30,10)),torch.randn((30,10))])
  assert out.shape == (1,300)
  assert np.isclose(torch.norm(out).item(), 19.206192016601562)
  assert count_parameters(fusion) == 0

def test_MULTModel(set_seeds):
    class HParams():
            num_heads = 8
            layers = 4
            attn_dropout = 0.1
            attn_dropout_modalities = [0,0,0.1]
            relu_dropout = 0.1
            res_dropout = 0.1
            out_dropout = 0.1
            embed_dropout = 0.2
            embed_dim = 40
            attn_mask = True
            output_dim = 1
            all_steps = False

    data = [torch.randn([32, 50, 20]), torch.randn([32, 50, 5]), torch.randn([32, 50, 300]), torch.cat((torch.ones((16,1)),torch.randn((16,1))),dim=0).long()]
    hparams = HParams()
    fusion = MULTModel(3, [20, 5, 300], hyp_params=hparams)
    out = fusion([data[0],data[1],data[2]])
    assert out.shape == (32,1)
    assert np.isclose(torch.norm(out).item(), 2.946321487426758)
    assert count_parameters(fusion) == 3076961
    
    hparams.all_steps = True
    hparams.embed_dim = 9
    hparams.num_heads = 3
    fusion = MULTModel(3, [20, 5, 300], hyp_params=hparams)
    out = fusion([data[0],data[1],data[2]])
    assert out.shape == (32,50,1)
    assert np.isclose(torch.norm(out).item(), 21.332000732421875)
    assert count_parameters(fusion) == 165007

    mp = make_positions(torch.randn([1,3,3]), padding_idx=0, left_pad=1)
    assert (mp.numpy() == np.vstack([np.arange(3)+1 for _ in range(3)])).all()

def test_EarlyFusionTransformer(set_seeds):
    fusion = EarlyFusionTransformer(10)
    out = fusion(torch.randn((3,10,10)))
    assert out.shape == (3,1)
    assert np.isclose(torch.norm(out).item(),1.1512198448181152)

def test_LateFusionTransformer(set_seeds):
    fusion = LateFusionTransformer()
    out = fusion(torch.randn((3,10,10)))
    assert out.shape == (3,9)
    assert np.isclose(torch.norm(out).item(),5.1961259841918945)
