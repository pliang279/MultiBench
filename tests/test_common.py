from unimodals.common_models import *
import torch


DATA_PATH = '/home/arav/MultiBench/MultiBench/'

def test_id():
    """Test Identity module."""
    id = Identity()
    test = torch.Tensor([0])
    assert id(test) == test
    
def test_linear():
    """Test Linear Module."""
    lin = Linear(3,4)
    test = torch.zeros((4,3))
    assert lin(test).shape == (4,4)
    
    lin = Linear(3,4, True)
    test = torch.zeros((4,3))
    assert lin(test).shape == (4,4)
    
def test_squeeze():
    """Test squeeze module."""
    lin = Squeeze(1)
    test = torch.zeros((4,1))
    assert lin(test).shape == (4,)
    lin = Squeeze()
    assert lin(test).shape == (4,)
    
def test_sequential():
    """Test sequential module."""
    lin = Sequential(Linear(1,2), Squeeze())
    test = torch.zeros((1,))
    assert lin(test, training=True).shape == (2,)
    
def test_reshape():
    """Test common module."""
    lin = Reshape((4,4))
    test = torch.zeros((16,))
    assert lin(test).shape == (4,4)
    
def test_transpose():
    """Test common module."""
    lin = Transpose(0,1)
    test = torch.zeros((3,4))
    assert lin(test).shape == (4,3)
    
def test_MLP():
    """Test common module."""
    lin = MLP(3,2,1, True, 0.1,True)
    test = torch.zeros((3,3))
    out = lin(test)
    assert out[0] == 0
    assert out[1].shape == test.shape
    assert out[2].shape == (3,2)
    assert out[3].shape == (3,2)
    lin = MLP(3,2,1)
    assert lin(test).shape == (3,1)


def test_MLP():
    """Test common module."""
    lin = MLP(3,2,1, True, 0.1,True)
    test = torch.zeros((3,3))
    out = lin(test)
    assert out[0] == 0
    assert out[1].shape == test.shape
    assert out[2].shape == (3,2)
    assert out[3].shape == (3,2)
    lin = MLP(3,2,1)
    assert lin(test).shape == (3,1)
    
def test_GRU():
    """Test common module."""
    lin = GRU(3,2,1, True)
    test = torch.zeros((3,3,3))
    out = lin(test)
    assert out[0].shape == (3,2)
    lin.flatten = True
    assert lin(test).shape == (3,6)
    lin.last_only = True
    assert lin(test).shape == (3,2)

def test_Constant():
    """Test constant module."""
    cons = Constant(1)
    test = torch.ones((3,3))
    assert cons(test).shape == (1,)
    assert cons(test)[0] == 0

def test_DAN():
    """Test DAN."""
    test = torch.ones((2,4))
    model = DAN(4,2)
    assert model(test).shape == (2,)


def test_Transformer():
    """Test Transformer Shape."""
    test = torch.ones((2,40,10))
    model = Transformer(10,10)
    assert model(test).shape == (2,10)


def test_GlobalPooling():
    """Test Module."""
    test = torch.ones((2,40,40))
    model = GlobalPooling2D()
    assert model(test).shape == (2,40)

def test_MaxOut_MLP():
    """Test Module."""
    test = torch.ones((2,40))
    model = MaxOut_MLP(10, number_input_feats=40)
    assert model(test).shape == (2,10)

def test_MaxOut():
    """Test Module."""
    test = torch.ones((2,10))
    model = Maxout(10,10,1)
    assert model(test).shape == (2,10)

def test_VGG():
    """Test Module."""
    test = torch.ones((1,3,128,128))
    model = VGG16(10)
    assert model(test).shape == (1,10)
    model = VGG16Slim(10)
    assert model(test).shape == (1,10)
    model = VGG11Slim(10)
    assert model(test).shape == (1,10)
    model = VGG11Pruned(10)
    assert model(test).shape == (1,10)
    model = VGG16Pruned(10)
    assert model(test).shape == (1,10)
    
def test_LeNet():
    """Test module."""
    test = torch.ones((1,3,128,128))
    model = LeNet(3,2,1)
    assert model(test).shape == (4,32,32)
    

def test_Seq():
    """Test stuff."""
    test = torch.ones((8,3,3))
    lin = GRUWithLinear(3,2,1, True)
    assert lin(test).shape == (8,3,1)
    lin = TwoLayersLSTM(3,3,True)
    assert lin(test).shape == (8,3,6)

def test_Resnet3d():
    """Test stuff."""
    from unimodals.res3d import generate_model
    test = torch.ones((3,3,128,128,128))
    lin = generate_model(10)
    assert lin(test).shape == (3,400)



"""
def test_integration():
    import torch
    import sys
    import os
    from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
    from training_structures.Supervised_Learning import train, test  # noqa
    from datasets.affect.get_data import get_dataloader  # noqa
    from fusions.common_fusions import ConcatEarly  # noqa


    traindata, validdata, testdata = get_dataloader(
        DATA_PATH+'mosi_raw.pkl', robust_test=False, max_pad=True, data_type='mosi', max_seq_len=50)

    encoders = [Identity(), Identity(), Identity()]
    head = Sequential(GRU(409, 512, dropout=True, has_padding=False,
                    batch_first=True, last_only=True), MLP(512, 512, 1))

    fusion = ConcatEarly()

    train(encoders, fusion, head, traindata, validdata, 1, task="regression", optimtype=torch.optim.AdamW,
        is_packed=False, lr=1e-3, save='mosi_ef_r0.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

    head = Sequential(Transformer(409, 300).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), MLP(300, 128, 1)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


    fusion = ConcatEarly().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    train(encoders, fusion, head, traindata, validdata, 1, task="regression", optimtype=torch.optim.AdamW,
        is_packed=False, lr=1e-3, save='mosi_ef_r0.pt', weight_decay=0.01, objective=torch.nn.L1Loss())


def test_integration2():
    from torch import nn
    import torch
    import sys
    import os
    from private_test_scripts.all_in_one import all_in_one_train # noqa
    from training_structures.MCTN_Level2 import train, test # noqa
    from unimodals.common_models import GRU, MLP # noqa
    from fusions.MCTN import Encoder, Decoder # noqa
    from datasets.affect.get_data import get_dataloader # noqa


    traindata, validdata, testdata = \
        get_dataloader(DATA_PATH+'mosi_raw.pkl', robust_test=False)

    max_seq = 20
    feature_dim = 300
    hidden_dim = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    encoder0 = Encoder(feature_dim, hidden_dim, n_layers=1, dropout=0.0).to(device)
    decoder0 = Decoder(hidden_dim, feature_dim, n_layers=1, dropout=0.0).to(device)
    encoder1 = Encoder(hidden_dim, hidden_dim, n_layers=1, dropout=0.0).to(device)
    decoder1 = Decoder(hidden_dim, feature_dim, n_layers=1, dropout=0.0).to(device)

    reg_encoder = nn.GRU(hidden_dim, 32).to(device)
    head = MLP(32, 64, 1).to(device)

    allmodules = [encoder0, decoder0, encoder1, decoder1, reg_encoder, head]


    def trainprocess():
        train(
            traindata, validdata,
            encoder0, decoder0, encoder1, decoder1,
            reg_encoder, head,
            criterion_t0=nn.MSELoss(), criterion_c=nn.MSELoss(),
            criterion_t1=nn.MSELoss(), criterion_r=nn.L1Loss(),
            max_seq_len=20,
            mu_t0=0.01, mu_c=0.01, mu_t1=0.01,
            dropout_p=0.15, early_stop=False, patience_num=15,
            lr=1e-4, weight_decay=0.01, op_type=torch.optim.AdamW,
            epoch=1, model_save='best_mctn.pt')


    all_in_one_train(trainprocess, allmodules)
 """
""" def test_integration3():
  from training_structures.gradient_blend import train, test # noqa
  from unimodals.common_models import GRU, MLP, Transformer # noqa
  from datasets.affect.get_data import get_dataloader # noqa
  from fusions.common_fusions import Concat # noqa


  # mosi_data.pkl, mosei_senti_data.pkl
  # mosi_raw.pkl, mosei_senti_data.pkl, sarcasm.pkl, humor.pkl
  # raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
  traindata, validdata, test_robust = \
      get_dataloader(DATA_PATH+'mosi_raw.pkl',
                    task='classification', robust_test=False, max_pad=True)

  # mosi/mosei
  encoders = [Transformer(35, 70).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
              Transformer(74, 150).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
              Transformer(300, 600).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
  head = MLP(820, 512, 2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

  unimodal_heads = [MLP(70, 32, 2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), MLP(
      150, 64, 2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), MLP(600, 256, 2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]

  # humor/sarcasm
  # encoders=[Transformer(371,700).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), \
  #     Transformer(81,150).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),\
  #     Transformer(300,600).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
  # head=MLP(1450,512,2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

  # unimodal_heads=[MLP(700,512,2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),MLP(150,64,2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),MLP(600,256,2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]

  fusion = Concat().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

  # training_structures.gradient_blend.criterion = nn.L1Loss()

  train(encoders, head, unimodal_heads, fusion, traindata, validdata, num_epoch=1, gb_epoch=1, lr=1e-3, AUPRC=True,
        classification=True, optimtype=torch.optim.AdamW, savedir='mosi_best_gb.pt', weight_decay=0.1, finetune_epoch=1)

  print("Testing:")
  #model = torch.load('mosi_besf_gb.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

   """
