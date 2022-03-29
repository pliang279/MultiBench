import torch
import numpy as np

    
def test_sl():
    from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
    from training_structures.Supervised_Learning import train, test  # noqa
    from fusions.common_fusions import ConcatEarly  # noqa
    data = [torch.zeros([32, 50, 35]), torch.zeros([32, 50, 74]), torch.zeros([32, 50, 300]), torch.cat((torch.ones((16,1)),torch.zeros((16,1))),dim=0)]
    my_dataset = torch.utils.data.TensorDataset(*data)
    dl = torch.utils.data.DataLoader(my_dataset)

    encoders = [Identity().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), Identity().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), Identity().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
    head = Sequential(GRU(409, 512, dropout=True, has_padding=False,
                  batch_first=True, last_only=True), MLP(512, 512, 1)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    fusion = ConcatEarly().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    train(encoders, fusion, head, dl, dl, 1, task="regression", optimtype=torch.optim.AdamW,
      is_packed=False, lr=1e-3, save='mosi_ef_r0.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

    model = torch.load('mosi_ef_r0.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    test(model, dl, 'affect', is_packed=False,
     criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)

def test_sl2():
  from private_test_scripts.all_in_one import all_in_one_train # noqa
  from training_structures.gradient_blend import train, test # noqa
  from unimodals.common_models import GRU, MLP, Transformer # noqa
  from fusions.common_fusions import Concat # noqa
  data = [torch.zeros([32, 50, 35]), torch.zeros([32, 50, 74]), torch.zeros([32, 50, 300]), torch.cat((torch.ones((16,1)),torch.zeros((16,1))),dim=0).long()]
  my_dataset = torch.utils.data.TensorDataset(*data)
  dl = torch.utils.data.DataLoader(my_dataset, batch_size=32)

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

  train(encoders, head, unimodal_heads, fusion, dl, dl, num_epoch=1, gb_epoch=1, lr=1e-3, AUPRC=False,
        classification=True, optimtype=torch.optim.AdamW, savedir='mosi_best_gb.pt', weight_decay=0.1, finetune_epoch=1)

  # test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification')

def test_sl3():
  from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
  from training_structures.unimodal import train, test  # noqa
  from fusions.common_fusions import ConcatEarly  # noqa
  data = [torch.zeros([32, 50, 35]), torch.zeros([32, 50, 74]), torch.zeros([32, 50, 300]), torch.cat((torch.ones((16,1)),torch.zeros((16,1))),dim=0).long()]
  my_dataset = torch.utils.data.TensorDataset(*data)
  dl = torch.utils.data.DataLoader(my_dataset, batch_size=32)

  modality_num = 2

# mosi/mosei
  encoder = GRU(300, 600, dropout=True, has_padding=False,
                batch_first=True, last_only=True).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
  head = MLP(600, 512, 1).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


  train(encoder, head, dl, dl, 1, task="regression", optimtype=torch.optim.AdamW, lr=2e-3,
        weight_decay=0.01, criterion=torch.nn.L1Loss(), save_encoder='encoder.pt', save_head='head.pt', modalnum=modality_num)

  print("Testing:")
  encoder = torch.load('encoder.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
  head = torch.load('head.pt')
  test(encoder, head, dl, 'affect', criterion=torch.nn.L1Loss(),
      task="posneg-classification", modalnum=modality_num, no_robust=True)


def test_sl4():
  from private_test_scripts.all_in_one import all_in_one_train # noqa
  from training_structures.MCTN_Level2 import train, test # noqa
  from unimodals.common_models import GRU, MLP # noqa
  from fusions.MCTN import Encoder, Decoder # noqa
  import torch.nn as nn

  # Faking a dataloader with a list of lists, to test that training structures execute
  dl_faked = [[[torch.zeros((32,28,35)),torch.zeros((32,28,74)), torch.zeros((32,28,300))], [torch.zeros((32,)),torch.zeros((32,)),torch.zeros((32,))],  torch.cat((torch.ones((16,1)),torch.zeros((16,1))),dim=0).long(),torch.cat((torch.ones((16,1)),torch.zeros((16,1))),dim=0).long()]]


  max_seq = 20
  feature_dim = 300
  hidden_dim = 32

  encoder0 = Encoder(feature_dim, hidden_dim, n_layers=1, dropout=0.0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
  decoder0 = Decoder(hidden_dim, feature_dim, n_layers=1, dropout=0.0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
  encoder1 = Encoder(hidden_dim, hidden_dim, n_layers=1, dropout=0.0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
  decoder1 = Decoder(hidden_dim, feature_dim, n_layers=1, dropout=0.0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

  reg_encoder = nn.GRU(hidden_dim, 32).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
  head = MLP(32, 64, 1).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

  allmodules = [encoder0, decoder0, encoder1, decoder1, reg_encoder, head]


  def trainprocess():
      train(
          dl_faked, dl_faked,
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

  model = torch.load('best_mctn.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

  test(model, dl_faked, 'mosi', no_robust=True)


