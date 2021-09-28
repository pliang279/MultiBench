import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch

from fusions.common_fusions import Concat
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import GRU, MLP

from training_structures.Supervised_Learning import train, test


# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
traindata, validdata, testdata = \
    get_dataloader('/home/paul/MultiBench/mosei_senti_data.pkl', robust_test=False)

# mosi/mosei
encoders=[GRU(35,64,dropout=True,has_padding=True).cuda(), \
    GRU(74,128,dropout=True,has_padding=True).cuda(),\
    GRU(300,512,dropout=True,has_padding=True).cuda()]
head=MLP(704,512,1).cuda()

# humor/sarcasm
# encoders=[GRU(371,512,dropout=True,has_padding=True).cuda(), \
#     GRU(81,256,dropout=True,has_padding=True).cuda(),\
#     GRU(300,600,dropout=True,has_padding=True).cuda()]
# head=MLP(1368,512,1).cuda()

fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, is_packed=True,early_stop=True, lr=1e-4, save='mosei_lf_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())


print("Testing:")
model = torch.load('mosei_lf_best.pt').cuda()

test(model=model, test_dataloaders_all=testdata, dataset='mosi', is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification')



