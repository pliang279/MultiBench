import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch

from fusions.common_fusions import Concat
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import GRU, MLP

from training_structures.Supervised_Learning import train, test

from private_test_scripts.all_in_one import all_in_one_train

# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
traindata, validdata, test_robust = \
    get_dataloader('/home/van/backup/pack/mosi/mosi_raw.pkl', robust_test=False)

# mosi/mosei
encoders=[GRU(35,64,dropout=True,has_padding=True).cuda(), \
    GRU(74,128,dropout=True,has_padding=True).cuda(),\
    GRU(300,512,dropout=True,has_padding=True).cuda()]
head=MLP(704,512,1).cuda()

# encoders=[GRU(20,40,dropout=True,has_padding=True).cuda(), \
#     GRU(5, 20,dropout=True,has_padding=True).cuda(),\
#     GRU(300, 600,dropout=True,has_padding=True).cuda()]
# head=MLP(660,512,1, dropoutp=True).cuda()

# humor/sarcasm
# encoders=[GRU(371,512,dropout=True,has_padding=True).cuda(), \
#     GRU(81,256,dropout=True,has_padding=True).cuda(),\
#     GRU(300,600,dropout=True,has_padding=True).cuda()]
# head=MLP(1368,512,1).cuda()

all_modules = [*encoders, head]

fusion = Concat().cuda()

def trainprocess():
    train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, early_stop=False, is_packed=True, lr=1e-3, save='mosi_lf_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())


all_in_one_train(trainprocess, all_modules)

print("Testing:")
model = torch.load('mosi_lf_best.pt').cuda()

test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification')
