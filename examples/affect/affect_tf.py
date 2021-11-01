import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch

from fusions.common_fusions import Concat, TensorFusion
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import GRUWithLinear, MLP

from training_structures.Supervised_Learning import train, test

from private_test_scripts.all_in_one import all_in_one_train

# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
traindata, validdata, test_robust = \
    get_dataloader('/home/paul/MultiBench/mosi_raw.pkl', robust_test=False)

# mosi/mosei
encoders=[GRUWithLinear(35,64,4,dropout=True,has_padding=True).cuda(), \
    GRUWithLinear(74,128,19,dropout=True,has_padding=True).cuda(),\
    GRUWithLinear(300,512,79,dropout=True,has_padding=True).cuda()]
head=MLP(8000,512,1).cuda()

# humor/sarcasm
# encoders=[GRU(371,512,dropout=True,has_padding=True).cuda(), \
#     GRU(81,256,dropout=True,has_padding=True).cuda(),\
#     GRU(300,600,dropout=True,has_padding=True).cuda()]
# head=MLP(1368,512,1).cuda()

fusion = TensorFusion().cuda()

train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, early_stop=True, is_packed=True, lr=1e-3, save='mosi_tf_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('mosi_tf_best.pt').cuda()

test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=True, criterion=torch.nn.L1Loss(), task='posneg-classification')
