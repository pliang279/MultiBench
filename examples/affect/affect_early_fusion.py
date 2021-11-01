import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch
os.environ['CUDA_VISIBLE_DEVICES']='1'
from fusions.common_fusions import ConcatEarly
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import GRU, MLP,Sequential,Identity

from training_structures.Supervised_Learning import train, test


# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
traindata, validdata, testdata = get_dataloader('/home/paul/MultiBench/mosi_raw.pkl')

# mosi/mosei
encoders = [Identity().cuda(),Identity().cuda(),Identity().cuda()]
head = Sequential(GRU(409,512,dropout=True,has_padding=True),MLP(512, 256, 1)).cuda()

# humor/sarcasm
# encoders = GRU(752, 1128, dropout=True, has_padding=True).cuda()
# head = MLP(1128, 512, 1).cuda()

fusion = ConcatEarly().cuda()

train(encoders, fusion, head, traindata, validdata, 1000, task="regression", optimtype=torch.optim.AdamW,is_packed=True,
        lr=1e-4, save='mosi_ef_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('mosi_ef_best.pt').cuda()
test(model, testdata,'affect', is_packed=True, criterion=torch.nn.L1Loss(), task="posneg-classification")
