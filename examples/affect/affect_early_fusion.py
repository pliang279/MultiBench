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
# traindata, validdata, testdata = get_dataloader('/home/pliang/multibench/affect/pack/mosi/mosi_raw.pkl', robust_test=False)
traindata, validdata, testdata = get_dataloader('/home/pliang/multibench/affect/pack/mosi/mosi_raw.pkl', robust_test=False, max_pad=True, data_type='mosi', max_seq_len=50)

# mosi/mosei
encoders = [Identity().cuda(),Identity().cuda(),Identity().cuda()]
head = Sequential(GRU(409,512,dropout=True,has_padding=False, batch_first=True),MLP(512, 512, 1)).cuda()

# humor/sarcasm
# encoders = [Identity().cuda(),Identity().cuda(),Identity().cuda()]
# head = Sequential(GRU(752, 1128, dropout=True, has_padding=False, batch_first=True), MLP(1128, 512, 1)).cuda()

fusion = ConcatEarly().cuda()

train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW,is_packed=False, lr=1e-3, save='mosi_ef_r0.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('mosi_ef_r0.pt').cuda()
test(model, testdata,'affect', is_packed=False, criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)
