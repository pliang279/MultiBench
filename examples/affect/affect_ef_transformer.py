import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch
os.environ['CUDA_VISIBLE_DEVICES']='0'
from fusions.common_fusions import ConcatEarly
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import Transformer, MLP,Sequential,Identity

from training_structures.Supervised_Learning import train, test

from private_test_scripts.all_in_one import all_in_one_train

# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
traindata, validdata, testdata = get_dataloader('/home/paul/MultiBench/mosei_senti_data.pkl', robust_test=False)

# mosi/mosei
encoders = [Identity().cuda(),Identity().cuda(),Identity().cuda()]
head = Sequential(Transformer(409, 300).cuda(),MLP(300, 128, 1)).cuda()

# humor/sarcasm
# encoders = GRU(752, 1128, dropout=True, has_padding=True).cuda()
# head = MLP(1128, 512, 1).cuda()

# all_modules = [*encoders, head]

fusion = ConcatEarly().cuda()

train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, is_packed=True, early_stop=True,
        lr=1e-4, save='mosi_ef_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

# all_in_one_train(trainprocess, all_modules)

print("Testing:")
model = torch.load('mosi_ef_best.pt').cuda()
test(model, testdata, 'affect', is_packed=True, criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)
