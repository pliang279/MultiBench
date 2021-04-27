import sys
import os
sys.path.append(os.getcwd())

import torch

from training_structures.Contrastive_Learning import train, test
from fusions.common_fusions import Concat
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import GRU, MLP


traindata, validdata, testdata = get_dataloader(
    '../affect/processed/mosi_data.pkl',100,task="classification")

'''
encoders=[GRU(20,50,dropout=True,has_padding=True).cuda(), \
    GRU(5,15,dropout=True,has_padding=True).cuda(),\
    GRU(300,600,dropout=True,has_padding=True).cuda()]
'''
encoders=[GRU(20,40,dropout=True,has_padding=True).cuda(), \
    GRU(5,40,dropout=True,has_padding=True).cuda()]
head=MLP(80,40,7).cuda()
fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,1000,True,True, \
    optimtype=torch.optim.AdamW,lr=1e-4,save='best_contrast.pt', \
    weight_decay=0.01)

print("Testing:")
model=torch.load('best_contrast.pt').cuda()
test(model,testdata,True,)