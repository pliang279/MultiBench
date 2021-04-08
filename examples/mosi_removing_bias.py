import sys
import os
sys.path.append(os.getcwd())

import torch

from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import GRU, MLP


traindata, validdata, testdata = get_dataloader('../affect/processed/mosi_data.pkl')
'''
encoders=[GRU(20,50,dropout=True,has_padding=True).cuda(), \
    GRU(5,15,dropout=True,has_padding=True).cuda(),\
    GRU(300,600,dropout=True,has_padding=True).cuda()]
head=MLP(665,300,1).cuda()
fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,1000,True,True, \
    task="regression",optimtype=torch.optim.AdamW,lr=1e-4,save='best2.pt', \
    weight_decay=0.01,criterion=torch.nn.MSELoss(),regularization=False)
'''
print("Testing:")
model=torch.load('best2.pt').cuda()
test(model,testdata,True,torch.nn.MSELoss(),"regression",)