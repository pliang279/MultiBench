import sys
import os
sys.path.append(os.getcwd())

import torch

from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import GRU, MLP


traindata, validdata, testdata = get_dataloader('multibench/affect/processed/mosi_data.pkl')

encoders=[GRU(35,70,dropout=True).cuda(), GRU(5,10,dropout=True).cuda(), GRU(300,450,dropout=True).cuda()]
head=MLP(26500,1000,1).cuda()
fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,\
    100,optimtype=torch.optim.SGD,lr=0.01,weight_decay=0.002,\
        criterion=torch.nn.MSELoss(),regularization=True)

print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata)