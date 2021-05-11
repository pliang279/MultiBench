import sys
import os
sys.path.append(os.getcwd())
from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat
from datasets.enrico.get_data import get_dataloader
from unimodals.common_models import VGG16,DAN,MLP
from torch import nn
import torch
from torchvision import models as tmodels

traindata, validdata, testdata = get_dataloader('datasets/enrico/dataset')
channels=3
encoders=[VGG16(8).cuda(), DAN(4, 8, dropout=True, dropoutp=0.25).cuda(), DAN(28, 8, dropout=True, dropoutp=0.25).cuda()]
head=MLP(24,32,20).cuda()

fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,100,optimtype=torch.optim.Adam,lr=0.0001,weight_decay=0.0)

print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata)


