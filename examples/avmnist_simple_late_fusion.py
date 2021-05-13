import sys
import os
sys.path.append(os.getcwd())
from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat
from datasets.avmnist.get_data import get_dataloader
from unimodals.common_models import LeNet,MLP,Constant
from torch import nn
import torch

traindata, validdata, testdata = get_dataloader('/data/yiwei/avmnist/_MFAS/avmnist')
channels=3
encoders=[LeNet(1,channels,3).cuda(),LeNet(1,channels,5).cuda()]
head=MLP(channels*40,100,10).cuda()

fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,30,optimtype=torch.optim.SGD,lr=0.1,weight_decay=0.0001)

print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata)


