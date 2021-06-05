import sys
import os
sys.path.append(os.getcwd())
from training_structures.Contrastive_Learning import train, test
from fusions.common_fusions import Concat
from datasets.avmnist.get_data import get_dataloader
from unimodals.common_models import LeNet,MLP,Constant
from torch import nn
from utils.helper_modules import Sequential2
import torch

traindata, validdata, testdata = get_dataloader('/home/pliang/yiwei/avmnist/_MFAS/avmnist',batch_size=20)
channels=6
encoders=[Sequential2(LeNet(1,channels,3),nn.Linear(channels*8,channels*32)).cuda(),LeNet(1,channels,5).cuda()]
head=MLP(channels*64,100,10).cuda()
refiner=MLP(channels*64,1000,13328)
fusion=Concat().cuda()

train(encoders,fusion,head,refiner,traindata,validdata,25,task='classification',optimtype=torch.optim.SGD,lr=0.001,criterion=torch.nn.CrossEntropyLoss())

print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata,criterion=torch.nn.CrossEntropyLoss(),task='classification')


