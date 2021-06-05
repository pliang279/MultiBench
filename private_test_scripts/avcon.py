import sys
import os
sys.path.append(os.getcwd())
from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat
from datasets.avmnist.get_data import get_dataloader
from unimodals.common_models import LeNet,MLP,Constant
from torch import nn
from utils.helper_modules import Sequential2
import torch
from private_test_scripts.all_in_one import all_in_one_train,all_in_one_test
traindata, validdata, testdata = get_dataloader('/home/pliang/yiwei/avmnist/_MFAS/avmnist')
channels=6
encoders=[Sequential2(LeNet(1,channels,3),nn.Linear(channels*8,channels*32)).cuda(),LeNet(1,channels,5).cuda()]
head=MLP(channels*64,100,10).cuda()
refiner=MLP(channels*64,1000,13328)
fusion=Concat().cuda()
def trpr():
    train(encoders,fusion,head,traindata,validdata,15,optimtype=torch.optim.SGD,lr=0.1)
all_in_one_test(trpr,self.encoders+[self.fusion,self.head,self.refiner])
print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata)


