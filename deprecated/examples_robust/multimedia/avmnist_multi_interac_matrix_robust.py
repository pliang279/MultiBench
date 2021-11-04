import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat, MultiplicativeInteractions2Modal
from datasets.avmnist.get_data_robust import get_dataloader
from unimodals.common_models import LeNet,MLP,Constant
from torch import nn
import torch

filename='avmnist_multi_interac_matrix_best.pt'
traindata, validdata, testdata, robustdata = get_dataloader('../../../../yiwei/avmnist/_MFAS/avmnist')
channels=6
encoders=[LeNet(1,channels,3).cuda(),LeNet(1,channels,5).cuda()]
head=MLP(channels*40,100,10).cuda()

#fusion=Concat().cuda()
fusion = MultiplicativeInteractions2Modal([channels*8,channels*32],channels*40,'matrix')

train(encoders,fusion,head,traindata,validdata,20,optimtype=torch.optim.SGD,lr=0.05,weight_decay=0.0001,save=filename)

model=torch.load(filename).cuda()
print("Testing:")
test(model,testdata)

print("Robustness testing:")
test(model,testdata)
