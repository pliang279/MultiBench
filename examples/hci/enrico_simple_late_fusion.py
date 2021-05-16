import sys
import os
sys.path.append(os.getcwd())
from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat
from datasets.enrico.get_data import get_dataloader
from unimodals.common_models import VGG16, VGG16Slim,DAN,Linear

import torch

traindata, validdata, testdata = get_dataloader('datasets/enrico/dataset')
channels=3
# encoders=[VGG16Slim(64).cuda(), DAN(4, 16, dropout=True, dropoutp=0.25).cuda(), DAN(28, 16, dropout=True, dropoutp=0.25).cuda()]
# head = Linear(96, 20)
encoders=[VGG16Slim(128).cuda()]
head = Linear(128, 20)

fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,100,optimtype=torch.optim.Adam,lr=0.001,weight_decay=0.0)

print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata)


