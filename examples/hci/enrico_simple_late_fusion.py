import sys
import os
from torch import nn
sys.path.append(os.getcwd())
from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat
from datasets.enrico.get_data import get_dataloader
from unimodals.common_models import VGG16, VGG16Slim,DAN,Linear

import torch

dls, weights = get_dataloader('datasets/enrico/dataset')
traindata, validdata, testdata = dls
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).cuda()
# encoders=[VGG16Slim(64).cuda(), DAN(4, 16, dropout=True, dropoutp=0.25).cuda(), DAN(28, 16, dropout=True, dropoutp=0.25).cuda()]
# head = Linear(96, 20)
encoders=[VGG16Slim(16, dropout=True, dropoutp=0.2).cuda()]
head = Linear(16, 20)

fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,100,optimtype=torch.optim.Adam,lr=0.0001,weight_decay=0,criterion=criterion)

print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata)


