import sys
import os
from torch import nn
sys.path.append(os.getcwd())
from training_structures.unimodal import train, test
from fusions.common_fusions import Concat
from datasets.enrico.get_data import get_dataloader
from unimodals.common_models import VGG16, VGG16Slim,DAN,Linear,MLP, VGG11Slim, VGG11Pruned, VGG16Pruned

from private_test_scripts.all_in_one import all_in_one_train,all_in_one_test
from memory_profiler import memory_usage

import torch

dls, weights = get_dataloader('datasets/enrico/dataset')
traindata, validdata, testdata = dls
modalnum = 0
encoder=VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda()
head = Linear(16, 20).cuda()
# head = MLP(16, 32, 20, dropout=False).cuda()

allmodules = [encoder, head]

def trainprocess():
    train(encoder,head,traindata,validdata,50,optimtype=torch.optim.Adam,lr=0.0001,weight_decay=0,modalnum=modalnum)

all_in_one_train(trainprocess, allmodules)


model=torch.load('best.pt').cuda()
test(encoder,head , testdata, dataset='enrico', modalnum=modalnum)
