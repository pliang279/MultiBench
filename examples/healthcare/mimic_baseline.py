import sys
import os
sys.path.append(os.getcwd())
from training_structures.Supervised_Learning import train, test
from fusions.common_fusions import Concat
from datasets.mimic.get_data import get_dataloader
from unimodals.common_models import MLP, GRU
from torch import nn
import torch

#get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(7, imputed_path='/home/pliang/yiwei/im.pk')

#build encoders, head and fusion layer
encoders = [MLP(5, 10, 10,dropout=False).cuda(), GRU(12, 30,dropout=False).cuda()]
head = MLP(730, 40, 2, dropout=False).cuda()
fusion = Concat().cuda()

#train
train(encoders, fusion, head, traindata, validdata, 20, auprc=True)

#test
print("Testing: ")
model = torch.load('best.pt').cuda()
test(model, testdata, auprc=True)
