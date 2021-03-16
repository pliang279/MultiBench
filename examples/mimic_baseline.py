import sys
import os
sys.path.append(os.getcwd())
from training_structures.Simple_Late_Fusion import train,test
from fusions.common_fusions import Concat
from datasets.mimic.get_data import get_dataloader
from unimodals.common_models import MLP_dropout,GRU_dropout
from torch import nn
import torch

#get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(7, imputed_path='datasets/mimic/im.pk')

#build encoders, head and fusion layer
encoders = [MLP_dropout(5, 10, 10).cuda(), GRU_dropout(12, 30).cuda()]
head = MLP_dropout(730, 40, 2).cuda()
fusion = Concat().cuda()

#train
train(encoders, fusion, head, traindata, validdata, 20, auprc=True)

#test
print("Testing: ")
model = torch.load('best.pt').cuda()
test(model, testdata, auprc=True)
