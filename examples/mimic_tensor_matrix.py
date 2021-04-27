import sys
import os
sys.path.append(os.getcwd())
from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import MultiplicativeInteractions2Modal
from datasets.mimic.get_data import get_dataloader
from unimodals.common_models import MLP, GRU
from torch import nn
import torch

#get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(-1, imputed_path='datasets/mimic/im.pk')

#build encoders, head and fusion layer
encoders = [MLP(5, 10, 10,dropout=True).cuda(), GRU(12, 30,dropout=True).cuda()]
head = MLP(100,40, 2, dropout=True).cuda()
fusion = MultiplicativeInteractions2Modal([10,720],100,'matrix',flatten=True)

#train
train(encoders, fusion, head, traindata, validdata, 20, auprc=True)

#test
print("Testing: ")
model = torch.load('best.pt').cuda()
test(model, testdata, auprc=True)
