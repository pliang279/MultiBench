import sys
import os
sys.path.append(os.getcwd())
from training_structures.gradient_blend import train, test
from fusions.common_fusions import Concat
from datasets.mimic.get_data import get_dataloader
from unimodals.common_models import MLP, GRU
from torch import nn
import torch

#get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(7, imputed_path='datasets/mimic/im.pk')

#build encoders, head and fusion layer
encoders = [MLP(5, 10, 10).cuda(), GRU(12, 30, flatten=True).cuda()]
head = MLP(730, 40, 2).cuda()
fusion = Concat().cuda()
unimodal_heads=[MLP(10,20,2).cuda(),MLP(720,40,2).cuda()]

#train
train(encoders, head, unimodal_heads, fusion, traindata, validdata, 300, lr=0.005, AUPRC=True)

#test
print("Testing: ")
model = torch.load('best.pt').cuda()
test(model, testdata, auprc=True)
