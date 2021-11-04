import torch
from torch import nn
from unimodals.common_models import MLP, GRU
from datasets.mimic.get_data import get_dataloader
from training_structures.unimodal import train, test
import sys
import os
sys.path.append(os.getcwd())

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    1, imputed_path='datasets/mimic/im.pk')
modalnum = 0
# build encoders, head and fusion layer
#encoders = [MLP(5, 10, 10,dropout=False).cuda(), GRU(12, 30,dropout=False).cuda()]
encoder = MLP(5, 10, 10).cuda()
head = MLP(10, 40, 2, dropout=False).cuda()


# train
train(encoder, head, traindata, validdata, 20, auprc=False, modalnum=modalnum)

# test
print("Testing: ")
encoder = torch.load('encoder.pt').cuda()
head = torch.load('head.pt').cuda()
test(encoder, head, testdata, auprc=False, modalnum=modalnum)
