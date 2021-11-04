import torch
from torch import nn
from unimodals.common_models import MLP, GRU
from datasets.mimic.get_data import get_dataloader
from fusions.common_fusions import LowRankTensorFusion
from training_structures.Simple_Late_Fusion import train, test
import sys
import os
sys.path.append(os.getcwd())

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    1, imputed_path='datasets/mimic/im.pk')

# build encoders, head and fusion layer
encoders = [MLP(5, 10, 10, dropout=False).cuda(),
            GRU(12, 30, dropout=False).cuda()]
head = MLP(100, 40, 2, dropout=False).cuda()
fusion = LowRankTensorFusion([10, 720], 100, 40).cuda()

# train
train(encoders, fusion, head, traindata, validdata, 50, auprc=True)

# test
print("Testing: ")
model = torch.load('best.pt').cuda()
test(model, testdata, auprc=True)
