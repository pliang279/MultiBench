import torch
from torch import nn
from unimodals.common_models import MLP, GRU
from datasets.mimic.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.gradient_blend import train, test
import sys
import os
sys.path.append(os.getcwd())

filename = 'bbest10.pt'

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    -1, imputed_path='datasets/mimic/im.pk')

# build encoders, head and fusion layer
encoders = [MLP(5, 10, 10).cuda(), GRU(12, 30, flatten=True).cuda()]
head = MLP(730, 40, 6).cuda()
fusion = Concat().cuda()
unimodal_heads = [MLP(10, 20, 6).cuda(), MLP(720, 40, 6).cuda()]

# train
train(encoders, head, unimodal_heads, fusion, traindata,
      validdata, 300, lr=0.005, AUPRC=False, savedir=filename)

# test
print("Testing: ")
model = torch.load(filename).cuda()
test(model, testdata, auprc=False)
