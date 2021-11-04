import torch
from torch import nn
from unimodals.common_models import MLP, GRU
from datasets.mimic.multitask import get_dataloader
from fusions.common_fusions import Concat
from private_test_scripts.Simple_Late_Fusion_Multitask import train, test
import sys
import os
sys.path.append(os.getcwd())

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    imputed_path='/home/yiwei/im.pk')

# build encoders, head and fusion layer
encoders = [MLP(5, 10, 10, dropout=False).cuda(),
            GRU(12, 30, dropout=False).cuda()]
head1 = MLP(730, 40, 6, dropout=False).cuda()
head2 = MLP(730, 40, 2, dropout=False).cuda()
fusion = Concat().cuda()

# train
train(encoders, fusion, head1, head2, traindata,
      validdata, 40, lr=0.01, save='best1.pt')

# test
print("Testing: ")
model = torch.load('best1.pt').cuda()
test(model, testdata)
