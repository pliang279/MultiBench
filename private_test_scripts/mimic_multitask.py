import sys
import os
sys.path.append(os.getcwd())
from private_test_scripts.Simple_Late_Fusion_Multitask import train, test
from fusions.common_fusions import Concat
from datasets.mimic.multitask import get_dataloader
from unimodals.common_models import MLP, GRU
from torch import nn
import torch

#get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(imputed_path='/home/pliang/yiwei/im.pk')

#build encoders, head and fusion layer
encoders = [MLP(5, 10, 10,dropout=False).cuda(), GRU(12, 30,dropout=False).cuda()]
head1 = MLP(730, 40, 6, dropout=False).cuda()
head2 = MLP(730, 40, 2, dropout=False).cuda()
fusion = Concat().cuda()

#train
train(encoders, fusion, head1,head2 , traindata, validdata, 80, lr=0.001)

#test
print("Testing: ")
model = torch.load('best.pt').cuda()
test(model, testdata)
