import torch
from torch import nn
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import MLP, GRU # noqa
from datasets.mimic.get_data import get_dataloader # noqa
from training_structures.unimodal import train, test # noqa

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    1, imputed_path='/home/pliang/yiwei/im.pk')
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
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(encoder, head, testdata, dataset='mimic 1', auprc=False, modalnum=modalnum)
