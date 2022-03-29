import sys
import os
import torch
from torch import nn

sys.path.append(os.getcwd())

from unimodals.common_models import MLP, GRU # noqa
from datasets.mimic.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa
from training_structures.Supervised_Learning import train, test # noqa


# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    7, imputed_path='/home/pliang/yiwei/im.pk')

# build encoders, head and fusion layer
encoders = [MLP(5, 10, 10, dropout=False).cuda(), GRU(
    12, 30, dropout=False, batch_first=True).cuda()]
head = MLP(730, 40, 2, dropout=False).cuda()
fusion = Concat().cuda()

# train
train(encoders, fusion, head, traindata, validdata, 20, auprc=True)

# test
print("Testing: ")
model = torch.load('best.pt').cuda()
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(model, testdata, dataset='mimic 7', auprc=True)
