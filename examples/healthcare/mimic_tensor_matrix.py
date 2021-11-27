import sys
import os
import torch
from torch import nn

sys.path.append(os.getcwd())

from unimodals.common_models import MLP, GRU # noqa
from datasets.mimic.get_data import get_dataloader # noqa
from fusions.common_fusions import MultiplicativeInteractions2Modal # noqa
from training_structures.Supervised_Learning import train, test # noqa


filename = 'besttensor.pt'

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    1, imputed_path='/home/paul/yiwei/im.pk')
# build encoders, head and fusion layer
encoders = [MLP(5, 10, 10, dropout=False).cuda(), GRU(
    12, 30, dropout=False, batch_first=True).cuda()]
head = MLP(100, 40, 2, dropout=False).cuda()
fusion = MultiplicativeInteractions2Modal(
    [10, 720], 100, 'matrix', flatten=True)

# train
train(encoders, fusion, head, traindata,
      validdata, 20, auprc=True, save=filename)

# test
print("Testing: ")
model = torch.load(filename).cuda()
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(model, testdata, dataset='mimic 1', auprc=True)
