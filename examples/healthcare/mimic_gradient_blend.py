import torch
from torch import nn
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import MLP, GRU  # noqa
from datasets.mimic.get_data import get_dataloader  # noqa
from fusions.common_fusions import Concat  # noqa
from training_structures.gradient_blend import train, test  # noqa


filename = 'bbest10.pt'

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    -1, imputed_path='/home/pliang/yiwei/im.pk')

# build encoders, head and fusion layer
encoders = [MLP(5, 10, 10).cuda(), GRU(
    12, 30, flatten=True, batch_first=True).cuda()]
head = MLP(730, 40, 6).cuda()
fusion = Concat().cuda()
unimodal_heads = [MLP(10, 20, 6).cuda(), MLP(720, 40, 6).cuda()]

# train
train(encoders, head, unimodal_heads, fusion, traindata,
      validdata, 300, lr=0.005, AUPRC=False, savedir=filename)

# test
print("Testing: ")
model = torch.load(filename).cuda()
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(model, testdata, dataset='mimic mortality', auprc=False)
