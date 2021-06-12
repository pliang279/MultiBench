import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn
from fusions.mult import MULTModel
from training_structures.unimodal import train, test
from unimodals.common_models import MLP

from datasets.affect.get_data import get_dataloader


traindata, validdata, testdata = get_dataloader('../affect/processed/mosei_senti_data.pkl')

#mosi
# encoders=GRU(325,512,dropout=True,has_padding=True).cuda()
# head=MLP(512,256, 1).cuda()

#mosei
encoders = MULTModel(3).cuda()
head = nn.Identity()

train(encoders, head, traindata, validdata, 1000, True, True, task="regression", optimtype=torch.optim.AdamW, lr=1e-5,
      save='mosei_mult_best.pt', weight_decay=0.01,criterion=torch.nn.L1Loss(), regularization=False)


print("Testing:")
model = torch.load('mosei_mult_best.pt').cuda()
test(model, testdata, True, torch.nn.L1Loss(), "regression")

