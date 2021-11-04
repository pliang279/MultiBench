from unimodals.common_models import LeNet, MLP, Constant, Linear
import torch
from torch import nn
from datasets.avmnist.get_data_robust import get_dataloader
from fusions.common_fusions import Concat
from training_structures.unimodal import train, test
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

traindata, validdata, testdata, robustdata = get_dataloader(
    '../../../../yiwei/avmnist/_MFAS/avmnist')
channels = 3
encoders = LeNet(1, channels, 5).cuda()
head = Linear(channels*32, 10).cuda()
mn = 1

train(encoders, head, traindata, validdata, 100, optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001,
      modalnum=mn, save_encoder='avmnist_pretraining_encoder.pt', save_head='avmnist_pretraining_head.pt')

encoder = torch.load('avmnist_pretraining_encoder.pt').cuda()
head = torch.load('avmnist_pretraining_head.pt')
print("Testing:")
test(encoder, head, testdata, modalnum=mn)

print("Robustness testing:")
test(encoder, head, robustdata, modalnum=mn)

# Testing:
# acc: 0.4183
# Robustness testing:
# acc: 0.1019
