import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.gradient_blend import train, test


filename = 'best3.pt'
traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist')
channels = 6
encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
mult_head = MLP(channels*40, 100, 10).cuda()
uni_head = [MLP(channels*8, 100, 10).cuda(), MLP(channels*32, 100, 10).cuda()]

fusion = Concat().cuda()

train(encoders, mult_head, uni_head, fusion, traindata, validdata, 300,
      gb_epoch=10, optimtype=torch.optim.SGD, lr=0.01, savedir=filename)

print("Testing:")
model = torch.load(filename).cuda()
test(model, testdata, no_robust=True)
