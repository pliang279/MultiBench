import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat, MultiplicativeInteractions2Modal
from training_structures.Supervised_Learning import train, test


filename = 'bestmi.pt'
traindata, validdata, testdata = get_dataloader(
    '/home/pliang/yiwei/avmnist/_MFAS/avmnist')
channels = 6
encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
head = MLP(channels*40, 100, 10).cuda()

# fusion=Concat().cuda()
fusion = MultiplicativeInteractions2Modal(
    [channels*8, channels*32], channels*40, 'matrix')

train(encoders, fusion, head, traindata, validdata, 20,
      optimtype=torch.optim.SGD, lr=0.05, weight_decay=0.0001, save=filename)

print("Testing:")
model = torch.load(filename).cuda()
test(model, testdata, no_robust=True)
