import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from training_structures.unimodal import train, test


modalnum = 1
traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist')
channels = 6
# encoders=[LeNet(1,channels,3).cuda(),LeNet(1,channels,5).cuda()]
encoder = LeNet(1, channels, 5).cuda()
head = MLP(channels*32, 100, 10).cuda()


train(encoder, head, traindata, validdata, 20, optimtype=torch.optim.SGD,
      lr=0.1, weight_decay=0.0001, modalnum=modalnum)

print("Testing:")
encoder = torch.load('encoder.pt').cuda()
head = torch.load('head.pt')
test(encoder, head, testdata, modalnum=modalnum, no_robust=True)
