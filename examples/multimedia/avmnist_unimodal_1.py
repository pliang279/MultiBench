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
# encoders=[LeNet(1,channels,3).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),LeNet(1,channels,5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
encoder = LeNet(1, channels, 5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = MLP(channels*32, 100, 10).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


train(encoder, head, traindata, validdata, 20, optimtype=torch.optim.SGD,
      lr=0.1, weight_decay=0.0001, modalnum=modalnum)

print("Testing:")
encoder = torch.load('encoder.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = torch.load('head.pt')
test(encoder, head, testdata, modalnum=modalnum, no_robust=True)
