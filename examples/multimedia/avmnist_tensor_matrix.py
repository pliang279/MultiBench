import sys
import os
sys.path.append(os.getcwd())

from unimodals.common_models import LeNet, MLP, Constant
from fusions.common_fusions import Concat, MultiplicativeInteractions2Modal
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from training_structures.Supervised_Learning import train, test

traindata, validdata, testdata = get_dataloader(
    '/home/pliang/yiwei/avmnist/_MFAS/avmnist')
channels = 3
encoders = [LeNet(1, channels, 3).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), LeNet(1, channels, 5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
head = MLP(channels*32, 100, 10).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

fusion = MultiplicativeInteractions2Modal(
    [channels*8, channels*32], channels*32, 'matrix', True).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# fusion=MultiplicativeInteractions2Modal([channels*32,channels*8],channels*32,'vector',True,flip=True).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

train(encoders, fusion, head, traindata, validdata, 100,
      optimtype=torch.optim.SGD, lr=0.01, weight_decay=0.0001)

print("Testing:")
model = torch.load('best.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
test(model, testdata, no_robust=True)
