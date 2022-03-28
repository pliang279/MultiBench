from unimodals.common_models import LeNet, MLP, Constant
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.gradient_blend import train, test
import sys
import os
sys.path.append(os.getcwd())


filename = 'best3.pt'
traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist')
channels = 6
encoders = [LeNet(1, channels, 3).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), LeNet(1, channels, 5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
mult_head = MLP(channels*40, 100, 10).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
uni_head = [MLP(channels*8, 100, 10).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), MLP(channels*32, 100, 10).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]

fusion = Concat().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def trpr():
    train(encoders, mult_head, uni_head, fusion, traindata, validdata, 300,
          gb_epoch=10, optimtype=torch.optim.SGD, lr=0.01, savedir=filename)


all_in_one_train(trpr, [encoders[0], encoders[1],
                 mult_head, uni_head[0], uni_head[1], fusion])


print("Testing:")
model = torch.load(filename).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def tepr():
    test(model, testdata)


all_in_one_test(tepr, [model])
