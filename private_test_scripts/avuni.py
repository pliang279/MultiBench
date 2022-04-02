from unimodals.common_models import LeNet, MLP, Constant
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from training_structures.unimodal import train, test
import sys
import os
sys.path.append(os.getcwd())

modalnum = 1
traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist')
channels = 6
# encoders=[LeNet(1,channels,3).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),LeNet(1,channels,5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
encoder = LeNet(1, channels, 5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = MLP(channels*32, 100, 10).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def trainprocess():
    train(encoder, head, traindata, validdata, 20, optimtype=torch.optim.SGD,
          lr=0.01, weight_decay=0.0001, modalnum=modalnum)


all_in_one_train(trainprocess, [encoder, head])

print("Testing:")
encoder = torch.load('encoder.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = torch.load('head.pt')


def testprocess():
    test(encoder, head, testdata, modalnum=modalnum)


all_in_one_test(testprocess, [encoder, head])
