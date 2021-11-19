from unimodals.common_models import LeNet, MLP, Constant
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Simple_Late_Fusion import train, test
import sys
import os
sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist')
channels = 6
encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
head = MLP(channels*40, 100, 10).cuda()

fusion = Concat().cuda()


def trainprocess():
    train(encoders, fusion, head, traindata, validdata, 25,
          optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001)


# all_in_one_train(trainprocess,[encoders[0],encoders[1],head,fusion])
print("Testing:")

model = torch.load('best.pt').cuda()


def testprocess():
    test(model, testdata)


all_in_one_test(testprocess, [model])
