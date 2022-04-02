from unimodals.common_models import LeNet, MLP, Constant
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
import torch
from utils.helper_modules import Sequential2
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Contrastive_Learning import train, test
import sys
import os
sys.path.append(os.getcwd())
traindata, validdata, testdata = get_dataloader(
    '/home/pliang/yiwei/avmnist/_MFAS/avmnist', batch_size=20)
channels = 6
encoders = [Sequential2(LeNet(1, channels, 3), nn.Linear(
    channels*8, channels*32)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), LeNet(1, channels, 5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
head = MLP(channels*64, 100, 10).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
refiner = MLP(channels*64, 1000, 13328)
fusion = Concat().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def trpr():
    train(encoders, fusion, head, refiner, traindata, validdata, 15, task='classification',
          optimtype=torch.optim.SGD, lr=0.005, criterion=torch.nn.CrossEntropyLoss())


all_in_one_train(trpr, encoders+[fusion, head, refiner])
print("Testing:")
model = torch.load('best.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def tepr():
    test(model, testdata, criterion=torch.nn.CrossEntropyLoss(), task='classification')


all_in_one_test(tepr, encoders+[fusion, head])
