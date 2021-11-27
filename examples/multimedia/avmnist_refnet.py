import sys
import os
sys.path.append(os.getcwd())

from unimodals.common_models import LeNet, MLP, Constant
from objective_functions.objectives_for_supervised_learning import RefNet_objective
import torch
from utils.helper_modules import Sequential2
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test

traindata, validdata, testdata = get_dataloader(
    '/home/pliang/yiwei/avmnist/_MFAS/avmnist', batch_size=20)
channels = 6
encoders = [Sequential2(LeNet(1, channels, 3), nn.Linear(
    channels*8, channels*32)).cuda(), LeNet(1, channels, 5).cuda()]
head = MLP(channels*64, 100, 10).cuda()
refiner = MLP(channels*64, 1000, 13328).cuda()
fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 15, [
      refiner], optimtype=torch.optim.SGD, lr=0.005, objective=RefNet_objective(0.1), objective_args_dict={'refiner': refiner})

print("Testing:")
model = torch.load('best.pt').cuda()
test(model, testdata, no_robust=True)
