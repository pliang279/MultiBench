from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data_robust import get_dataloader
from fusions.common_fusions import LowRankTensorFusion
from training_structures.Simple_Late_Fusion import train, test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
filename = 'avmnist_low_rank_tensor_best.pt'
traindata, validdata, testdata, robustdata = get_dataloader(
    '../../../../yiwei/avmnist/_MFAS/avmnist')
channels = 6
encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
head = MLP(channels*20, 100, 10).cuda()

fusion = LowRankTensorFusion([channels*8, channels*32], channels*20, 40).cuda()

train(encoders, fusion, head, traindata, validdata, 30,
      optimtype=torch.optim.SGD, lr=0.05, weight_decay=0.0002, save=filename)

model = torch.load(filename).cuda()
print("Testing:")
test(model, testdata)

print("Robustness testing:")
test(model, testdata)
