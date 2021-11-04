from robustness.all_in_one import stocks_train, stocks_test
from get_data_robust import get_dataloader
from training_structures.Simple_Late_Fusion import train, test
from unimodals.common_models import LSTM, Identity
from fusions.common_fusions import ConcatWithLinear
from torch import nn
import torch.nn.functional as F
import torch
import pmdarima
import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

sys.path.append('/home/pliang/multibench/MultiBench/datasets/stocks')


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(
    stocks, stocks, [args.target_stock])

n_modalities = len(train_loader.dataset[0]) - 1
encoders = [LSTM(1, 16).cuda() for _ in range(n_modalities)]
fusion = ConcatWithLinear(n_modalities * 16, 1).cuda()
head = Identity().cuda()
allmodules = [*encoders, fusion, head]

num_training = 5


def trainprocess(filename):
    train(encoders, fusion, head, train_loader, val_loader, total_epochs=4,
          task='regression', optimtype=torch.optim.Adam, criterion=nn.MSELoss(), save=filename)


filenames = stocks_train(num_training, trainprocess, 'stocks_late_fusion_best')


def testprocess(model, noise_level):
    return test(model, test_loader[noise_level], task='regression', criterion=nn.MSELoss())


stocks_test(num_training, filenames, len(test_loader), testprocess)
