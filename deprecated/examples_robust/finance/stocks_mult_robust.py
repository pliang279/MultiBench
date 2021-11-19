from robustness.all_in_one import stocks_train, stocks_test
from training_structures.unimodal import train, test
from get_data import Grouping
from get_data_robust import get_dataloader
from fusions.mult import MULTModel
from unimodals.common_models import Identity
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
    stocks, stocks, [args.target_stock], modality_first=False)

n_modalities = 3
grouping = Grouping(n_modalities)

# Get n_features for each group
n_features = [x.size(-1) for x in grouping(next(iter(train_loader))[0])]

model = nn.Sequential(grouping, MULTModel(n_modalities, n_features)).cuda()
identity = Identity()
allmodules = [model, identity]

num_training = 5


def trainprocess(filename_encoder, filename_head):
    train(model, identity, train_loader, val_loader, total_epochs=4, task='regression',
          optimtype=torch.optim.Adam, criterion=nn.MSELoss(), save_encoder=filename_encoder, save_head=filename_head)


filenames_encoder, filenames_head = stocks_train(
    num_training, trainprocess, 'stocks_mult', encoder=True)


def testprocess(encoder, head, noise_level):
    return test(encoder, head, test_loader[noise_level], task='regression', criterion=nn.MSELoss())


stocks_test(num_training, [filenames_encoder, filenames_head], len(
    test_loader), testprocess, encoder=True)
