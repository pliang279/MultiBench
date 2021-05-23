import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse
import numpy as np
import pmdarima
import torch
import torch.nn.functional as F
from torch import nn
from fusions.common_fusions import ConcatWithLinear
from fusions.finance.late_fusion import LateFusionTransformer
sys.path.append('/home/pliang/multibench/MultiBench/datasets/stocks')
from get_data_robust import get_dataloader
from training_structures.Simple_Late_Fusion import train, test
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test_robust


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(stocks, stocks, [args.target_stock])

n_modalities = train_loader.dataset[0][0].size(0)
encoders = [LateFusionTransformer(embed_dim=9).cuda() for _ in range(n_modalities)]
fusion = ConcatWithLinear(n_modalities * 9).cuda()
head = nn.Identity().cuda()
allmodules = [*encoders, fusion, head]

filename = 'stocks_late_fusion_transformer_best.pt'
def trainprocess():
    train(encoders, fusion, head, train_loader, val_loader, total_epochs=4,
          task='regression', optimtype=torch.optim.Adam, criterion=nn.MSELoss(), save=filename)
all_in_one_train(trainprocess, allmodules)

model = torch.load(filename).cuda()
def testprocess():
    test(model, test_loader, task='regression')
acc = []
print("Robustness testing:")
for noise_level in range(len(test_loader)):
    print("Noise level {}: ".format(noise_level/10))
    acc.append(all_in_one_test_robust(testprocess, [model], test_loader[noise_level]))

print("Accuracy of different noise levels:", acc)
