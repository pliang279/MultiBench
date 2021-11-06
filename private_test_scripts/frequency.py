import torch
from torch import nn
from unimodals.common_models import MLP, GRU
from datasets.mimic.get_data import get_dataloader
from training_structures.unimodal import train, test
import sys
import os
sys.path.append(os.getcwd())

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    1, imputed_path='datasets/mimic/im.pk')
a = [0, 0, 0, 0, 0, 0]
total = 0
for j in testdata:
    for i in j[-1]:
        a[i] += 1
        total += 1
print(a)
print(total)
