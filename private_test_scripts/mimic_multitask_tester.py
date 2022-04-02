import torch
from torch import nn
from unimodals.common_models import MLP, GRU
from datasets.mimic.multitask import get_dataloader
from fusions.common_fusions import Concat
from private_test_scripts.Augmented_Multitask1 import train, test
import sys
import os
sys.path.append(os.getcwd())

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    imputed_path='/home/pliang/yiwei/im.pk')


# test
print("Testing: ")
model = torch.load('best2.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
test(model, testdata)
