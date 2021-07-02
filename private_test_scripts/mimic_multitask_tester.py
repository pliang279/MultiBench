import sys
import os
sys.path.append(os.getcwd())
from private_test_scripts.Augmented_Multitask1 import train, test
from fusions.common_fusions import Concat
from datasets.mimic.multitask import get_dataloader
from unimodals.common_models import MLP, GRU
from torch import nn
import torch

#get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(imputed_path='/home/pliang/yiwei/im.pk')


#test
print("Testing: ")
model = torch.load('best2.pt').cuda()
test(model, testdata)
