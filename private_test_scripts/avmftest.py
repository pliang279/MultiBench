import sys
import os
sys.path.append(os.getcwd())
from training_structures.architecture_search import train,test
from fusions.common_fusions import Concat
from datasets.avmnist.get_data import get_dataloader
from unimodals.common_models import LeNet,MLP,Constant
from torch import nn
import torch
import utils.surrogate as surr
from private_test_scripts.all_in_one import all_in_one_train,all_in_one_test

traindata, validdata, testdata = get_dataloader('/data/yiwei/avmnist/_MFAS/avmnist',batch_size=32)
model = torch.load('temp/best.pt').cuda()
def testprocess():
    test(model,testdata)
all_in_one_test(testprocess,[model])
