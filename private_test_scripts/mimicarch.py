import sys
import os
sys.path.append(os.getcwd())
from training_structures.architecture_search import train
from fusions.common_fusions import Concat
from datasets.mimic.get_data import get_dataloader
from unimodals.common_models import LeNet,MLP,Constant,GRUWithLinear
from torch import nn
import torch
import utils.surrogate as surr

traindata, validdata, testdata = get_dataloader(-1, imputed_path='datasets/mimic/im.pk')

from private_test_scripts.all_in_one import all_in_one_train,all_in_one_test


def trpr():
    train(['pretrained/mimic/static_encoder_mortality.pt','pretrained/mimic/ts_encoder_mortality.pt'],16,6,[(5,10,10),(288,720,360)],
        traindata,validdata,surr.SimpleRecurrentSurrogate().cuda(),(3,3,2),epochs=6)
all_in_one_train(trpr,[torch.load('pretrained/mimic/static_encoder_mortality.pt'),torch.load('pretrained/mimic/ts_encoder_mortality.pt'),surr.SimpleRecurrentSurrogate()])

"""
print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata)
"""

