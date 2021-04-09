import sys
import os
sys.path.append(os.getcwd())
from training_structures.architecture_search import train
from fusions.common_fusions import Concat
from datasets.avmnist.get_data import get_dataloader
from unimodals.common_models import LeNet,MLP,Constant
from torch import nn
import torch
import utils.surrogate as surr

traindata, validdata, testdata = get_dataloader('/data/yiwei/avmnist/_MFAS/avmnist')

train(['pretrained/avmnist/image_encoder.pt','pretrained/avmnist/audio_encoder.pt'],16,10,[(6,12,24),(6,12,24,48,96)],
        traindata,validdata,surr.SimpleRecurrentSurrogate().cuda(),(3,5,2))

"""
print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata)
"""

