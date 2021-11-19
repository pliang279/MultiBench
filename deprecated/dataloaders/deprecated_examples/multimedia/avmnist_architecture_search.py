from unimodals.common_models import LeNet, MLP, Constant
import utils.surrogate as surr
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.architecture_search import train
import sys
import os
sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist', batch_size=32)

s_data = train(['pretrained/avmnist/image_encoder.pt', 'pretrained/avmnist/audio_encoder.pt'], 16, 10, [(6, 12, 24), (6, 12, 24, 48, 96)],
               traindata, validdata, surr.SimpleRecurrentSurrogate().cuda(), (3, 5, 2), epochs=6)

"""
print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata)
"""
