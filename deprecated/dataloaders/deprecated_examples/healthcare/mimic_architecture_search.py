from unimodals.common_models import LeNet, MLP, Constant, GRUWithLinear
import utils.surrogate as surr
import torch
from torch import nn
from datasets.mimic.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.architecture_search import train
import sys
import os
sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader(
    1, imputed_path='datasets/mimic/im.pk')


s_data = train(['pretrained/mimic/static_encoder_mortality.pt', 'pretrained/mimic/ts_encoder_mortality.pt'], 16, 2, [(5, 10, 10), (288, 720, 360)],
               traindata, validdata, surr.SimpleRecurrentSurrogate().cuda(), (3, 3, 2), epochs=6)

"""
print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata)
"""
