from unimodals.common_models import LeNet, MLP, Constant
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
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


def trpr():
    train(['pretrained/avmnist/image_encoder.pt', 'pretrained/avmnist/audio_encoder.pt'], 16, 10, [(6, 12, 24), (6, 12, 24, 48, 96)],
          traindata, validdata, surr.SimpleRecurrentSurrogate().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), (3, 5, 2), epochs=6)


all_in_one_train(trpr, [torch.load('pretrained/avmnist/image_encoder.pt'), torch.load(
    'pretrained/avmnist/audio_encoder.pt'), surr.SimpleRecurrentSurrogate()])
"""
print("Testing:")
model=torch.load('best.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
test(model,testdata)
"""
