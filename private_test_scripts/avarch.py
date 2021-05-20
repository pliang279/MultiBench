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


from private_test_scripts.all_in_one import all_in_one_train,all_in_one_test



traindata, validdata, testdata = get_dataloader('/data/yiwei/avmnist/_MFAS/avmnist',batch_size=32)
def trpr():
    train(['pretrained/avmnist/image_encoder.pt','pretrained/avmnist/audio_encoder.pt'],16,10,[(6,12,24),(6,12,24,48,96)],
        traindata,validdata,surr.SimpleRecurrentSurrogate().cuda(),(3,5,2),epochs=6)

all_in_one_train(trpr,[torch.load('pretrained/avmnist/image_encoder.pt'),torch.load('pretrained/avmnist/audio_encoder.pt'),surr.SimpleRecurrentSurrogate()])
"""
print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata)
"""

