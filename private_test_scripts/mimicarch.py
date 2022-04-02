from unimodals.common_models import LeNet, MLP, Constant, GRUWithLinear
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
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
    -1, imputed_path='datasets/mimic/im.pk')


def trpr():
    train(['pretrained/mimic/static_encoder_mortality.pt', 'pretrained/mimic/ts_encoder_mortality.pt'], 16, 6, [(5, 10, 10), (288, 720, 360)],
          traindata, validdata, surr.SimpleRecurrentSurrogate().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), (3, 3, 2), epochs=6)


all_in_one_train(trpr, [torch.load('pretrained/mimic/static_encoder_mortality.pt'),
                 torch.load('pretrained/mimic/ts_encoder_mortality.pt'), surr.SimpleRecurrentSurrogate()])

"""
print("Testing:")
model=torch.load('best.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
test(model,testdata)
"""
