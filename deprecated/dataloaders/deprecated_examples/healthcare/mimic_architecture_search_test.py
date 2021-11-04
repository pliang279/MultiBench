from unimodals.common_models import LeNet, MLP, Constant
from training_structures.architecture_search import train, test
import utils.surrogate as surr
import torch
from torch import nn
from datasets.mimic.get_data import get_dataloader
from fusions.common_fusions import Concat
import sys
import os
sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader(
    1, imputed_path='datasets/mimic/im.pk')

model = torch.load('temp/best.pt').cuda()
test(model, testdata, auprc=True)
