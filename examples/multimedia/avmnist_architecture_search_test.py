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

traindata, validdata, testdata = get_dataloader('/data/yiwei/avmnist/_MFAS/avmnist',batch_size=32)
model = torch.load('temp/best.pt').cuda()
test(model,testdata,no_robust=True)
