import sys
import os
sys.path.append(os.getcwd())
from training_structures.architecture_search import train,test
from fusions.common_fusions import Concat
from datasets.mimic.get_data import get_dataloader
from unimodals.common_models import LeNet,MLP,Constant
from torch import nn
import torch
import utils.surrogate as surr

traindata, validdata, testdata = get_dataloader(1, imputed_path='datasets/mimic/im.pk')

model = torch.load('temp/best.pt').cuda()
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(model,testdata,dataset='mimic 1',auprc=True)
