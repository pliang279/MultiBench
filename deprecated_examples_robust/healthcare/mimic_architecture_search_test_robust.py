import sys
import os
sys.path.append(os.getcwd())
from training_structures.architecture_search import train,test
from fusions.common_fusions import Concat
from datasets.mimic.get_data_robust import get_dataloader
from unimodals.common_models import LeNet,MLP,Constant
from torch import nn
import torch
import utils.surrogate as surr


traindata, validdata, testdata, robustdata = get_dataloader(7, imputed_path='datasets/mimic/im.pk')

model = torch.load('/home/pliang/yiwei/best_icd9_70_79.pt').cuda()
test(model,testdata,auprc=True)
acc = []
print("Robustness testing:")
for noise_level in range(len(robustdata)):
    print("Noise level {}: ".format(noise_level/10))
    acc.append(test(model, robustdata[noise_level], auprc=False))

print("Accuracy of different noise levels:", acc)
