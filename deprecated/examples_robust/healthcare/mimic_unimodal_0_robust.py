import torch
from torch import nn
from unimodals.common_models import MLP, GRU
from datasets.mimic.get_data_robust import get_dataloader
from training_structures.unimodal import train, test
import sys
import os
sys.path.append(os.getcwd())

# get dataloader for icd9 classification task 7
filename_encoder = 'mimic_unimodal_0_encoder.pt'
filename_head = 'mimic_unimodal_0_head.pt'
traindata, validdata, testdata, robustdata = get_dataloader(
    1, imputed_path='datasets/mimic/im.pk', timeseries_robust=False)
modalnum = 0
# build encoders, head and fusion layer
#encoders = [MLP(5, 10, 10,dropout=False).cuda(), GRU(12, 30,dropout=False).cuda()]
encoder = MLP(5, 10, 10).cuda()
head = MLP(10, 40, 2, dropout=False).cuda()


# train
train(encoder, head, traindata, validdata, 20, auprc=False,
      modalnum=modalnum, save_encoder=filename_encoder, save_head=filename_head)

# test
encoder = torch.load(filename_encoder).cuda()
head = torch.load(filename_head).cuda()
acc = []
print("Robustness testing:")
for noise_level in range(len(robustdata)):
    print("Noise level {}: ".format(noise_level/10))
    acc.append(
        test(encoder, head, robustdata[noise_level], auprc=False, modalnum=modalnum))

print("Accuracy of different noise levels:", acc)
