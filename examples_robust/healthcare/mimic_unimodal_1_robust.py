import sys
import os
sys.path.append('/home/pliang/multibench/MultiBench/datasets/mimic')
sys.path.append('/home/pliang/multibench/MultiBench')
from training_structures.unimodal import train, test
from get_data_robust import get_dataloader
from unimodals.common_models import MLP, GRUWithLinear
from torch import nn
import torch

#get dataloader for icd9 classification task 7
filename_encoder = 'mimic_unimodal_1_encoder.pt'
filename_head = 'mimic_unimodal_1_head.pt'
traindata, validdata, testdata, robustdata = get_dataloader(7, imputed_path='../../datasets/mimic/im.pk', tabular_robust=False)
modalnum =1 
#build encoders, head and fusion layer
#encoders = [MLP(5, 10, 10,dropout=False).cuda(), GRU(12, 30,dropout=False).cuda()]
encoder = GRUWithLinear(12,30,15,flatten=True).cuda()
head = MLP(360, 40, 2, dropout=False).cuda()


#train
train(encoder, head, traindata, validdata, 20, auprc=False,modalnum=modalnum,save_encoder=filename_encoder,save_head=filename_head)

#test
encoder = torch.load(filename_encoder).cuda()
head = torch.load(filename_head).cuda()
acc = []
print("Robustness testing:")
for noise_level in range(len(robustdata)):
    print("Noise level {}: ".format(noise_level/10))
    acc.append(test(encoder, head, robustdata[noise_level], auprc=False, modalnum=modalnum))

print("Accuracy of different noise levels:", acc)
