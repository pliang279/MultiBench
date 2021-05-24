import sys
import os
sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')
sys.path.append('/home/pliang/multibench/MultiBench')
import torch

from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat
from get_data_robust import get_dataloader
from unimodals.common_models import LeNet,MLP,VGG,MaxOut_MLP

robustdata = get_dataloader('../../../video/mmimdb')

# encoders=[MaxOut_MLP(23).cuda(),VGG(23).cuda()]
# head=MLP(4096,512,23).cuda()
# fusion=Concat().cuda()

# Train
# train(encoders,fusion,head,traindata,validdata,100,optimtype=torch.optim.SGD,lr=0.01,weight_decay=0.002, save='mmimdb_baseline_best.pt')

#test
model=torch.load('mmimdb_baseline_best.pt').cuda()
acc = []
print("Robustness testing:")
for noise_level in range(len(robustdata)):
    print("Noise level {}: ".format(noise_level/10))
    acc.append(test(model, robustdata[noise_level]))

print("Accuracy of different noise levels:", acc)
