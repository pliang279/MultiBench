import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from training_structures.MVAE_mixed import train_MVAE,test_MVAE
from fusions.MVAE import ProductOfExperts
from unimodals.common_models import MLP
from unimodals.MVAE import LeNetEncoder,DeLeNet
from torch import nn
import torch
from objective_functions.recon import elbo_loss,sigmloss1dcentercrop
from datasets.avmnist.get_data import get_dataloader

filename1 = 'avmnist_MVAE_mixed_best1.pt'
filename2 = 'avmnist_MVAE_mixed_best2.pt'
traindata, validdata, testdata, robustdata = get_dataloader('../../../../yiwei/avmnist/_MFAS/avmnist')

classes=10
n_latent=200
fuse=ProductOfExperts((1,40,n_latent))


channels=6
encoders=[LeNetEncoder(1,channels,3,n_latent).cuda(),LeNetEncoder(1,channels,5,n_latent).cuda()]
decoders=[DeLeNet(1,channels,3,n_latent).cuda(),DeLeNet(1,channels,5,n_latent).cuda()]
head=MLP(n_latent,40,classes).cuda()
elbo=elbo_loss([sigmloss1dcentercrop(28,34),sigmloss1dcentercrop(112,130)],[1.0,1.0],0.0)
train_MVAE(encoders,decoders,head,fuse,traindata,validdata,elbo,20,savedirbackbone=filename1,savedirhead=filename2)

mvae=torch.load(filename1)
head=torch.load(filename2)
print("Testing:")
test_MVAE(mvae,head,testdata)

print("Robustness testing:")
test_MVAE(mvae,testdata)
