import sys
import os
sys.path.append(os.getcwd())
from training_structures.MVAE_finetune import train_MVAE,test_MVAE
from fusions.MVAE import ProductOfExperts
from unimodals.MVAE import MLPEncoder,TSEncoder,TSDecoder
from unimodals.common_models import MLP
from torch import nn
import torch
from objective_functions.recon import elbo_loss,sigmloss1d
from datasets.mimic.get_data import get_dataloader


traindata, validdata, testdata = get_dataloader(7, imputed_path='datasets/mimic/im.pk',flatten_time_series=True)
classes=2
n_latent=200
series_dim=12
timestep=24
fuse=ProductOfExperts((1,40,n_latent))
encoders=[MLPEncoder(5,20,n_latent).cuda(),TSEncoder(series_dim,30,n_latent,timestep).cuda()]
decoders=[MLP(n_latent,20,5).cuda(),TSDecoder(series_dim,30,n_latent,timestep).cuda()]
head=MLP(n_latent,20,classes).cuda()
elbo=elbo_loss([sigmloss1d,sigmloss1d],[1.0,1.0],0.0)
train_MVAE(encoders,decoders,head,fuse,traindata,validdata,elbo,6,35,5)
mvae=torch.load('best1.pt')
head=torch.load('best2.pt')
test_MVAE(mvae,head,testdata)
