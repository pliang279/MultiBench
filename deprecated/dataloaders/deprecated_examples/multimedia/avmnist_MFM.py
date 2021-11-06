from objective_functions.recon import recon_weighted_sum, sigmloss1dcentercrop
from unimodals.MVAE import LeNetEncoder, DeLeNet
from training_structures.MFM import train_MFM, test_MFM
from datasets.avmnist.get_data import get_dataloader
import torch
from torch import nn
from unimodals.common_models import MLP
from fusions.common_fusions import Concat
import sys
import os
sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist')
channels = 6

classes = 10
n_latent = 200
fuse = Concat()

encoders = [LeNetEncoder(1, channels, 3, n_latent, twooutput=False).cuda(
), LeNetEncoder(1, channels, 5, n_latent, twooutput=False).cuda()]
decoders = [DeLeNet(1, channels, 3, n_latent).cuda(),
            DeLeNet(1, channels, 5, n_latent).cuda()]

intermediates = [MLP(n_latent, n_latent//2, n_latent//2).cuda(), MLP(n_latent,
                                                                     n_latent//2, n_latent//2).cuda(), MLP(2*n_latent, n_latent, n_latent//2).cuda()]
head = MLP(n_latent//2, 40, classes).cuda()
recon_loss = recon_weighted_sum([sigmloss1dcentercrop(
    28, 34), sigmloss1dcentercrop(112, 130)], [1.0, 1.0])
train_MFM(encoders, decoders, head, intermediates,
          fuse, recon_loss, traindata, validdata, 25)
model = torch.load('best.pt')
test_MFM(model, testdata)
