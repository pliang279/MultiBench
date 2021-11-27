import sys
import os
sys.path.append(os.getcwd())

from objective_functions.recon import recon_weighted_sum, sigmloss1dcentercrop
from unimodals.MVAE import LeNetEncoder, DeLeNet
from training_structures.Supervised_Learning import train, test
from objective_functions.objectives_for_supervised_learning import MFM_objective
from utils.helper_modules import Sequential2
from datasets.avmnist.get_data import get_dataloader
import torch
from torch import nn
from unimodals.common_models import MLP
from fusions.common_fusions import Concat


traindata, validdata, testdata = get_dataloader(
    '/home/pliang/yiwei/avmnist/_MFAS/avmnist')
channels = 6

classes = 10
n_latent = 200
fuse = Concat()
fuse = Sequential2(Concat(), MLP(2*n_latent, n_latent, n_latent//2)).cuda()
encoders = [LeNetEncoder(1, channels, 3, n_latent, twooutput=False).cuda(
), LeNetEncoder(1, channels, 5, n_latent, twooutput=False).cuda()]
decoders = [DeLeNet(1, channels, 3, n_latent).cuda(),
            DeLeNet(1, channels, 5, n_latent).cuda()]

intermediates = [MLP(n_latent, n_latent//2, n_latent//2).cuda(),
                 MLP(n_latent, n_latent//2, n_latent//2).cuda()]
head = MLP(n_latent//2, 40, classes).cuda()
objective = MFM_objective(2.0, [sigmloss1dcentercrop(
    28, 34), sigmloss1dcentercrop(112, 130)], [1.0, 1.0])
train(encoders, fuse, head, traindata, validdata, 25, decoders+intermediates,
      objective=objective, objective_args_dict={'decoders': decoders, 'intermediates': intermediates})
model = torch.load('best.pt')
test(model, testdata, no_robust=True)
