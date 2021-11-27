import sys
import os

sys.path.append(os.getcwd())

from objective_functions.recon import elbo_loss, sigmloss1dcentercrop
from unimodals.MVAE import LeNetEncoder, DeLeNet
from training_structures.Supervised_Learning import train, test
from datasets.avmnist.get_data import get_dataloader
from objective_functions.objectives_for_supervised_learning import MVAE_objective
import torch
from torch import nn
from unimodals.common_models import MLP
from fusions.MVAE import ProductOfExperts_Zipped


traindata, validdata, testdata = get_dataloader(
    '/home/pliang/yiwei/avmnist/_MFAS/avmnist')

classes = 10
n_latent = 200
fuse = ProductOfExperts_Zipped((1, 40, n_latent))


channels = 6
encoders = [LeNetEncoder(1, channels, 3, n_latent).cuda(
), LeNetEncoder(1, channels, 5, n_latent).cuda()]
decoders = [DeLeNet(1, channels, 3, n_latent).cuda(),
            DeLeNet(1, channels, 5, n_latent).cuda()]
head = MLP(n_latent, 40, classes).cuda()
elbo = MVAE_objective(2.0, [sigmloss1dcentercrop(
    28, 34), sigmloss1dcentercrop(112, 130)], [1.0, 1.0], annealing=0.0)
train(encoders, fuse, head, traindata, validdata, 20, decoders,
      objective=elbo, objective_args_dict={'decoders': decoders})
mvae = torch.load('best.pt')
test(mvae, testdata, no_robust=True)
