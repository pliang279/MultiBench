from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
from objective_functions.recon import elbo_loss, sigmloss1dcentercrop
from unimodals.MVAE import LeNetEncoder, DeLeNet
from training_structures.MVAE_mixed import train_MVAE, test_MVAE
from datasets.avmnist.get_data import get_dataloader
import torch
from torch import nn
from unimodals.common_models import MLP
from fusions.MVAE import ProductOfExperts
import sys
import os
sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader(
    '/data/yiwei/avmnist/_MFAS/avmnist')

classes = 10
n_latent = 200
fuse = ProductOfExperts((1, 40, n_latent))


channels = 6
encoders = [LeNetEncoder(1, channels, 3, n_latent).cuda(
), LeNetEncoder(1, channels, 5, n_latent).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
decoders = [DeLeNet(1, channels, 3, n_latent).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
            DeLeNet(1, channels, 5, n_latent).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
head = MLP(n_latent, 40, classes).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
elbo = elbo_loss([sigmloss1dcentercrop(28, 34),
                 sigmloss1dcentercrop(112, 130)], [1.0, 1.0], 0.0)


def trpr():
    train_MVAE(encoders, decoders, head, fuse, traindata, validdata, elbo, 20)


# all_in_one_train(trpr,[encoders[0],encoders[1],decoders[0],decoders[1],head])
mvae = torch.load('best1.pt')
head = torch.load('best2.pt')


def tepr():
    test_MVAE(mvae, head, testdata)


all_in_one_test(tepr, [encoders[0], encoders[1], head])
