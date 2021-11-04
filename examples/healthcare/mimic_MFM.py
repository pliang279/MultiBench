from unimodals.MVAE import TSEncoder, TSDecoder
from training_structures.Supervised_Learning import train, test
from utils.helper_modules import Sequential2
from datasets.mimic.get_data import get_dataloader
from objective_functions.objectives_for_supervised_learning import MFM_objective
from objective_functions.recon import sigmloss1d
import torch
from torch import nn
from unimodals.common_models import MLP
from fusions.common_fusions import Concat
import sys
import os
sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader(
    7, imputed_path='/home/pliang/yiwei/im.pk', flatten_time_series=True)
classes = 2
n_latent = 200
series_dim = 12
timestep = 24
fuse = Sequential2(Concat(), MLP(2*n_latent, n_latent, n_latent//2)).cuda()
encoders = [MLP(5, 20, n_latent).cuda(), TSEncoder(
    series_dim, 30, n_latent, timestep, returnvar=False, batch_first=True).cuda()]
decoders = [MLP(n_latent, 20, 5).cuda(), TSDecoder(
    series_dim, 30, n_latent, timestep).cuda()]
intermediates = [MLP(n_latent, n_latent//2, n_latent//2).cuda(),
                 MLP(n_latent, n_latent//2, n_latent//2).cuda()]
head = MLP(n_latent//2, 20, classes).cuda()
argsdict = {'decoders': decoders, 'intermediates': intermediates}
additional_modules = decoders+intermediates
objective = MFM_objective(2.0, [sigmloss1d, sigmloss1d], [1.0, 1.0])

train(encoders, fuse, head, traindata, validdata, 20, additional_modules,
      objective=objective, objective_args_dict=argsdict)

mvae = torch.load('best.pt')
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(mvae, testdata, dataset='mimic 7')
