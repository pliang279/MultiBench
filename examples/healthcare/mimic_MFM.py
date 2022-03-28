import sys
import os
import torch 
from torch import nn
sys.path.append(os.getcwd())

from unimodals.MVAE import TSEncoder, TSDecoder # noqa
from training_structures.Supervised_Learning import train, test # noqa
from utils.helper_modules import Sequential2 # noqa
from datasets.mimic.get_data import get_dataloader # noqa
from objective_functions.objectives_for_supervised_learning import MFM_objective # noqa
from objective_functions.recon import sigmloss1d # noqa
from unimodals.common_models import MLP # noqa
from fusions.common_fusions import Concat # noqa


traindata, validdata, testdata = get_dataloader(
    7, imputed_path='/home/pliang/yiwei/im.pk', flatten_time_series=True)
classes = 2
n_latent = 200
series_dim = 12
timestep = 24
fuse = Sequential2(Concat(), MLP(2*n_latent, n_latent, n_latent//2)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
encoders = [MLP(5, 20, n_latent).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), TSEncoder(
    series_dim, 30, n_latent, timestep, returnvar=False, batch_first=True).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
decoders = [MLP(n_latent, 20, 5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), TSDecoder(
    series_dim, 30, n_latent, timestep).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
intermediates = [MLP(n_latent, n_latent//2, n_latent//2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
                 MLP(n_latent, n_latent//2, n_latent//2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
head = MLP(n_latent//2, 20, classes).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
argsdict = {'decoders': decoders, 'intermediates': intermediates}
additional_modules = decoders+intermediates
objective = MFM_objective(2.0, [sigmloss1d, sigmloss1d], [1.0, 1.0])

train(encoders, fuse, head, traindata, validdata, 20, additional_modules,
      objective=objective, objective_args_dict=argsdict)

mvae = torch.load('best.pt')
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(mvae, testdata, dataset='mimic 7')
