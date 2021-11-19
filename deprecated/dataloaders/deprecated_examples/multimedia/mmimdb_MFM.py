from objective_functions.recon import recon_weighted_sum, sigmloss1dcentercrop, sigmloss1d
from training_structures.MFM import train_MFM, test_MFM
from datasets.imdb.get_data import get_dataloader
import torch
from unimodals.common_models import Linear, MLP, MaxOut_MLP
from fusions.common_fusions import Concat
import sys
import os
sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader(
    '../video/multimodal_imdb.hdf5', vgg=True, batch_size=128)

classes = 23
n_latent = 512
fuse = Concat()

encoders = [MaxOut_MLP(512, 512, 300, n_latent, False).cuda(
), MaxOut_MLP(512, 1024, 4096, n_latent, False).cuda()]
decoders = [MLP(n_latent, 600, 300).cuda(), MLP(n_latent, 2048, 4096).cuda()]

intermediates = [MLP(n_latent, n_latent//2, n_latent//2).cuda(),
                 MLP(n_latent, n_latent//2, n_latent//2).cuda(), MLP(2*n_latent, n_latent, n_latent//2).cuda()]
head = Linear(n_latent//2, classes).cuda()
recon_loss = recon_weighted_sum([sigmloss1d, sigmloss1d], [1.0, 1.0])
train_MFM(encoders, decoders, head, intermediates, fuse, recon_loss, traindata, validdata, 1000, learning_rate=5e-3,
          savedir="best_mfm.pt", task="multilabel", early_stop=True, criterion=torch.nn.BCEWithLogitsLoss())
model = torch.load('best_mfm.pt')
test_MFM(model, testdata, task="multilabel")
