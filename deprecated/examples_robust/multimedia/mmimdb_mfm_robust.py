from objective_functions.recon import recon_weighted_sum, sigmloss1d
from unimodals.common_models import MLP, VGG16, MaxOut_MLP, Linear
from robustness.all_in_one import general_train, general_test
from get_data_robust import get_dataloader, get_dataloader_robust
from training_structures.MFM import train_MFM, test_MFM
from fusions.common_fusions import Concat
import torch
import sys
import os
sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')
sys.path.append('/home/pliang/multibench/MultiBench')


traindata, validdata = get_dataloader(
    '../../../video/multimodal_imdb.hdf5', batch_size=128, vgg=True)
robustdata = get_dataloader_robust(
    '../../../video/mmimdb', '../../../video/multimodal_imdb.hdf5', batch_size=128)

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


def trainprocess(filename):
    train_MFM(encoders, decoders, head, intermediates, fuse, recon_loss, traindata, validdata, 1000, learning_rate=5e-3,
              savedir=filename, task="multilabel", early_stop=True, criterion=torch.nn.BCEWithLogitsLoss())


filename = general_train(trainprocess, 'mmimdb_mfm')


def testprocess(model, robustdata):
    return test_MFM(model, robustdata, task="multilabel")


general_test(testprocess, filename, robustdata, multi_measure=True)
