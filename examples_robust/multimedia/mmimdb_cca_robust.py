
import sys
import os
sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')
sys.path.append('/home/pliang/multibench/MultiBench')

import torch

import torch

from training_structures.cca_onestage import train, test
from fusions.common_fusions import Concat
from get_data_robust import get_dataloader, get_dataloader_robust
from unimodals.common_models import MLP, VGG16, Linear
from robustness.all_in_one import general_train, general_test

traindata, validdata = get_dataloader('../../../video/multimodal_imdb.hdf5', vgg=True, batch_size=800)
robustdata = get_dataloader_robust('../../../video/mmimdb', batch_size=800)

outdim = 32
encoders=[MLP(300,512,outdim), MLP(4096,1024,outdim)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
#encoders=[Linear(300, 512), Linear(4096,512)]
head=MLP(2*outdim,2*outdim,23).cuda()
fusion=Concat().cuda()

def trainprocess(filename):
    train(encoders,fusion,head,traindata,validdata,1000, early_stop=True,task="multilabel",outdim=outdim,save=filename,optimtype=torch.optim.RMSprop,lr=1e-3,weight_decay=1e-5, criterion=torch.nn.BCEWithLogitsLoss())
filename = general_train(trainprocess, 'mmimdb_cca')

def testprocess(model, noise_level):
    return test(model,robustdata[noise_level],criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")
general_test(testprocess, filename, len(robustdata), multi_measure=True)
