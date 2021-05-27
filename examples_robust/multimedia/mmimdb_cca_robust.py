
import sys
import os
sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')
sys.path.append('/home/pliang/multibench/MultiBench')

import torch

from training_structures.cca import train, test
from fusions.common_fusions import Concat
from get_data_robust import get_dataloader, get_dataloader_robust
from unimodals.common_models import MLP, VGG16, Linear
from robustness.all_in_one import general_train, general_test

traindata, validdata = get_dataloader('../../../video/multimodal_imdb.hdf5', vgg=True)
robustdata = get_dataloader_robust('../../../video/mmimdb')

#encoders=[MLP(300, 512, 512), MLP(4096, 1000, 512)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
encoders=[Linear(300, 512), Linear(4096,512)]
head=MLP(1024,512,23).cuda()
fusion=Concat().cuda()

def trainprocess(filename):
    train(encoders,fusion,head,traindata,validdata,1000, early_stop=True,task="multilabel",save=filename, optimtype=torch.optim.RMSprop,lr=1e-10,weight_decay=0, criterion=torch.nn.BCEWithLogitsLoss())
filename = general_train(trainprocess, 'mmimdb_cca')

def testprocess(model, noise_level):
    return test(model,robustdata[noise_level],criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")
general_test(testprocess, filename, len(robustdata), multi_measure=True)
