import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch

from training_structures.Contrastive_Learning import train, test
from fusions.common_fusions import Concat
sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')
from get_data_robust import get_dataloader, get_dataloader_robust
from robustness.all_in_one import general_train, general_test
from unimodals.common_models import MLP, VGG16, MaxOut_MLP, Linear

traindata, validdata = get_dataloader('../../../video/multimodal_imdb.hdf5', batch_size=128, vgg=True)
robustdata = get_dataloader_robust('../../../video/mmimdb', '../../../video/multimodal_imdb.hdf5',batch_size=128)

encoders=[MaxOut_MLP(512, 512, 300, linear_layer=False), MaxOut_MLP(512, 1024, 4096, 512, False)]
#encoders=[MLP(300, 512, 512), MLP(4096, 1000, 512)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
head=MLP(1024,512,23).cuda()
refiner = MLP(1024,3072,4396).cuda()
#refiner = MLP(1024,2048,1024).cuda()
fusion=Concat().cuda()

def trainprocess(filename):
    train(encoders,fusion,head,refiner,traindata,validdata,1000, early_stop=True,task="multilabel",\
    save=filename, optimtype=torch.optim.AdamW,lr=1e-3,weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())
filename = general_train(trainprocess, 'mmimdb_contrast')

def testprocess(model, robustdata):
    return test(model, robustdata,criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")
general_test(testprocess, filename, robustdata, multi_measure=True)
