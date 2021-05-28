import sys
import os
sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')
sys.path.append('/home/pliang/multibench/MultiBench')

import torch

from training_structures.unimodal import train, test
from get_data_robust import get_dataloader, get_dataloader_robust
from unimodals.common_models import MLP
from robustness.all_in_one import general_train, general_test

traindata, validdata = get_dataloader('../../../video/multimodal_imdb.hdf5', vgg=True)
robustdata = get_dataloader_robust('../../../video/mmimdb')

encoders=MLP(300, 512, 512).cuda()
#encoders=[MLP(300, 512, 512), VGG16(512)]
head=MLP(512,256,23).cuda()

def trainprocess(filename_encoder, filename_head):
    train(encoders,head,traindata,validdata,1000, early_stop=True,task="multilabel", save_encoder=filename_encoder, modalnum=0, save_head=filename_head, optimtype=torch.optim.AdamW,lr=5e-5,weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())
filename = general_train(trainprocess, 'mmimdb_unimodal', encoder=True)

def testprocess(encoder, head, noise_level):
    return test(encoder,head,robustdata[noise_level],criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")
general_test(testprocess, filename, len(robustdata), multi_measure=True)
