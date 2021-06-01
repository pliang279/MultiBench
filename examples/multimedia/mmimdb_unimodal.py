import sys
import os
sys.path.append(os.getcwd())

import torch

from training_structures.unimodal import train, test
from datasets.imdb.get_data import get_dataloader
from unimodals.common_models import MLP, Linear, MaxOut_MLP

traindata, validdata, testdata = get_dataloader('../video/multimodal_imdb.hdf5', vgg=True)

#encoders=MaxOut_MLP(512, 512, 300, linear_layer=False).cuda()
encoders=MaxOut_MLP(512, 1024, 4096, 512, False).cuda()
#encoders=[MLP(300, 512, 512), VGG16(512)]
head=Linear(512,23).cuda()

train(encoders,head,traindata,validdata,1000, early_stop=True,task="multilabel", save_encoder="encoder_i.pt", modalnum=1,\
    save_head="head_i.pt", optimtype=torch.optim.AdamW,lr=8e-3,weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())

print("Testing:")
encoder=torch.load('encoder_i.pt').cuda()
head=torch.load('head_i.pt').cuda()
test(encoder,head,testdata,task="multilabel", modalnum=1)