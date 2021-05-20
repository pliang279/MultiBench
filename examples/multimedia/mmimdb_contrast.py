import sys
import os
sys.path.append(os.getcwd())

import torch

from training_structures.Contrastive_Learning import train, test
from fusions.common_fusions import Concat
from datasets.imdb.get_data import get_dataloader
from unimodals.common_models import MLP

traindata, validdata, testdata = get_dataloader('../video/multimodal_imdb.hdf5')

encoders=[MLP(300, 512, 512), MLP(4096, 1000, 512)]
head=MLP(1024,512,23).cuda()
refiner = MLP(1024,3072,4396).cuda()
fusion=Concat().cuda()

train(encoders,fusion,head,refiner,traindata,validdata,1000, early_stop=True,task="multilabel",\
    optimtype=torch.optim.AdamW,lr=5e-5,weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())

print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata,criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")