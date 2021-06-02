import sys
import os
sys.path.append(os.getcwd())

import torch

from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat
from datasets.imdb.get_data import get_dataloader
from unimodals.common_models import MLP, VGG16, Linear, MaxOut_MLP

traindata, validdata, testdata = get_dataloader('../video/multimodal_imdb.hdf5', vgg=True)

encoders=[MaxOut_MLP(512, 128, 300, linear_layer=False), MaxOut_MLP(512, 1024, 4096, 128, False)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
head= Linear(256, 23).cuda()
fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,1000, early_stop=True,task="multilabel", regularization=True,\
    save="best_reg.pt", optimtype=torch.optim.AdamW,lr=1e-2,weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())

print("Testing:")
model=torch.load('best_reg.pt').cuda()
test(model,testdata,criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")