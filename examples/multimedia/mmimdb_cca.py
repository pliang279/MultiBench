import sys
import os
sys.path.append(os.getcwd())

import torch

from training_structures.cca_onestage import train, test
from fusions.common_fusions import Concat
from datasets.imdb.get_data import get_dataloader
from unimodals.common_models import MLP, VGG16, Linear

traindata, validdata, testdata = get_dataloader('../video/multimodal_imdb.hdf5', vgg=True, batch_size=800)

outdim = 32
encoders=[MLP(300,512,outdim), MLP(4096,1024,outdim)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
#encoders=[Linear(300, 512), Linear(4096,512)]
head=MLP(2*outdim,2*outdim,23).cuda()
fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,1000, early_stop=True,task="multilabel",outdim=outdim,\
    save="best_cca.pt", optimtype=torch.optim.RMSprop,lr=1e-3,weight_decay=1e-5, criterion=torch.nn.BCEWithLogitsLoss())

print("Testing:")
model=torch.load('best_cca.pt').cuda()
test(model,testdata,criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")