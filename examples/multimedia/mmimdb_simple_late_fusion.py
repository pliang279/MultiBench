import sys
import os
sys.path.append(os.getcwd())

import torch

from training_structures.Supervised_Learning import train, test
from fusions.common_fusions import Concat
from datasets.imdb.get_data import get_dataloader
from unimodals.common_models import MLP, VGG16, Linear, MaxOut_MLP

traindata, validdata, testdata = get_dataloader('/home/pliang/multibench/video/multimodal_imdb.hdf5', vgg=True, batch_size=128)

encoders=[MaxOut_MLP(512, 512, 300, linear_layer=False), MaxOut_MLP(512, 1024, 4096, 512, False)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
head= Linear(1024, 23).cuda()
fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,1000, early_stop=True,task="multilabel",    save="best_lf.pt", optimtype=torch.optim.AdamW,lr=8e-3,weight_decay=0.01, objective=torch.nn.BCEWithLogitsLoss())

print("Testing:")
model=torch.load('best_lf.pt').cuda()
test(model,testdata,criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")
