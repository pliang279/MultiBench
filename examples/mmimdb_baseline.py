import sys
import os
sys.path.append(os.getcwd())

import torch

from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat
from datasets.imdb.get_data import get_dataloader
from unimodals.common_models import LeNet,MLP,VGG,MaxOut_MLP

traindata, validdata, testdata = get_dataloader('multibench/video/multimodal_imdb.hdf5')

encoders=[MaxOut_MLP(23).cuda(),VGG(23).cuda()]
head=MLP(4096,512,23).cuda()
fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,100,optimtype=torch.optim.SGD,lr=0.01,weight_decay=0.002)

print("Testing:")
model=torch.load('best.pt').cuda()
test(model,testdata)