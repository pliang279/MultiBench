import sys
import os
sys.path.append(os.getcwd())

from training_structures.architecture_search import train,test
from fusions.common_fusions import Concat
from datasets.imdb.get_data import get_dataloader

import torch
import utils.surrogate as surr

traindata, validdata, testdata = get_dataloader('../video/multimodal_imdb.hdf5', vgg=True, batch_size=128)
# Todo
# s_data=train(['pretrained/mmimdb/encoder_t.pt','pretrained/mmimdb/encoder_i.pt'],16,23,[(6,12,24),(6,12,24,48,96)],
#         traindata,validdata,surr.SimpleRecurrentSurrogate().cuda(),(3,5,2),epochs=6)


# print("Testing:")
# model=torch.load('best.pt').cuda()
# test(model,testdata)


