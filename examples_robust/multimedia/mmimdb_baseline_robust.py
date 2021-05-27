import sys
import os
sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')
sys.path.append('/home/pliang/multibench/MultiBench')
import torch

from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat
from unimodals.common_models import MLP, VGG16, Linear
# from get_data_robust import get_dataloader, get_dataloader_robust
from get_data import get_dataloader
from robustness.all_in_one import general_train, general_test

# traindata, validdata = get_dataloader('../../../video/multimodal_imdb.hdf5', vgg=True)
# # robustdata = get_dataloader_robust('../../../video/mmimdb')

# encoders=[MLP(300, 512, 512), MLP(4096, 1000, 512)]
# #encoders=[MLP(300, 512, 512), VGG16(512)]
# head=MLP(1024,512,23).cuda()
# fusion=Concat().cuda()

# def trainprocess(filename):
#     train(encoders,fusion,head,traindata,validdata,1000, early_stop=True,task="multilabel", regularization=True, save=filename, optimtype=torch.optim.AdamW,lr=5e-5,weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())
# filename = general_train(trainprocess, 'mmimdb_baseline')

# def testprocess(model, noise_level):
#     return test(model,robustdata[noise_level],criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")
# general_test(testprocess, filename, len(robustdata))

traindata, validdata, testdata = get_dataloader('../../../video/multimodal_imdb.hdf5', batch_size=1, vgg=True)

encoders=[MLP(300, 512, 512), MLP(4096, 1000, 512)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
head=MLP(1024,512,23).cuda()
fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,1000, early_stop=True,task="multilabel", regularization=True,\
    save="best_reg.pt", optimtype=torch.optim.AdamW,lr=5e-5,weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())

print("Testing:")
model=torch.load('best_reg.pt').cuda()
test(model,testdata,criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")