import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch

from training_structures.Simple_Late_Fusion import train, test
from fusions.common_fusions import Concat
sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')
from get_data_robust import get_dataloader, get_dataloader_robust
from robustness.all_in_one import general_train, general_test
from unimodals.common_models import MLP, VGG16, Linear, MaxOut_MLP

traindata, validdata = get_dataloader('../../../video/multimodal_imdb.hdf5', vgg=True)
robust_vision, robust_text = get_dataloader_robust('../../../video/mmimdb')

encoders=[MaxOut_MLP(512, 512, 300, linear_layer=False), MaxOut_MLP(512, 1024, 4096, 512, False)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
head= Linear(1024, 23).cuda()
fusion=Concat().cuda()

def trainprocess(filename):
    train(encoders,fusion,head,traindata,validdata,1000, early_stop=True,task="multilabel", regularization=False,\
    save=filename, optimtype=torch.optim.AdamW,lr=8e-3,weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())
filename = general_train(trainprocess, 'mmimdb_simple_late_fusion')

def testprocess(model, robustdata):
    return test(model, robustdata,criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")
general_test(testprocess, filename, [robust_vision, robust_text], multi_measure=True)