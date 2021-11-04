from unimodals.common_models import VGG16, VGG16Slim, DAN, Linear, MLP, VGG11Slim, VGG11Pruned
import torch
from robustness.all_in_one import general_train, general_test
from datasets.enrico.get_data_robust import get_dataloader_robust
from datasets.enrico.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.gradient_blend import train, test
import sys
import os
from torch import nn
sys.path.append(os.getcwd())


dls, weights = get_dataloader('datasets/enrico/dataset')
traindata, validdata, _ = dls
robustdata = get_dataloader_robust('datasets/enrico/dataset')
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).cuda()
# encoders=[VGG16Slim(64).cuda(), DAN(4, 16, dropout=True, dropoutp=0.25).cuda(), DAN(28, 16, dropout=True, dropoutp=0.25).cuda()]
# head = Linear(96, 20)
encoders = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda(
), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda()]
# encoders = [DAN(4, 16, dropout=True, dropoutp=0.25).cuda(), DAN(28, 16, dropout=True, dropoutp=0.25).cuda()]
mult_head = Linear(32, 20).cuda()
uni_head = [Linear(16, 20).cuda(), Linear(16, 20).cuda()]

fusion = Concat().cuda()

# train(encoders,fusion,head,traindata,validdata,num_epoch=50,gb_epoch=10,optimtype=torch.optim.Adam,lr=0.0001,weight_decay=0)
allmodules = encoders + [mult_head, fusion] + uni_head


def trainprocess(filename):
    train(encoders, mult_head, uni_head, fusion, traindata, validdata, 50, gb_epoch=10,
          optimtype=torch.optim.Adam, lr=0.0001, weight_decay=0, savedir=filename)


filename = general_train(trainprocess, 'enrico_gradient_blend')


def testprocess(model, testdata):
    return test(model, testdata)


general_test(testprocess, filename, robustdata)
