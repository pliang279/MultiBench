import sys
import os
import torch
from torch import nn

sys.path.append(os.getcwd())

from unimodals.common_models import VGG16, VGG16Slim, DAN, Linear, MLP, VGG11Slim, VGG11Pruned # noqa
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa
from objective_functions.objectives_for_supervised_learning import RefNet_objective # noqa
from datasets.enrico.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa
from training_structures.Supervised_Learning import train, test # noqa



dls, weights = get_dataloader('datasets/enrico/dataset')
traindata, validdata, testdata = dls
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).cuda()
# encoders=[VGG16Slim(64).cuda(), DAN(4, 16, dropout=True, dropoutp=0.25).cuda(), DAN(28, 16, dropout=True, dropoutp=0.25).cuda()]
# head = Linear(96, 20)
encoders = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda(
), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda()]
# encoders = [DAN(4, 16, dropout=True, dropoutp=0.25).cuda(), DAN(28, 16, dropout=True, dropoutp=0.25).cuda()]
refiner = Linear(32, 256*128*3*2).cuda()
head = Linear(32, 20).cuda()

fusion = Concat().cuda()

allmodules = encoders + [refiner, head, fusion]


def trainprocess():
    train(encoders, fusion, head, traindata, validdata, 50, [refiner], optimtype=torch.optim.Adam, lr=0.0001,
          weight_decay=0, task="classification", objective=RefNet_objective(0.1), objective_args_dict={'refiner': refiner})


all_in_one_train(trainprocess, allmodules)

print("Testing:")
model = torch.load('best.pt').cuda()

test(model, testdata, dataset='enrico')
