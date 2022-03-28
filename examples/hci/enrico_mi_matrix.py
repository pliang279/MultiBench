import sys
import os
import torch
from torch import nn
sys.path.append(os.getcwd())

from unimodals.common_models import VGG16, VGG16Slim, DAN, Linear, MLP, VGG11Slim, VGG11Pruned # noqa
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa
from datasets.enrico.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat, MultiplicativeInteractions2Modal # noqa
from training_structures.Supervised_Learning import train, test # noqa



dls, weights = get_dataloader('datasets/enrico/dataset')
traindata, validdata, testdata = dls
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# encoders=[VGG16Slim(64).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), DAN(4, 16, dropout=True, dropoutp=0.25).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), DAN(28, 16, dropout=True, dropoutp=0.25).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
# head = Linear(96, 20)
encoders = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda(
), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
# encoders = [DAN(4, 16, dropout=True, dropoutp=0.25).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), DAN(28, 16, dropout=True, dropoutp=0.25).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
head = Linear(32, 20).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# fusion=Concat().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
fusion = MultiplicativeInteractions2Modal([16, 16], 32, "matrix").to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

allmodules = encoders + [head, fusion]


def trainprocess():
    train(encoders, fusion, head, traindata, validdata, 50,
          optimtype=torch.optim.Adam, lr=0.0001, weight_decay=0)


all_in_one_train(trainprocess, allmodules)

print("Testing:")
model = torch.load('best.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

test(model, testdta, dataset='enrico')
