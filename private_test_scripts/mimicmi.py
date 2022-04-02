from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
import torch
from torch import nn
from unimodals.common_models import MLP, GRU
from datasets.mimic.get_data import get_dataloader
from fusions.common_fusions import MultiplicativeInteractions2Modal
from training_structures.Simple_Late_Fusion import train, test
import sys
import os
sys.path.append(os.getcwd())


# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    1, imputed_path='datasets/mimic/im.pk')

# build encoders, head and fusion layer
encoders = [MLP(5, 10, 10, dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
            GRU(12, 30, dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
head = MLP(100, 40, 2, dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#fusion = LowRankTensorFusion([10,720],100,40).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
fusion = MultiplicativeInteractions2Modal(
    [10, 720], 100, 'matrix', flatten=True)

# train


def trainprocess():
    train(encoders, fusion, head, traindata, validdata, 20, auprc=True)


all_in_one_train(trainprocess, [encoders[0], encoders[1], fusion, head])


# test
print("Testing: ")
model = torch.load('best.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def testprocess():
    test(model, testdata, auprc=True)


all_in_one_test(testprocess, [model])
