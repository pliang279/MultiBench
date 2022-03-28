from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
import torch
from torch import nn
from unimodals.common_models import MLP, GRU
from datasets.mimic.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.gradient_blend import train, test
import sys
import os
sys.path.append(os.getcwd())

filename = 'bbest10.pt'

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    -1, imputed_path='datasets/mimic/im.pk')

# build encoders, head and fusion layer
encoders = [MLP(5, 10, 10).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), GRU(12, 30, flatten=True).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
head = MLP(730, 40, 6).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
fusion = Concat().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
unimodal_heads = [MLP(10, 20, 6).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), MLP(720, 40, 6).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
allmodules = [encoders[0], encoders[1], head,
              fusion, unimodal_heads[0], unimodal_heads[1]]

# train


def trainprocess():
    train(encoders, head, unimodal_heads, fusion, traindata,
          validdata, 300, lr=0.005, AUPRC=False, savedir=filename)


all_in_one_train(trainprocess, allmodules)

# test
print("Testing: ")
model = torch.load(filename).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def testprocess():
    test(model, testdata, auprc=False)


all_in_one_test(testprocess, [model])
