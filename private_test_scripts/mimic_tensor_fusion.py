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
    7, imputed_path='datasets/mimic/im.pk')

# build encoders, head and fusion layer
encoders = [MLP(5, 10, 10, dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
            GRU(12, 30, dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
head = MLP(100, 40, 2, dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
fusion = MultiplicativeInteractions2Modal(
    [10, 720], 100, 'matrix', flatten=True).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# train
train(encoders, fusion, head, traindata, validdata, 20, lr=0.0001, auprc=True)

# test
print("Testing: ")
model = torch.load('best.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
test(model, testdata, auprc=True)
