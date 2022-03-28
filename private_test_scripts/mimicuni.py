from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
import torch
from torch import nn
from unimodals.common_models import MLP, GRUWithLinear
from datasets.mimic.get_data import get_dataloader
from training_structures.unimodal import train, test
import sys
import os
sys.path.append(os.getcwd())

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    1, imputed_path='datasets/mimic/im.pk')
modalnum = 1
# build encoders, head and fusion layer
#encoders = [MLP(5, 10, 10,dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), GRU(12, 30,dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]

encoder = GRUWithLinear(12, 30, 15, flatten=True).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = MLP(360, 40, 2, dropout=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


# train
def trainprocess():
    train(encoder, head, traindata, validdata,
          20, auprc=False, modalnum=modalnum)


all_in_one_train(trainprocess, [encoder, head])

# test
print("Testing: ")
encoder = torch.load('encoder.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = torch.load('head.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def testprocess():
    test(encoder, head, testdata, auprc=False, modalnum=modalnum)


all_in_one_test(testprocess, [encoder, head])
