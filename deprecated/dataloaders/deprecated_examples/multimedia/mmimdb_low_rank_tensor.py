import torch
from torch import nn
from unimodals.common_models import MaxOut_MLP, Linear
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import LowRankTensorFusion
from training_structures.Simple_Late_Fusion import train, test
import sys
import os
sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader(
    '../video/multimodal_imdb.hdf5', vgg=True, batch_size=128)
encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False),
            MaxOut_MLP(512, 1024, 4096, 512, False)]
head = Linear(512, 23).cuda()

fusion = LowRankTensorFusion([512, 512], 512, 128).cuda()

train(encoders, fusion, head, traindata, validdata, 1000, early_stop=True, task="multilabel", regularization=False,
      save="best_lrtf.pt", optimtype=torch.optim.AdamW, lr=8e-3, weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())

print("Testing:")
model = torch.load('best_lrtf.pt').cuda()
test(model, testdata, criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")
