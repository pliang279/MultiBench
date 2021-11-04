from unimodals.common_models import MLP
from datasets.imdb.get_data import get_dataloader
from training_structures.unimodal import train, test
import torch
import sys
import os
sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader(
    '../video/multimodal_imdb.hdf5', vgg=True, batch_size=128)

encoders = MLP(300, 512, 512).cuda()
# encoders=MLP(4096,1024,512).cuda()
head = MLP(512, 512, 23).cuda()

train(encoders, head, traindata, validdata, 1000, early_stop=True, task="multilabel", save_encoder="encoder_t.pt", modalnum=0,
      save_head="head_t.pt", optimtype=torch.optim.AdamW, lr=1e-4, weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())

print("Testing:")
encoder = torch.load('encoder_t.pt').cuda()
head = torch.load('head_t.pt').cuda()
test(encoder, head, testdata, task="multilabel", modalnum=0)
