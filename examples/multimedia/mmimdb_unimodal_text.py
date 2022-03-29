import torch
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import MLP
from datasets.imdb.get_data import get_dataloader
from training_structures.unimodal import train, test


encoderfile = "encoder_text.pt"
headfile = "head_text.pt"
traindata, validdata, testdata = get_dataloader(
    "../video/multimodal_imdb.hdf5", "../video/mmimdb", vgg=True, batch_size=128)

encoders = MLP(300, 512, 512).cuda()
head = MLP(512, 512, 23).cuda()

train(encoders, head, traindata, validdata, 1000, early_stop=True, task="multilabel", save_encoder=encoderfile, modalnum=0,
      save_head=headfile, optimtype=torch.optim.AdamW, lr=1e-4, weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())

print("Testing:")
encoder = torch.load(encoderfile).cuda()
head = torch.load(headfile).cuda()
test(encoder, head, testdata, "imdb",
     "unimodal_image", task="multilabel", modalnum=0)
