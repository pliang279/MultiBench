import torch
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import MaxOut_MLP, Identity
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test


filename = "best_ef.pt"
traindata, validdata, testdata = get_dataloader(
    "../video/multimodal_imdb.hdf5", "../video/mmimdb", vgg=True, batch_size=128)

encoders = [Identity(), Identity()]
head = MaxOut_MLP(23, 512, 4396).cuda()
fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 1000, early_stop=True, task="multilabel",
      save=filename, optimtype=torch.optim.AdamW, lr=4e-2, weight_decay=0.01, objective=torch.nn.BCEWithLogitsLoss())

print("Testing:")
model = torch.load(filename).cuda()
test(model, testdata, method_name="ef", dataset="imdb",
     criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")
