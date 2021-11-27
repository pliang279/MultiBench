import torch
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import Linear, MaxOut_MLP
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat
from objective_functions.objectives_for_supervised_learning import RMFE_object
from training_structures.Supervised_Learning import train, test



filename = "best_reg.pt"
traindata, validdata, testdata = get_dataloader(
    "../video/multimodal_imdb.hdf5", "../video/mmimdb", vgg=True, batch_size=128)

encoders = [MaxOut_MLP(512, 128, 300, linear_layer=False),
            MaxOut_MLP(512, 1024, 4096, 128, False)]
head = Linear(256, 23).cuda()
fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 1000, early_stop=True, task="multilabel", save=filename, objective_args_dict={},
      optimtype=torch.optim.AdamW, lr=1e-2, weight_decay=0.01, objective=RMFE_object())

print("Testing:")
model = torch.load(filename).cuda()
test(model, testdata, method_name="rmfe", dataset="imdb",
     criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")
