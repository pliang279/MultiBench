import torch
import sys
import os

sys.path.append(os.getcwd())

from unimodals.common_models import Linear, MaxOut_MLP
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat
from objective_functions.objectives_for_supervised_learning import CCA_objective
from training_structures.Supervised_Learning import train, test


filename = "best_cca.pt"
traindata, validdata, testdata = get_dataloader(
    "../video/multimodal_imdb.hdf5", "../video/mmimdb", vgg=True, batch_size=800)

outdim = 256
encoders = [MaxOut_MLP(512, 512, 300, outdim, False),
            MaxOut_MLP(512, 1024, 4096, outdim, False)]
head = Linear(2*outdim, 23).cuda()
fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 1000, early_stop=True, task="multilabel", save=filename, objective_args_dict={},
      optimtype=torch.optim.AdamW, lr=1e-2, weight_decay=0.01, objective=CCA_objective(outdim, criterion=torch.nn.BCEWithLogitsLoss()))

print("Testing:")
model = torch.load(filename).cuda()
test(model, testdata, method_name="cca", dataset="imdb",
     criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")
