from unimodals.common_models import MLP, VGG16, MaxOut_MLP, Linear
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Contrastive_Learning_2 import train, test
import torch
import sys
import os
sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader(
    '../video/multimodal_imdb.hdf5', vgg=True, batch_size=128)

encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False),
            MaxOut_MLP(512, 1024, 4096, 512, False)]
#encoders=[MLP(300, 512, 512), MLP(4096, 1000, 512)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
# head=MLP(1024,512,23).cuda()
head = Linear(1024, 23).cuda()
refiner = MLP(1024, 3072, 4396).cuda()
#refiner = MLP(1024,2048,1024).cuda()
fusion = Concat().cuda()

train(encoders, fusion, head, refiner, traindata, validdata, 1000, early_stop=True, task="multilabel",
      save="best_contrast.pt", optimtype=torch.optim.AdamW, lr=1e-2, weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())

print("Testing:")
model = torch.load('best_contrast.pt').cuda()
test(model, testdata, criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")
