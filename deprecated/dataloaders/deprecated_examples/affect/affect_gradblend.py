from private_test_scripts.all_in_one import all_in_one_train
from training_structures.gradient_blend import train, test
from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
traindata, validdata, testdata = get_dataloader(
    '/home/paul/MultiBench/mosi_raw.pkl')

# mosi/mosei
encoders = [GRU(35, 70, dropout=True, has_padding=True).cuda(),
            GRU(74, 150, dropout=True, has_padding=True).cuda(),
            GRU(300, 600, dropout=True, has_padding=True).cuda()]
head = MLP(820, 400, 1).cuda()
unimodal_heads = [MLP(70, 50, 1).cuda(), MLP(
    150, 100, 1).cuda(), MLP(600, 256, 1).cuda()]

# humor/sarcasm
# encoders=[GRU(371,512,dropout=True,has_padding=True).cuda(), \
#     GRU(81,256,dropout=True,has_padding=True).cuda(),\
#     GRU(300,600,dropout=True,has_padding=True).cuda()]
# head=MLP(1368,512,1).cuda()

all_modules = [*encoders, head, *unimodal_heads]

fusion = Concat().cuda()


def trainprocess():
    train(encoders, head, unimodal_heads, fusion, traindata,
          validdata, 300, lr=0.005, AUPRC=False, savedir='gb.pt')


all_in_one_train(trainprocess, all_modules)

print("Testing:")
model = torch.load('gb.pt').cuda()
test(model, testdata)
