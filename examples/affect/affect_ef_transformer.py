import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from unimodals.common_models import Transformer, MLP, Sequential, Identity # noqa
from private_test_scripts.all_in_one import all_in_one_train # noqa 
from training_structures.Supervised_Learning import train, test # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import ConcatEarly # noqa



# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_senti_data.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, testdata = get_dataloader(
    '/home/paul/MultiBench/mosei_senti_data.pkl', robust_test=False)

# mosi/mosei
encoders = [Identity().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), Identity().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), Identity().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
head = Sequential(Transformer(409, 300).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), MLP(300, 128, 1)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# humor/sarcasm
# encoders = [Identity().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),Identity().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),Identity().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
# head = Sequential(Transformer(752, 300).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),MLP(300, 128, 1)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


fusion = ConcatEarly().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, is_packed=True, early_stop=True,lr=1e-4, save='mosi_ef_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('mosi_ef_best.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
test(model, testdata, 'affect', is_packed=True,
     criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)
