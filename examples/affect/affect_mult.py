import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from training_structures.Supervised_Learning import train, test # noqa
from fusions.mult import MULTModel # noqa
from unimodals.common_models import Identity, MLP # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa


# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, test_robust = get_dataloader('/home/paul/MultiBench/mosi_data.pkl', robust_test=False, max_pad=True)


class HParams():
        num_heads = 8
        layers = 4
        attn_dropout = 0.1
        attn_dropout_modalities = [0,0,0.1]
        relu_dropout = 0.1
        res_dropout = 0.1
        out_dropout = 0.1
        embed_dropout = 0.2
        embed_dim = 40
        attn_mask = True
        output_dim = 1
        all_steps = False

encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
fusion = MULTModel(3, [20, 5, 300], hyp_params=HParams).cuda()
# fusion = MULTModel(3, [371, 81, 300], hyp_params=HParams).cuda()
head = Identity().cuda()

train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, early_stop=False, is_packed=False, lr=1e-3, clip_val=1.0, save='mosi_mult_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('mosi_mult_best.pt').cuda()

test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=False,
     criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)
