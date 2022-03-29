import torch
import numpy as np
def test_blah():
    from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
    from training_structures.Supervised_Learning import train, test  # noqa
    from datasets.affect.get_data import get_dataloader  # noqa
    from fusions.common_fusions import ConcatEarly  # noqa

    data = [torch.zeros([32, 50, 35]), torch.zeros([32, 50, 74]), torch.zeros([32, 50, 300]), torch.cat((torch.ones((16,1)),torch.zeros((16,1))),dim=0)]
    my_dataset = torch.utils.data.TensorDataset(*data)
    dl = torch.utils.data.DataLoader(my_dataset)
    encoders = [Identity().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), Identity().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), Identity().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
    head = Sequential(GRU(409, 512, dropout=True, has_padding=False,
                  batch_first=True, last_only=True), MLP(512, 512, 1)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    fusion = ConcatEarly().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    train(encoders, fusion, head, dl, dl, 1, task="regression", optimtype=torch.optim.AdamW,
      is_packed=False, lr=1e-3, save='mosi_ef_r0.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

    model = torch.load('mosi_ef_r0.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    test(model, dl, 'affect', is_packed=False,
     criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)
