import os
import sys
sys.path.append(os.getcwd())
from objective_functions.cca import *
from objective_functions.contrast import *
from objective_functions.objectives_for_supervised_learning import *
from objective_functions.recon import *
from objective_functions.regularization import *
from unimodals.common_models import MLP, Linear, GRU
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import MMDL
import numpy as np
from tests.common import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_CCALoss(set_seeds):
    loss_singular = CCALoss(10, True, device)
    assert loss_singular(torch.zeros((10,10)).to(device),torch.zeros((10,10)).to(device)).shape == ()
    out = loss_singular(torch.randn((10,10)).to(device), torch.randn((10,10)).to(device))
    assert np.isclose(out.item(),-9.280478477478027)
    loss = CCALoss(10, False, device)
    assert loss(torch.zeros((10,10)).to(device),torch.zeros((10,10)).to(device)).shape == ()
    out = loss(torch.randn((10,10)).to(device), torch.randn((10,10)).to(device))
    assert np.isclose(out.item(),-8.985898971557617)

def test_Alias(set_seeds):
    alias = AliasMethod(torch.tensor([3.0,1.0,2.0]).float())
    assert alias.draw(10).shape == (10,)
    assert torch.norm(alias.draw(10).float()).item() == 4.582575798034668

def test_MutlSimLoss(set_seeds):
    loss = MultiSimilarityLoss()
    assert loss(torch.zeros((10,10)),torch.cat([torch.zeros((5,10)),torch.ones((5,10))],dim=0)).shape == ()
    assert loss(torch.zeros((10,10)),torch.zeros((10,10))).item() == 0
    assert loss(torch.zeros((10,10)),torch.zeros((10,10))).shape == ()
    assert loss(torch.rand(10,10)*0.5+0.5,torch.cat([torch.zeros((1,10)),torch.ones((9,10))],dim=0)).item() == 0

def test_NCESoftmaxLoss(set_seeds):
    loss = NCESoftmaxLoss()
    assert loss(torch.ones((10,10)).to(device)).shape == ()
    assert np.isclose(loss(torch.ones((10,10)).to(device)).item(),2.3026)
    loss = NCECriterion(10)
    assert loss(torch.ones((10,10)).to(device)).shape == (1,)
    assert np.isclose(loss(torch.ones((10,10)).to(device)).item(),7.3668)
    NCEAverage(3,4,1)

def test_MFM_objective(set_seeds):
    objective1 = MFM_objective(2.0, [torch.nn.MSELoss(), torch.nn.MSELoss()], [1.0, 1.0])
    objective2 = MFM_objective(2.0, [torch.nn.MSELoss(), torch.nn.MSELoss()], [1.0, 1.0], input_to_float=False)
    decoders = [Linear(45, 10).to(device) for _ in range(2)]
    intermediates = [MLP(10, 15, 15).to(device) for _ in range(2)]
    args = dict()
    args['decoders'] = decoders
    args['intermediates'] = intermediates
    args['reps'] = [torch.randn((10,10)).to(device),torch.randn((10,10)).to(device)]
    args['fused'] = torch.randn((10,30)).to(device)
    args['inputs'] = [torch.randn((10,10)).to(device),torch.randn((10,10)).to(device)]
    assert np.isclose(objective1(torch.rand((10,10)).to(device),torch.randint(low=0, high=10, size=(10,)).to(device),args).item(),7.1342926025390625)
    assert np.isclose(objective2(torch.rand((10,10)).to(device),torch.randint(low=0, high=10, size=(10,)).to(device),args).item(),7.19914436340332)

def test_MVAE_objective(set_seeds):
    objective1 = MVAE_objective(2.0, [torch.nn.MSELoss(), torch.nn.MSELoss()], [1.0, 1.0])
    decoders = [Linear(10, 10).to(device) for _ in range(2)]
    args = dict()
    args['decoders'] = decoders
    args['reps'] = [(torch.randn((10,10)).to(device),torch.randn((10,10)).to(device)),(torch.randn((10,10)).to(device),torch.randn((10,10)).to(device))]
    args['fused'] = [torch.randn((10,10)).to(device),torch.randn((10,10)).to(device)]
    args['inputs'] = [torch.randn((10,10)).to(device),torch.randn((10,10)).to(device)]
    args['training'] = True
    out = objective1(torch.randn((10,10)).to(device),torch.randint(low=0, high=10, size=(10,)).to(device),args).cpu().detach().numpy()
    assert np.isclose(out,45.058445)
    args['training'] = False
    out = objective1(torch.randn((10,10)).to(device),torch.randint(low=0, high=10, size=(10,)).to(device),args).cpu().detach().numpy()
    assert np.isclose(out,42.121914)
    objective2 = MVAE_objective(2.0, [torch.nn.MSELoss(), torch.nn.MSELoss()], [1.0, 1.0], input_to_float=False)
    out = objective2(torch.randn((10,10)).to(device),torch.randint(low=0, high=10, size=(10,)).to(device),args).cpu().detach().numpy()
    assert np.isclose(out,41.353207)

def test_CCA_objective(set_seeds):
    objective = CCA_objective(10)
    args = dict()
    args['reps'] = [torch.rand((10,10)).to(device),torch.rand((10,10)).to(device)]
    assert np.isclose(objective(torch.rand((10,10)).to(device),torch.randint(low=0, high=10, size=(10,)).to(device),args).item(),2.27751851)

def test_RefNet_objective(set_seeds):
    objective1 = RefNet_objective(0.1)
    objective2 = RefNet_objective(0.1, input_to_float=False)
    args = dict()
    args['fused'] = torch.rand((10,20)).to(device)
    args['inputs'] = [torch.rand((10,10)).to(device),torch.rand((10,10)).to(device)]
    args['refiner'] = MLP(20, 50, 100).to(device)
    assert np.isclose(objective1(torch.rand((10,10)).to(device),torch.randint(low=0, high=10, size=(10,)).to(device),args).item(),2.615685)
    assert np.isclose(objective2(torch.rand((10,10)).to(device),torch.randint(low=0, high=10, size=(10,)).to(device),args).item(),2.4153285)

def test_RMFE_object(set_seeds):
    objective = RMFE_object()
    encoders = [Linear(10, 10).to(device) for _ in range(2)]
    fusion = Concat().to(device)
    head = Linear(20, 10).to(device)
    model = MMDL(encoders, fusion, head,).to(device)
    args = dict()
    args['model'] = model
    args['inputs'] = [torch.rand((10,10)).to(device),torch.rand((10,10)).to(device)]
    assert np.isclose(objective(torch.rand((10,10)).to(device),torch.ones([10,10]).to(device),args).item(),0.4805612862110138)
    args['inputs'] = []
    try:
        objective(torch.rand((10,10)).to(device),torch.ones([10,10]).to(device),args)
    except Exception as e:
        assert isinstance(e, RuntimeError)

def test_sigmloss1d(set_seeds):
    a = torch.rand((1,10))
    b = torch.randint(low=0, high=10, size=(1,10))
    assert np.isclose(sigmloss1d(a,b).item(),0.50266945)

def test_nosigmloss1d(set_seeds):
    a = torch.rand((1,10))
    b = torch.randint(low=0, high=10, size=(1,10))
    assert np.isclose(nosigmloss1d(a,b).item(),1.7327522)

def test_sigmloss1dcentercrop(set_seeds):
    loss = sigmloss1dcentercrop(5, 10)
    a = torch.rand((5,5,6,6))
    b = torch.randint(low=0, high=10, size=(5,5,10,10))
    assert np.isclose(torch.norm(loss(a,b)).item(),1.1827)
    a = torch.rand((5,5,10,10))
    b = torch.randint(low=0, high=10, size=(5,5,6,6))
    assert np.isclose(torch.norm(loss(a,b)).item(),4.101116)

def test_elbo_loss(set_seeds):
    loss = elbo_loss([torch.nn.MSELoss(), torch.nn.MSELoss()], [1.0, 1.0])
    recons = torch.rand((2,10))
    origs = torch.rand((2,10))
    mu = torch.mean(recons)
    logvar = torch.log(torch.var(recons)) + 1000000
    assert np.isclose(loss(recons, origs, mu, logvar).item(),999997.5625)

def test_Regularization(set_seeds):
    re = Regularization()
    assert np.isclose(re.get_batch_statistics(torch.rand(10,), 10).item(),0.01460908)
    assert np.isclose(re.get_batch_statistics(torch.rand(10,), 10, estimation='var').item(),0.05936414)
    assert np.isclose(re.get_batch_statistics(torch.rand(10,), 10, estimation='dif_ent').item(),2.36509323)
    assert np.isclose(re.get_batch_norm(torch.rand((10,10)), torch.rand((10,))).item(),7.3596701622)
    assert np.isclose(re.get_regularization_term(torch.rand((10,))).item(),22.45313263)
    assert np.isclose(re.get_regularization_term(torch.rand((10,)), optim_method='min_ent').item(),1.5079478)
    assert np.isclose(re.get_regularization_term(torch.rand((10,)), optim_method='max_ent_minus').item(),-1.138665)
    try:
        re.get_regularization_term(torch.rand((10,)), optim_method='')
    except Exception as e:
        assert isinstance(e, NotImplementedError)

def test_RegularizationLoss(set_seeds):
    encoders = [GRU(10, 5, dropout=True, has_padding=True, batch_first=True).to(device),GRU(10, 5, dropout=True, has_padding=True, batch_first=True).to(device)]
    fusion = Concat().to(device)
    head = Linear(10, 2).to(device)
    model = MMDL(encoders, fusion, head, has_padding=True).to(device)
    rgl = RegularizationLoss(torch.nn.BCEWithLogitsLoss(), model)
    logits = torch.rand((1,2)).to(device)
    inputs = [torch.rand((1,10,10)).to(device),torch.rand((1,10,10)).to(device)]
    inputs_len = torch.tensor([[10],[10]])
    assert np.isclose(rgl(logits, (inputs, inputs_len)).item(),0.004487053025513887) 
