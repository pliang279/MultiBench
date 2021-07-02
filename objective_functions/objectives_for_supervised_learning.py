from objective_functions.recon import recon_weighted_sum,elbo_loss
import torch
from objective_functions.cca import CCALoss
from objective_functions.regularization import RegularizationLoss

# deals with some built-in criterions
def criterioning(pred,truth,criterion):
    if type(criterion)==torch.nn.CrossEntropyLoss:
        return criterion(pred,truth.long().cuda())
    elif type(criterion)==torch.nn.modules.loss.BCEWithLogitsLoss or type(criterion)==torch.nn.MSELoss:
        return criterion(pred,truth.float().cuda())

# objective for MFM
# ce_weight: weight of simple supervised loss
# modal_loss_funcs: list of functions that takes in reconstruction and input of each modality and compute reconstruction loss
# recon_weights: list of float values indicating the weight of reconstruction loss of each modality
# criterion: the criterion for supervised loss
def MFM_objective(ce_weight,modal_loss_funcs,recon_weights,input_to_float=True,criterion=torch.nn.CrossEntropyLoss()):
    recon_loss_func = recon_weighted_sum(modal_loss_funcs,recon_weights)
    def actualfunc(pred,truth,args):
        ints = args['intermediates']
        reps = args['reps']
        fused = args['fused']
        decoders = args['decoders']
        inps = args['inputs']
        recons = []
        for i in range(len(reps)):
            recons.append(decoders[i](torch.cat([ints[i](reps[i]),fused],dim=1)))
        ce_loss = criterioning(pred,truth,criterion)
        if input_to_float:
            inputs = [i.float().cuda() for i in inps]
        else:
            inputs = [i.cuda() for i in inps]
        recon_loss = recon_loss_func(recons,inputs)
        return ce_loss*ce_weight+recon_loss
    return actualfunc

def reparameterize(mu, logvar, training):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu

# ce_weight: weight of simple supervised loss
# modal_loss_funcs: list of functions that takes in reconstruction and input of each modality and compute reconstruction loss
# recon_weights: list of float values indicating the weight of reconstruction loss of each modality
# criterion: the criterion for supervised loss
# annealing: the annealing factor (i.e. weight of kl)
# input_to_float: whether to convert input to float or not
def MVAE_objective(ce_weight,modal_loss_funcs,recon_weights,input_to_float=True,annealing=1.0,criterion=torch.nn.CrossEntropyLoss()):
    recon_loss_func = elbo_loss(modal_loss_funcs,recon_weights,annealing)
    def allnonebuti(i,item):
        ret=[None for w in modal_loss_funcs]
        ret[i]=item
        return ret
    def actualfunc(pred,truth,args):
        training = args['training']
        reps = args['reps']
        fusedmu,fusedlogvar = args['fused']
        decoders = args['decoders']
        inps = args['inputs']
        reconsjoint = []
        
        if input_to_float:
            inputs = [i.float().cuda() for i in inps]
        else:
            inputs = [i.cuda() for i in inps]
        for i in range(len(inps)):
            reconsjoint.append(decoders[i](reparameterize(fusedmu,fusedlogvar,training)))
        total_loss = recon_loss_func(reconsjoint,inputs,fusedmu,fusedlogvar)
        for i in range(len(inps)):
            mu,logvar = reps[i]
            recon = decoders[i](reparameterize(mu,logvar,training))
            total_loss += recon_loss_func(allnonebuti(i,recon),allnonebuti(i,inputs[i]),mu,logvar)
        total_loss += ce_weight * criterioning(pred,truth,criterion)
        return total_loss
    return actualfunc

# out_dim: output dimension
# cca_weight: weight of cca loss
# criterion: criterion for supervised loss
def CCA_objective(out_dim,cca_weight=0.001,criterion=torch.nn.CrossEntropyLoss()):
    lossfunc = CCALoss(out_dim,False, device=torch.device("cuda"))
    def actualfunc(pred,truth,args):
        ce_loss = criterioning(pred,truth,criterion)
        outs = args['reps']
        cca_loss = lossfunc(outs[0],outs[1])
        return cca_loss * cca_weight + ce_loss
    return actualfunc

# ref_weight: weight of refiner loss
# criterion: criterion for supervised loss
# input_to_float: whether to convert input to float or not
def RefNet_objective(ref_weight,criterion=torch.nn.CrossEntropyLoss(),input_to_float=True):
    ss_criterion=torch.nn.CosineEmbeddingLoss()
    def actualfunc(pred,truth,args):
        ce_loss = criterioning(pred,truth,criterion)
        refiner = args['refiner']
        fused = args['fused']
        inps = args['inputs']
        refinerout = refiner(fused)
        if input_to_float:
            inputs = [torch.flatten(t,start_dim=1).float().cuda() for t in inps]
        else:
            inputs = [torch.flatten(t,start_dim=1).cuda() for t in inps]

        inputsizes = [t.size(1) for t in inputs]
        ss_loss=0.0
        loc=0
        for i in range(len(inps)):
            out = refinerout[:,loc:loc+inputsizes[i]]
            loc += inputsizes[i]
            ss_loss += ss_criterion(out,inputs[i],torch.ones(out.size(0)).cuda())
        return ce_loss + ss_loss*ref_weight
    return actualfunc

# model: model used for inference
# reg_weight: weight of regularization term
# criterion: criterion for supervised loss
# is_packed: packed for LSTM or not
def RMFE_object(reg_weight=1e-10, criterion=torch.nn.BCEWithLogitsLoss(), is_packed=False):
    def regfunc(pred, truth, args):
        model=args['model']
        lossfunc = RegularizationLoss(criterion, model, reg_weight, is_packed)
        ce_loss = criterioning(pred,truth,criterion)
        inps = args['inputs']
        try:
            reg_loss = lossfunc(pred, [i.cuda() for i in inps])
        except RuntimeError:
            print("No reg loss for validation")
            reg_loss = 0
        return ce_loss+reg_loss
    return regfunc
