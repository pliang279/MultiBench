
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import time
from eval_scripts.performance import AUPRC, f1_score, accuracy
#import pdb

softmax = nn.Softmax()


class MMDL(nn.Module):
    def __init__(self, encoders, fusion, head, has_padding=False):
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
        self.reps = outs
        out = self.fuse(outs)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        return self.head(out)


def deal_with_objective(objective, pred, truth, args):
    if type(objective) == nn.CrossEntropyLoss:
        if len(truth.size()) == len(pred.size()):
            truth1 = truth.squeeze(len(pred.size())-1)
        else:
            truth1 = truth
        return objective(pred, truth1.long().cuda())
    elif type(objective) == nn.MSELoss or type(objective) == nn.modules.loss.BCEWithLogitsLoss:
        return objective(pred, truth.float().cuda())
    else:
        return objective(pred, truth, args)


def train(
        encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs, additional_optimizing_modules=[], is_packed=False,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8):

    model = MMDL(encoders, fusion, head, is_packed).cuda()
    additional_params = []
    for m in additional_optimizing_modules:
        additional_params.extend(
            [p for p in m.parameters() if p.requires_grad])
    op = optimtype([p for p in model.parameters() if p.requires_grad] +
                   additional_params, lr=lr, weight_decay=weight_decay)
    bestvalloss = 10000
    bestacc = 0
    bestf1 = 0
    patience = 0

    def processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp

    for epoch in range(total_epochs):
        totalloss = 0.0
        totals = 0
        model.train()
        for j in train_dataloader:
            op.zero_grad()
            if is_packed:
                with torch.backends.cudnn.flags(enabled=False):
                    out = model([[processinput(i).cuda()
                                for i in j[0]], j[1]])

            else:
                out = model([processinput(i).cuda()
                            for i in j[:-1]])
            if not (objective_args_dict is None):
                objective_args_dict['reps'] = model.reps
                objective_args_dict['fused'] = model.fuseout
                objective_args_dict['inputs'] = j[:-1]
                objective_args_dict['training'] = True
            loss = deal_with_objective(
                objective, out, j[-1], objective_args_dict)

            totalloss += loss * len(j[-1])
            totals += len(j[-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            op.step()
        print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
        validstarttime = time.time()
        if validtime:
            print("train total: "+str(totals))
        model.eval()
        with torch.no_grad():
            totalloss = 0.0
            pred = []
            true = []
            pts = []
            for j in valid_dataloader:
                if is_packed:
                    out = model([[processinput(i).cuda()
                                for i in j[0]], j[1]])
                else:
                    out = model([processinput(i).cuda()
                                for i in j[:-1]])

                if not (objective_args_dict is None):
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = False
                loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)
                totalloss += loss*len(j[-1])
                
                if task == "classification":
                    pred.append(torch.argmax(out, 1))
                elif task == "multilabel":
                    pred.append(torch.sigmoid(out).round())
                true.append(j[-1])
                if auprc:
                    # pdb.set_trace()
                    sm = softmax(out)
                    pts += [(sm[i][1].item(), j[-1][i].item())
                            for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0)
        true = torch.cat(true, 0)
        totals = true.shape[0]
        valloss = totalloss/totals
        if task == "classification":
            acc = accuracy(true, pred)
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                  " acc: "+str(acc))
            if acc > bestacc:
                patience = 0
                bestacc = acc
                print("Saving Best")
                torch.save(model, save)
            else:
                patience += 1
        elif task == "multilabel":
            f1_micro = f1_score(true, pred, average="micro")
            f1_macro = f1_score(true, pred, average="macro")
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                  " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
            if f1_macro > bestf1:
                patience = 0
                bestf1 = f1_macro
                print("Saving Best")
                torch.save(model, save)
            else:
                patience += 1
        elif task == "regression":
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
            if valloss < bestvalloss:
                patience = 0
                bestvalloss = valloss
                print("Saving Best")
                torch.save(model, save)
            else:
                patience += 1
        if early_stop and patience > 7:
            break
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        validendtime = time.time()
        if validtime:
            print("valid time:  "+str(validendtime-validstarttime))
            print("Valid total: "+str(totals))

        # scheduler.step()


def test(
        model, test_dataloader, is_packed=False,
        criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True):
    def processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp
    with torch.no_grad():
        totalloss = 0.0
        pred = []
        true = []
        pts = []
        model.eval()
        for j in test_dataloader:
            if is_packed:
                out = model([[processinput(i).cuda()
                            for i in j[0]], j[1]])
            else:
                out = model([processinput(i).float().cuda()
                            for i in j[:-1]])
            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss or type(criterion) == torch.nn.MSELoss:
                loss = criterion(out, j[-1].float().cuda())
            else:
                loss = criterion(out, j[-1].cuda())
            totalloss += loss*len(j[-1])
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            true.append(j[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out)
                pts += [(sm[i][1].item(), j[-1][i].item())
                        for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0)
        true = torch.cat(true, 0)
        totals = true.shape[0]
        testloss = totalloss/totals
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if task == "classification":
            print("acc: "+str(accuracy(true, pred)))
            return accuracy(true, pred)
        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro")) +
                  " f1_macro: "+str(f1_score(true, pred, average="macro")))
            return f1_score(true, pred, average="micro"), f1_score(true, pred, average="macro")
        elif task == "regression":
            print("mse: "+str(testloss.item()))
            return testloss.item()
