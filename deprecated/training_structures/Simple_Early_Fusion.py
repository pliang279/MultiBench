from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from utils.AUPRC import AUPRC
from objective_functions.regularization import RegularizationLoss
# ADD MOSI MOSI metric
from utils.evaluation_metric import eval_mosei_senti

# import pdb

softmax = nn.Softmax()


class MMDL(nn.Module):
    def __init__(self, encoder, fusion, head, has_padding=False):
        super(MMDL, self).__init__()
        self.encoder = encoder
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding

    def forward(self, inputs, training=False):
        if self.has_padding:
            input_data = self.fuse(inputs[0])
            out = self.encoder([input_data, inputs[1][0]], training=training)
        else:
            input_data = self.fuse(inputs)
            #out = self.encoder(input_data, training=training)
            out = input_data

        
        return self.head(out, training=training)


def train(
        encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs, is_packed=False,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        criterion=nn.CrossEntropyLoss(), regularization=False, auprc=False, save='best.pt'):

    model = MMDL(encoders, fusion, head, is_packed).cuda()
    op = optimtype(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = ExponentialLR(op, 0.9)
    bestvalloss = 10000
    bestacc = 0
    bestf1 = 0
    patience = 0

    if regularization:
        regularize = RegularizationLoss(criterion, model, 1e-11)

    for epoch in range(total_epochs):
        totalloss = 0.0
        totalloss1 = 0.0
        totalloss2 = 0.0
        totals = 0
        model.train()
        for j in train_dataloader:
            
            op.zero_grad()
            if is_packed:
                with torch.backends.cudnn.flags(enabled=False):
                    out = model([[i.cuda()
                                for i in j[0]], j[1]], training=True)
                    
                    
                    loss1 = criterion(out, j[-1].cuda())
                    loss2 = regularize(
                        out, [[i.cuda() for i in j[0]], j[1]]) if regularization else 0
                    loss = loss1 + loss2
            else:
                out = model([i.float().cuda() for i in j[:-1]], training=True)
                
                if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                    loss1 = criterion(out, j[-1].float().cuda())
                else:
                    if len(j[-1].size()) > 1:
                        j[-1] = j[-1].squeeze()
                    loss1 = criterion(out, j[-1].long().cuda())
                loss2 = regularize(out, [i.float().cuda()
                                   for i in j[:-1]]) if regularization else 0
                loss = loss1+loss2
            
            totalloss += loss * len(j[-1])
            totals += len(j[-1])
            if regularization:
                totalloss1 += loss1 * len(j[-1])
                totalloss2 += loss2 * len(j[-1])
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8)
            op.step()
        if regularization:
            print("Epoch " + str(epoch) + " train loss: " + str(totalloss1 / totals) + " reg loss: " + str(
                totalloss2 / totals))
        else:
            print("Epoch " + str(epoch) +
                  " train loss: " + str(totalloss / totals))

        model.eval()
        with torch.no_grad():
            totalloss = 0.0
            pred = []
            true = []
            pts = []
            for j in valid_dataloader:
                if is_packed:
                    out = model([[i.cuda()
                                for i in j[0]], j[1]], training=False)
                else:
                    out = model([i.float().cuda()
                                for i in j[:-1]], training=False)
                if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                    loss = criterion(out, j[-1].float().cuda())
                else:
                    if len(j[-1].size()) > 1:
                        j[-1] = j[-1].squeeze()
                    loss = criterion(out, j[-1].long().cuda())
                totalloss += loss * len(j[-1])
                if task == "classification":
                    pred.append(torch.argmax(out, 1))
                elif task == "multilabel":
                    pred.append(torch.sigmoid(out).round())
                true.append(j[-1])
                if auprc:
                    # pdb.set_trace()
                    sm = softmax(out, 1)
                    pts += [(sm[i][1].item(), j[-1][i].item())
                            for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0).cpu().numpy()
        true = torch.cat(true, 0).cpu().numpy()
        totals = true.shape[0]
        valloss = totalloss / totals
        if task == "classification":
            acc = accuracy_score(true, pred)
            print("Epoch " + str(epoch) + " valid loss: " + str(valloss) +
                  " acc: " + str(acc))
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
            print("Epoch " + str(epoch) + " valid loss: " + str(valloss) +
                  " f1_micro: " + str(f1_micro) + " f1_macro: " + str(f1_macro))
            if f1_macro > bestf1:
                patience = 0
                bestf1 = f1_macro
                print("Saving Best")
                torch.save(model, save)
            else:
                patience += 1
        elif task == "regression":
            print("Epoch " + str(epoch) + " valid loss: " + str(valloss))
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
            print("AUPRC: " + str(AUPRC(pts)))

        # scheduler.step()


def test(
        model, test_dataloader, is_packed=False,
        criterion=nn.CrossEntropyLoss(), task="classification", auprc=False):
    with torch.no_grad():
        totalloss = 0.0
        pred = []
        true = []
        pts = []
        for j in test_dataloader:
            if is_packed:
                out = model([[i.cuda() for i in j[0]], j[1]], training=False)
            else:
                out = model([i.float().cuda() for i in j[:-1]], training=False)
            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                loss = criterion(out, j[-1].float().cuda())
            else:
                loss = criterion(out, j[-1].cuda())
            
            totalloss += loss * len(j[-1])
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            else:
                pred.append(out)
            true.append(j[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out, 1)
                pts += [(sm[i][1].item(), j[-1][i].item())
                        for i in range(j[-1].size(0))]

        if task == "classification":
            # eval mosi/mosei
            print('include 0:')
            eval_mosei_senti(torch.cat(pred, 0), torch.cat(
                true, 0), exclude_zero=False)
            print('exclude 0:')
            eval_mosei_senti(torch.cat(pred, 0), torch.cat(
                true, 0), exclude_zero=True)

        if pred:
            pred = torch.cat(pred, 0).cpu().numpy()
        true = torch.cat(true, 0).cpu().numpy()
        totals = true.shape[0]
        testloss = totalloss / totals
        if auprc:
            print("AUPRC: " + str(AUPRC(pts)))
        if task == "classification":
            print("acc: " + str(accuracy_score(true, pred)))
            return accuracy_score(true, pred)
        elif task == "multilabel":
            print(" f1_micro: " + str(f1_score(true, pred, average="micro")) +
                  " f1_macro: " + str(f1_score(true, pred, average="macro")))
            return f1_score(true, pred, average="micro"), f1_score(true, pred, average="macro")
        elif task == "regression":
            print("mse: " + str(testloss))
            return testloss.item()
