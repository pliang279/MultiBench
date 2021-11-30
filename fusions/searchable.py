import torch
import copy
import torch.optim as op
from torch import nn
import utils.aux_models as aux
import utils.scheduler as sc

# See https://github.com/slyviacassell/_MFAS/tree/master/models for hyperparameter details


def train_sampled_models(sampled_configurations, searchable_type, dataloaders,
                         use_weightsharing, device, unimodal_files, rep_size, classes, sub_sizes, batch_size, epochs,
                         eta_max, eta_min, Ti, Tm,
                         return_model=False, premodels=False, preaccuracies=False,
                         train_only_central_params=True,
                         state_dict=dict()):

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'dev']}
    num_batches_per_epoch = dataset_sizes['train'] / batch_size
    criterion = torch.nn.CrossEntropyLoss()

    real_accuracies = []

    if return_model:
        models = []

    for idx, configuration in enumerate(sampled_configurations):

        if not return_model or idx in return_model:

            # model to train
            if not premodels:
                sds = []
                for i in unimodal_files:
                    sds.append(torch.load(i))
                for sd in sds:
                    sd.output_each_layer = True
                rmode = searchable_type(
                    sds, rep_size, classes, configuration, sub_sizes)

            if train_only_central_params:
                params = rmode.central_params()

            # optimizer and scheduler
            optimizer = op.Adam(params, lr=eta_max, weight_decay=1e-4)
            scheduler = sc.LRCosineAnnealingScheduler(eta_max, eta_min, Ti, Tm,
                                                      num_batches_per_epoch)

            rmode.to(device)

            best_model_acc = train_track_acc(rmode, [criterion], optimizer, scheduler, dataloaders,
                                             dataset_sizes,
                                             device=device, num_epochs=epochs, verbose=False,
                                             multitask=False)

            real_accuracies.append(best_model_acc)

            if return_model:
                models.append(rmode)

    if return_model:
        return real_accuracies, models
    else:
        return real_accuracies


def train_track_acc(model, criteria, optimizer, scheduler, dataloaders, dataset_sizes,
                    device=None, num_epochs=200, verbose=False, multitask=False):
    best_model_sd = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:

            if phase == 'train':
                if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                    scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:

                # get the inputs
                inputs = [d.float().cuda() for d in data[:-1]]
                label = data[-1].cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs)

                    if not multitask:
                        _, preds = torch.max(output.detach(), 1)
                        loss = criteria[0](output, label)
                    else:
                        _, preds = torch.max(sum(output), 1)
                        loss = criteria[0](output[0], label) + criteria[1](output[1], label) + criteria[2](output[2],
                                                                                                           label)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                            scheduler.step()
                            scheduler.update_optimizer(optimizer)
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * data[0].size(0)
                running_corrects += torch.sum(preds == label.detach())

            epoch_acc = torch.true_divide(
                running_corrects, dataset_sizes[phase])

            print('{} Acc: {:.4f}'.format(phase, epoch_acc))

            # deep copy the model
            if phase == 'dev' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_sd = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_sd)
    model.train(False)
    torch.save(model, 'temp/best'+str(best_acc)+'.pt')

    return best_acc


class Searchable(nn.Module):
    def __init__(self, layered_encoders, rep_size, classes, conf, sub_sizes, alphas=False):
        super(Searchable, self).__init__()
        self.encoders = nn.ModuleList(layered_encoders)
        self.using_alphas = alphas
        self.conf = conf
        self.subs = sub_sizes
        self.hidden = rep_size
        self.classes = classes
        if alphas:
            self.alphas = self.alphasgen()
        self.fusion_layers = self.fcs()
        self.head = nn.Linear(rep_size, classes)
        for m in self.modules():
            if isinstance(m, aux.AlphaScalarMultiplication):
                nn.init.normal_(m.alpha_x, 0.0, 0.1)

    def forward(self, inputs):
        features = []
        for i in range(len(inputs)):
            feat = self.encoders[i](inputs[i])[1:]
            features.append([feat[idx] for idx in self.conf[:, i]])

        for layer, conf in enumerate(self.conf):
            feats = [f[layer] for f in features]
            if self.using_alphas:
                aout = self.alphas[layer](feats)
            else:
                aout = feats
            if layer == 0:
                fused = torch.cat(aout, 1)
                
                out = self.fusion_layers[layer](fused)
            else:
                aout.append(out)
                fused = torch.cat(aout, 1)
                out = self.fusion_layers[layer](fused)
        out = self.head(out)
        return out

    def central_params(self):
        if self.using_alphas:
            cent = [{'params': self.alphas.parameters()}, {'params': self.fusion_layers.parameters()}, {
                'params': self.head.parameters()}]
        else:
            cent = [{'params': self.fusion_layers.parameters()}, {
                'params': self.head.parameters()}]
        return cent

    def fcs(self):
        fusion_layers = []
        for i, conf in enumerate(self.conf):
            in_size = 0
            for j in range(len(self.encoders)):
                in_size += self.subs[j][self.conf[i][j]]
            if i > 0:
                in_size += self.hidden
            if conf[-1] == 0:
                nl = nn.ReLU()
            elif conf[-1] == 1:
                nl = nn.Sigmoid()
            elif conf[-1] == 2:
                nl = nn.LeakyReLU()
            op = nn.Sequential(nn.Linear(in_size, self.hidden), nl)
            fusion_layers.append(op)
        return nn.ModuleList(fusion_layers)

    def alphasgen():
        alphas = [aux.AlphaScalarMultiplication(
            self.subs[0][conf[0]], self.subs[0][conf[1]]) for conf in self.conf]
        return nn.ModuleList(alphas)


def get_possible_layer_configurations(max_labels):
    list_conf = []
    if len(max_labels) == 1:
        for a in range(max_labels[0]):
            list_conf.append([a])
    else:
        b = max_labels[1:]
        for a in range(max_labels[0]):
            li = get_possible_layer_configurations(b)
            for k in li:
                k.insert(0, a)
            list_conf.extend(li)
    return list_conf
