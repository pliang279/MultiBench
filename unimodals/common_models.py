import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as tmodels

class MLP(torch.nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=False,dropoutp=0.1):
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim,hiddim)
        self.fc2 = nn.Linear(hiddim,outdim)
        self.dropoutp = dropoutp
        self.dropout = dropout
    def forward(self, x, training=True):
        output = F.relu(self.fc(x))
        if self.dropout:
            output = F.dropout(output,p=self.dropout,training=training)
        output = self.fc2(output)
        if self.dropout:
            output = F.dropout(output,p=self.dropoutp,training=training)
        return output


class GRU(torch.nn.Module):
    def __init__(self,indim,hiddim,dropout=False,dropoutp=0.1):
        super(GRU,self).__init__()
        self.gru=nn.GRU(indim,hiddim)
        self.dropoutp=dropoutp
        self.dropout=dropout
    def forward(self,x,training=True):
        out=self.gru(x)[0]
        if self.dropout:
            out = F.dropout(out,p=self.dropoutp,training=training)
        return out


#adapted from centralnet code https://github.com/slyviacassell/_MFAS/blob/master/models/central/avmnist.py
class LeNet(nn.Module):
    def __init__(self,in_channels,args_channels,additional_layers,output_each_layer=False,linear=None,squeeze_output=True):
        super(LeNet,self).__init__()
        self.output_each_layer=output_each_layer
        self.convs=[nn.Conv2d(in_channels,args_channels,kernel_size=5,padding=2,bias=False)]
        self.bns=[nn.BatchNorm2d(args_channels)]
        self.gps=[GlobalPooling2D()]
        for i in range(additional_layers):
            self.convs.append(nn.Conv2d((2**i)*args_channels,(2**(i+1))*args_channels,kernel_size=3,padding=1,bias=False))
            self.bns.append(nn.BatchNorm2d(args_channels*(2**(i+1))))
            self.gps.append(GlobalPooling2D())
        self.convs=nn.ModuleList(self.convs)
        self.bns=nn.ModuleList(self.bns)
        self.gps=nn.ModuleList(self.gps)
        self.sq_out=squeeze_output
        self.linear=None
        if linear is not None:
            self.linear=nn.Linear(linear[0],linear[1])
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)
    def forward(self,x,training=False):
        tempouts=[]
        out=x
        for i in range(len(self.convs)):
            out=F.relu(self.bns[i](self.convs[i](out)))
            out=F.max_pool2d(out,2)
            gp=self.gps[i](out)
            tempouts.append(gp)
            #print(out.size())
        if self.linear is not None:
            out=self.linear(out)
        tempouts.append(out)
        if self.output_each_layer:
            if self.sq_out:
                return [t.squeeze() for t in tempouts]
            return tempouts
        if self.sq_out:
            return out.squeeze()
        return out


class VGG(nn.Module):
    def __init__(self, num_outputs):
        super(VGG, self).__init__()

        # self.vgg = tmodels.vgg19(pretrained='imagenet')
        vgg = list(tmodels.vgg19(pretrained='imagenet').features)
        self.vgg = nn.ModuleList(vgg)
        self.gp1 = GlobalPooling2D()
        self.gp2 = GlobalPooling2D()
        self.gp3 = GlobalPooling2D()
        self.gp4 = GlobalPooling2D()

        self.bn4 = nn.BatchNorm1d(512)  # only used for classifier

        self.classifier = nn.Linear(512, num_outputs)

    def forward(self, x):

        for i_l, layer in enumerate(self.vgg):

            x = layer(x)

            if i_l == 20:
                out_1 = self.gp1(x)

            if i_l == 26:
                out_2 = self.gp2(x)

            if i_l == 33:
                out_3 = self.gp3(x)

            if i_l == 36:
                out_4 = self.gp4(x)
                bn_4 = self.bn4(out_4)

        out = self.classifier(bn_4)

        # print()
        # print(out_4, out)

        return out_1, out_2, out_3, out_4, out


class Maxout(nn.Module):
    def __init__(self, d, m, k):
        super(Maxout, self).__init__()
        self.d_in, self.d_out, self.pool_size = d, m, k
        self.lin = nn.Linear(d, m * k)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(dim=max_dim)
        return m


class MaxOut_MLP(nn.Module):

    def __init__(self, num_outputs, first_hidden=64, number_input_feats=300):
        super(MaxOut_MLP, self).__init__()

        self.op1 = Maxout(number_input_feats, first_hidden, 5)
        self.op2 = nn.Sequential(nn.BatchNorm1d(first_hidden), nn.Dropout(0.5))
        self.op3 = Maxout(first_hidden, first_hidden * 2, 5)
        self.op4 = nn.Sequential(nn.BatchNorm1d(first_hidden * 2), nn.Dropout(0.5))

        # The linear layer that maps from hidden state space to output space
        self.hid2val = nn.Linear(first_hidden * 2, num_outputs)

    def forward(self, x):
        o1 = self.op1(x)
        o2 = self.op2(o1)
        o3 = self.op3(o2)
        o4 = self.op4(o3)
        o5 = self.hid2val(o4)

        return o1, o3, o5


class GlobalPooling2D(nn.Module):
    def __init__(self):
        super(GlobalPooling2D, self).__init__()

    def forward(self, x):
        # apply global average pooling
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        x = x.view(x.size(0), -1)

        return x


class Constant(nn.Module):
    def __init__(self,out_dim):
        super(Constant,self).__init__()
        self.out_dim=out_dim
    def forward(self,x,training=False):
        return torch.zeros(self.out_dim).cuda()
