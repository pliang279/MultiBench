from unimodals.common_models import LeNet
import torch
from torch import nn
from torch.nn import functional as F



class MLPEncoder(torch.nn.Module):
    """Implements MLP Encoder for MVAE."""
    
    def __init__(self, indim, hiddim, outdim):
        """Initialzies MLPEncoder Object.

        Args:
            indim (int): Input Dimension
            hiddim (int): Hidden Dimension
            outdim (int): Output Dimension
        """
        super(MLPEncoder, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, 2*outdim)
        self.outdim = outdim

    def forward(self, x):
        output = self.fc(x)
        output = F.relu(output)
        output = self.fc2(output)
        return output[:, :self.outdim], output[:, self.outdim:]


class TSEncoder(torch.nn.Module):
    def __init__(self, indim, outdim, finaldim, timestep, returnvar=True, batch_first=False):
        super(TSEncoder, self).__init__()
        self.gru = nn.GRU(input_size=indim, hidden_size=outdim,
                          batch_first=batch_first)
        self.indim = indim
        self.ts = timestep
        self.finaldim = finaldim
        if returnvar:
            self.linear = nn.Linear(outdim*timestep, 2*finaldim)
        else:
            self.linear = nn.Linear(outdim*timestep, finaldim)
        self.returnvar = returnvar

    def forward(self, x):
        batch = len(x)
        input = x.reshape(batch, self.ts, self.indim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        output = self.linear(output.flatten(start_dim=1))
        if self.returnvar:
            return output[:, :self.finaldim], output[:, self.finaldim:]
        return output


class TSDecoder(torch.nn.Module):
    def __init__(self, indim, outdim, finaldim, timestep):
        super(TSDecoder, self).__init__()
        self.gru = nn.GRU(input_size=indim, hidden_size=indim)
        self.linear = nn.Linear(finaldim, indim)
        self.ts = timestep
        self.indim = indim

    def forward(self, x):
        
        hidden = self.linear(x).unsqueeze(0)
        next = torch.zeros(1, len(x), self.indim).cuda()
        nexts = []
        for i in range(self.ts):
            next, hidden = self.gru(next, hidden)
            nexts.append(next.squeeze(0))
        return torch.cat(nexts, 1)


class DeLeNet(nn.Module):
    def __init__(self, in_channels, arg_channels, additional_layers, latent):
        super(DeLeNet, self).__init__()
        self.linear = nn.Linear(latent, arg_channels*(2**(additional_layers)))
        self.deconvs = []
        self.bns = []
        for i in range(additional_layers):
            self.deconvs.append(nn.ConvTranspose2d(arg_channels*(2**(additional_layers-i)), arg_channels*(
                2**(additional_layers-i-1)), kernel_size=4, stride=2, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(
                arg_channels*(2**(additional_layers-i-1))))
        self.deconvs.append(nn.ConvTranspose2d(
            arg_channels, in_channels, kernel_size=8, stride=4, padding=1, bias=False))
        self.deconvs = nn.ModuleList(self.deconvs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = self.linear(x).unsqueeze(2).unsqueeze(3)
        for i in range(len(self.deconvs)):
            out = self.deconvs[i](out)
            
            if i < len(self.deconvs)-1:
                out = self.bns[i](out)
        return out


class LeNetEncoder(nn.Module):
    def __init__(self, in_channels, arg_channels, additional_layers, latent, twooutput=True):
        super(LeNetEncoder, self).__init__()
        self.latent = latent
        self.lenet = LeNet(in_channels, arg_channels, additional_layers)
        if twooutput:
            self.linear = nn.Linear(
                arg_channels*(2**additional_layers), latent*2)
        else:
            self.linear = nn.Linear(
                arg_channels*(2**additional_layers), latent)

        self.twoout = twooutput

    def forward(self, x):
        out = self.lenet(x)
        out = self.linear(out)
        if self.twoout:
            return out[:, :self.latent], out[:, self.latent:]
        return out
