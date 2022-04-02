"""Implements various encoders and decoders for MVAE."""
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
        """Apply MLPEncoder to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        output = self.fc(x)
        output = F.relu(output)
        output = self.fc2(output)
        return output[:, :self.outdim], output[:, self.outdim:]


class TSEncoder(torch.nn.Module):
    """Implements a time series encoder for MVAE."""
    
    def __init__(self, indim, outdim, finaldim, timestep, returnvar=True, batch_first=False):
        """Instantiate TSEncoder Module.

        Args:
            indim (int): Input Dimension of GRU
            outdim (int): Output dimension of GRU
            finaldim (int): Output dimension of TSEncoder
            timestep (float): Number of timestamps
            returnvar (bool, optional): Whether to return the output split with the first encoded portion and the next or not. Defaults to True.
            batch_first (bool, optional): Whether the batching dimension is the first dimension of the input or not. Defaults to False.
        """
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
        """Apply TS Encoder to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        batch = len(x)
        input = x.reshape(batch, self.ts, self.indim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        output = self.linear(output.flatten(start_dim=1))
        if self.returnvar:
            return output[:, :self.finaldim], output[:, self.finaldim:]
        return output


class TSDecoder(torch.nn.Module):
    """Implements a time-series decoder for MVAE."""
    
    def __init__(self, indim, outdim, finaldim, timestep):
        """Instantiate TSDecoder Module.

        Args:
            indim (int): Input dimension
            outdim (int): (unused) Output dimension
            finaldim (int): Hidden dimension
            timestep (int): Number of timesteps
        """
        super(TSDecoder, self).__init__()
        self.gru = nn.GRU(input_size=indim, hidden_size=indim)
        self.linear = nn.Linear(finaldim, indim)
        self.ts = timestep
        self.indim = indim

    def forward(self, x):
        """Apply TSDecoder to layer input.

        Args:
            x (torch.Tensor): Layer Input   

        Returns:
            torch.Tensor: Layer Output
        """
        hidden = self.linear(x).unsqueeze(0)
        next = torch.zeros(1, len(x), self.indim).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        nexts = []
        for i in range(self.ts):
            next, hidden = self.gru(next, hidden)
            nexts.append(next.squeeze(0))
        return torch.cat(nexts, 1)


class DeLeNet(nn.Module):
    """Implements an image deconvolution decoder for MVAE."""
    
    def __init__(self, in_channels, arg_channels, additional_layers, latent):
        """Instantiate DeLeNet Module.

        Args:
            in_channels (int): Number of input channels
            arg_channels (int): Number of arg channels
            additional_layers (int): Number of additional layers.
            latent (int): Latent dimension size
        """
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
        """Apply DeLeNet to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        out = self.linear(x).unsqueeze(2).unsqueeze(3)
        for i in range(len(self.deconvs)):
            out = self.deconvs[i](out)
            
            if i < len(self.deconvs)-1:
                out = self.bns[i](out)
        return out


class LeNetEncoder(nn.Module):
    """Implements a LeNet Encoder for MVAE."""
    
    def __init__(self, in_channels, arg_channels, additional_layers, latent, twooutput=True):
        """Instantiate LeNetEncoder Module

        Args:
            in_channels (int): Input Dimensions
            arg_channels (int): Arg channels dimension size
            additional_layers (int): Number of additional layers
            latent (int): Latent dimension size
            twooutput (bool, optional): Whether to output twice the size of the latent. Defaults to True.
        """
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
        """Apply LeNetEncoder to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        out = self.lenet(x)
        out = self.linear(out)
        if self.twoout:
            return out[:, :self.latent], out[:, self.latent:]
        return out
