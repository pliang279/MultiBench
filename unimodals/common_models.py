"""Implements common unimodal encoders."""
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models as tmodels




class Linear(torch.nn.Module):
    """Linear Layer with Xavier Initialization, and 0 Bias."""
    
    def __init__(self, indim, outdim, xavier_init=False):
        """Initialize Linear Layer w/ Xavier Init.

        Args:
            indim (int): Input Dimension
            outdim (int): Output Dimension
            xavier_init (bool, optional): Whether to apply Xavier Initialization to Layer. Defaults to False.
        
        """
        super(Linear, self).__init__()
        self.fc = nn.Linear(indim, outdim)
        if xavier_init:
            nn.init.xavier_normal(self.fc.weight)
            self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        """Apply Linear Layer to Input.

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor
        
        """
        return self.fc(x)


class Squeeze(torch.nn.Module):
    """Custom squeeze module for easier Sequential usage."""
    
    def __init__(self, dim=None):
        """Initialize Squeeze Module.

        Args:
            dim (int, optional): Dimension to Squeeze on. Defaults to None.
        """ 
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Apply Squeeze Layer to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.dim is None:
            return torch.squeeze(x)
        else:
            return torch.squeeze(x, self.dim)


class Sequential(nn.Sequential):
    """Custom Sequential module for easier usage."""
    
    def __init__(self, *args, **kwargs):
        """Initialize Sequential Layer."""
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Apply args to Sequential Layer."""
        if 'training' in kwargs:
            del kwargs['training']
        return super().forward(*args, **kwargs)


class Reshape(nn.Module):
    """Custom reshape module for easier Sequential usage."""
    
    def __init__(self, shape):
        """Initialize Reshape Module.

        Args:
            shape (tuple): Tuple to reshape input to
        """
        super().__init__()
        self.shape = shape

    def forward(self, x):
        """Apply Reshape Module to Input.

        Args:
            x (torch.Tensor): Layer Input 

        Returns:
            torch.Tensor: Layer Output
        """
        return torch.reshape(x, self.shape)


class Transpose(nn.Module):
    """Custom transpose module for easier Sequential usage."""
    def __init__(self, dim0, dim1):
        """Initialize Transpose Module.

        Args:
            dim0 (int): Dimension 1 of Torch.Transpose
            dim1 (int): Dimension 2 of Torch.Transpose
        """
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        """Apply Transpose Module to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return torch.transpose(x, self.dim0, self.dim1)


class MLP(torch.nn.Module):
    """Two layered perceptron."""
    
    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, output_each_layer=False):
        """Initialize two-layered perceptron.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden layer dimension
            outdim (int): Output layer dimension
            dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            output_each_layer (bool, optional): Whether to return outputs of each layer as a list. Defaults to False.
        """
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Apply MLP to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        if self.output_each_layer:
            return [0, x, output, self.lklu(output2)]
        return output2



class GRU(torch.nn.Module):
    """Implements Gated Recurrent Unit (GRU)."""
    
    def __init__(self, indim, hiddim, dropout=False, dropoutp=0.1, flatten=False, has_padding=False, last_only=False,batch_first=True):
        """Initialize GRU Module.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden dimension
            dropout (bool, optional): Whether to apply dropout layer or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            flatten (bool, optional): Whether to flatten output before returning. Defaults to False.
            has_padding (bool, optional): Whether the input has padding or not. Defaults to False.
            last_only (bool, optional): Whether to return only the last output of the GRU. Defaults to False.
            batch_first (bool, optional): Whether to batch before applying or not. Defaults to True.
        """
        super(GRU, self).__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=True)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.last_only = last_only
        self.batch_first = batch_first

    def forward(self, x):
        """Apply GRU to input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=self.batch_first, enforce_sorted=False)
            out = self.gru(x)[1][-1]
        elif self.last_only:
            out = self.gru(x)[1][0]
            
            
            return out
        else:
            out, l = self.gru(x)
        if self.dropout:
            out = self.dropout_layer(out)
        if self.flatten:
            out = torch.flatten(out, 1)
        
        return out


class GRUWithLinear(torch.nn.Module):
    """Implements a GRU with Linear Post-Processing."""
    
    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, flatten=False, has_padding=False, output_each_layer=False, batch_first=False):
        """Initialize GRUWithLinear Module.

        Args:
            indim (int): Input Dimension
            hiddim (int): Hidden Dimension
            outdim (int): Output Dimension
            dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            flatten (bool, optional): Whether to flatten output before returning. Defaults to False.
            has_padding (bool, optional): Whether input has padding. Defaults to False.
            output_each_layer (bool, optional): Whether to return the output of every intermediate layer. Defaults to False.
            batch_first (bool, optional): Whether to apply batching before GRU. Defaults to False.
        """
        super(GRUWithLinear, self).__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=batch_first)
        self.linear = nn.Linear(hiddim, outdim)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Apply GRUWithLinear to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=True, enforce_sorted=False)
            hidden = self.gru(x)[1][-1]
        else:
            hidden = self.gru(x)[0]
        if self.dropout:
            hidden = self.dropout_layer(hidden)
        out = self.linear(hidden)
        if self.flatten:
            out = torch.flatten(out, 1)
        if self.output_each_layer:
            return [0, torch.flatten(x, 1), torch.flatten(hidden, 1), self.lklu(out)]
        return out



class LSTM(torch.nn.Module):
    """Extends nn.LSTM with dropout and other features."""
    
    def __init__(self, indim, hiddim, linear_layer_outdim=None, dropout=False, dropoutp=0.1, flatten=False, has_padding=False):
        """Initialize LSTM Object.

        Args:
            indim (int): Input Dimension
            hiddim (int): Hidden Layer Dimension
            linear_layer_outdim (int, optional): Linear Layer Output Dimension. Defaults to None.
            dropout (bool, optional): Whether to apply dropout to layer output. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            flatten (bool, optional): Whether to flatten out. Defaults to False.
            has_padding (bool, optional): Whether input has padding. Defaults to False.
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(indim, hiddim, batch_first=True)
        if linear_layer_outdim is not None:
            self.linear = nn.Linear(hiddim, linear_layer_outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.flatten = flatten
        self.has_padding = has_padding
        self.linear_layer_outdim = linear_layer_outdim

    def forward(self, x):
        """Apply LSTM to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=True, enforce_sorted=False)
            out = self.lstm(x)[1][0]
        else:
            if len(x.size()) == 2:
                x = x.unsqueeze(2)
            out = self.lstm(x)[1][0]
        out = out.permute([1, 2, 0])
        out = out.reshape([out.size()[0], -1])
        if self.dropout:
            out = self.dropout_layer(out)
        if self.flatten:
            out = torch.flatten(out, 1)
        if self.linear_layer_outdim is not None:
            out = self.linear(out)
        return out




class TwoLayersLSTM(torch.nn.Module):
    """Implements and Extends nn.LSTM for 2-layer LSTMs."""
    
    def __init__(self, indim, hiddim, dropout=False, dropoutp=0.1, flatten=False, has_padding=False,
                 LayNorm=True, isBidirectional=True):
        """Initialize TwoLayersLSTM Object.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden layer dimension
            dropout (bool, optional): Whether to apply dropout to layer output. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            flatten (bool, optional): Whether to flatten layer output before returning. Defaults to False.
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
            isBidirectional (bool, optional): Whether internal LSTMs are bidirectional. Defaults to True.
        """
        super(TwoLayersLSTM, self).__init__()
        self.lstm_0 = nn.LSTM(indim, hiddim, batch_first=True,
                              bidirectional=isBidirectional)
        self.lstm_1 = nn.LSTM(
            2*indim, hiddim, batch_first=True, bidirectional=isBidirectional)
        self.layer_norm = nn.LayerNorm(2*hiddim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.flatten = flatten
        self.has_padding = has_padding
        #self.LayerNorm = LayNorm

    def forward(self, x):
        """Apply TwoLayersLSTM to input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=True, enforce_sorted=False)
            out = self.lstm(x)[1][-1]

            packed_sequence = pack_padded_sequence(x[0], x[1])
            packed_h1, (_, _) = self.lstm_0(packed_sequence)
            padded_h1, _ = pad_packed_sequence(packed_h1)
            normed_h1 = self.layer_norm(padded_h1)
            packed_normed_h1 = pack_padded_sequence(normed_h1, x[1])
            _, (out, _) = self.lstm_1(packed_normed_h1)
        else:
            out = self.lstm_0(x)[0]
            out = self.lstm_1(out)[0]
        if self.dropout:
            out = self.dropout_layer(out)
        if self.flatten:
            out = torch.flatten(out, 1)
        return out



class LeNet(nn.Module):
    """Implements LeNet.
    
    Adapted from centralnet code https://github.com/slyviacassell/_MFAS/blob/master/models/central/avmnist.py.
    """
    
    def __init__(self, in_channels, args_channels, additional_layers, output_each_layer=False, linear=None, squeeze_output=True):
        """Initialize LeNet.

        Args:
            in_channels (int): Input channel number.
            args_channels (int): Output channel number for block.
            additional_layers (int): Number of additional blocks for LeNet.
            output_each_layer (bool, optional): Whether to return the output of all layers. Defaults to False.
            linear (tuple, optional): Tuple of (input_dim, output_dim) for optional linear layer post-processing. Defaults to None.
            squeeze_output (bool, optional): Whether to squeeze output before returning. Defaults to True.
        """
        super(LeNet, self).__init__()
        self.output_each_layer = output_each_layer
        self.convs = [
            nn.Conv2d(in_channels, args_channels, kernel_size=5, padding=2, bias=False)]
        self.bns = [nn.BatchNorm2d(args_channels)]
        self.gps = [GlobalPooling2D()]
        for i in range(additional_layers):
            self.convs.append(nn.Conv2d((2**i)*args_channels, (2**(i+1))
                              * args_channels, kernel_size=3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(args_channels*(2**(i+1))))
            self.gps.append(GlobalPooling2D())
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        self.gps = nn.ModuleList(self.gps)
        self.sq_out = squeeze_output
        self.linear = None
        if linear is not None:
            self.linear = nn.Linear(linear[0], linear[1])
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        """Apply LeNet to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        tempouts = []
        out = x
        for i in range(len(self.convs)):
            out = F.relu(self.bns[i](self.convs[i](out)))
            out = F.max_pool2d(out, 2)
            gp = self.gps[i](out)
            tempouts.append(gp)
            
        if self.linear is not None:
            out = self.linear(out)
        tempouts.append(out)
        if self.output_each_layer:
            if self.sq_out:
                return [t.squeeze() for t in tempouts]
            return tempouts
        if self.sq_out:
            return out.squeeze()
        return out


class VGG16(nn.Module):
    """Extends VGG16 for encoding."""
    
    def __init__(self, hiddim, pretrained=True):
        """Initialize VGG16 Object.

        Args:
            hiddim (int): Size of post-processing layer
            pretrained (bool, optional): Whether to instantiate VGG16 from pretrained. Defaults to True.
        """
        super(VGG16, self).__init__()
        self.hiddim = hiddim
        self.model = tmodels.vgg16_bn(pretrained=pretrained)
        self.model.classifier[6] = nn.Linear(4096, hiddim)

    def forward(self, x):
        """Apply VGG16 to Input.

        Args:
            x (torch.Tensor): Layer Input 

        Returns:
            torch.Tensor: Layer Output
        """
        return self.model(x)


class VGG16Slim(nn.Module):  
    """Extends VGG16 with a fewer layers in the classifier.
    
    Slimmer version of vgg16 model with fewer layers in classifier.
    """
    
    def __init__(self, hiddim, dropout=True, dropoutp=0.2, pretrained=True):
        """Initialize VGG16Slim object.

        Args:
            hiddim (int): Hidden dimension size
            dropout (bool, optional): Whether to apply dropout to ReLU output. Defaults to True.
            dropoutp (float, optional): Dropout probability. Defaults to 0.2.
            pretrained (bool, optional): Whether to initialize VGG16 from pretrained. Defaults to True.
        """
        super(VGG16Slim, self).__init__()
        self.hiddim = hiddim
        self.model = tmodels.vgg16_bn(pretrained=pretrained)
        self.model.classifier = nn.Linear(512 * 7 * 7, hiddim)
        if dropout:
            feats_list = list(self.model.features)
            new_feats_list = []
            for feat in feats_list:
                new_feats_list.append(feat)
                if isinstance(feat, nn.ReLU):
                    new_feats_list.append(nn.Dropout(p=dropoutp))

            self.model.features = nn.Sequential(*new_feats_list)

    def forward(self, x):
        """Apply VGG16Slim to model input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return self.model(x)


class VGG11Slim(nn.Module): 
    """Extends VGG11 with a fewer layers in the classifier.
    
    Slimmer version of vgg11 model with fewer layers in classifier.
    """
    
    def __init__(self, hiddim, dropout=True, dropoutp=0.2, pretrained=True, freeze_features=True):
        """Initialize VGG11Slim Object.

        Args:
            hiddim (int): Hidden dimension size
            dropout (bool, optional): Whether to apply dropout to output of ReLU. Defaults to True.
            dropoutp (float, optional): Dropout probability. Defaults to 0.2.
            pretrained (bool, optional): Whether to instantiate VGG11 from Pretrained. Defaults to True.
            freeze_features (bool, optional): Whether to keep VGG11 features frozen. Defaults to True.
        """
        super(VGG11Slim, self).__init__()
        self.hiddim = hiddim
        self.model = tmodels.vgg11_bn(pretrained=pretrained)
        self.model.classifier = nn.Linear(512 * 7 * 7, hiddim)
        if dropout:
            feats_list = list(self.model.features)
            new_feats_list = []
            for feat in feats_list:
                new_feats_list.append(feat)
                if isinstance(feat, nn.ReLU):
                    new_feats_list.append(nn.Dropout(p=dropoutp))

            self.model.features = nn.Sequential(*new_feats_list)
        for p in self.model.features.parameters():
            p.requires_grad = (not freeze_features)

    def forward(self, x):
        """Apply VGG11Slim to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return self.model(x)


class VGG11Pruned(nn.Module):
    """Extends VGG11 and prunes layers to make it even smaller.
    
    Slimmer version of vgg11 model with fewer layers in classifier.
    """
    
    def __init__(self, hiddim, dropout=True, prune_factor=0.25, dropoutp=0.2):
        """Initialize VGG11Pruned Object.

        Args:
            hiddim (int): Hidden Layer Dimension
            dropout (bool, optional): Whether to apply dropout after ReLU. Defaults to True.
            prune_factor (float, optional): Percentage of channels to prune. Defaults to 0.25.
            dropoutp (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(VGG11Pruned, self).__init__()
        self.hiddim = hiddim
        self.model = tmodels.vgg11_bn(pretrained=False)
        self.model.classifier = nn.Linear(
            int(512 * prune_factor) * 7 * 7, hiddim)
        if dropout:
            feats_list = list(self.model.features)
            new_feats_list = []
            for feat in feats_list:
                if isinstance(feat, nn.Conv2d):
                    pruned_feat = nn.Conv2d(int(feat.in_channels * prune_factor) if feat.in_channels != 3 else 3,
                                            int(feat.out_channels * prune_factor),
                                            kernel_size=feat.kernel_size,
                                            padding=feat.padding)
                    new_feats_list.append(pruned_feat)
                elif isinstance(feat, nn.BatchNorm2d):
                    pruned_feat = nn.BatchNorm2d(
                        int(feat.num_features * prune_factor))
                    new_feats_list.append(pruned_feat)
                else:
                    new_feats_list.append(feat)
                if isinstance(feat, nn.ReLU):
                    new_feats_list.append(nn.Dropout(p=dropoutp))

            self.model.features = nn.Sequential(*new_feats_list)

    def forward(self, x):
        """Apply VGG11Pruned to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return self.model(x)



class VGG16Pruned(nn.Module):
    """Extends VGG16 and prunes layers to make it even smaller.
    
    Slimmer version of vgg16 model with fewer layers in classifier.
    """
    
    def __init__(self, hiddim, dropout=True, prune_factor=0.25, dropoutp=0.2):
        """Initialize VGG16Pruned Object.

        Args:
            hiddim (int): Hidden Layer Dimension
            dropout (bool, optional): Whether to apply dropout after ReLU. Defaults to True.
            prune_factor (float, optional): Percentage of channels to prune. Defaults to 0.25.
            dropoutp (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(VGG16Pruned, self).__init__()
        self.hiddim = hiddim
        self.model = tmodels.vgg16_bn(pretrained=False)
        self.model.classifier = nn.Linear(
            int(512 * prune_factor) * 7 * 7, hiddim)
        if dropout:
            feats_list = list(self.model.features)
            new_feats_list = []
            for feat in feats_list:
                if isinstance(feat, nn.Conv2d):
                    pruned_feat = nn.Conv2d(int(feat.in_channels * prune_factor) if feat.in_channels != 3 else 3,
                                            int(feat.out_channels * prune_factor),
                                            kernel_size=feat.kernel_size,
                                            padding=feat.padding)
                    new_feats_list.append(pruned_feat)
                elif isinstance(feat, nn.BatchNorm2d):
                    pruned_feat = nn.BatchNorm2d(
                        int(feat.num_features * prune_factor))
                    new_feats_list.append(pruned_feat)
                else:
                    new_feats_list.append(feat)
                if isinstance(feat, nn.ReLU):
                    new_feats_list.append(nn.Dropout(p=dropoutp))

            self.model.features = nn.Sequential(*new_feats_list)

    def forward(self, x):
        """Apply VGG16Pruned to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return self.model(x)


class VGG(nn.Module):
    """Extends tmodels.vgg19 module with Global Pooling, BatchNorm, and a Linear Output."""
    
    def __init__(self, num_outputs):
        """Initialize VGG Object.

        Args:
            num_outputs (int): Output Dimension
        """
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
        """Apply VGG Module to Input.

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor
        """
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

        
        

        return out_1, out_2, out_3, out_4, out


class Maxout(nn.Module):
    """Implements Maxout module."""
    
    def __init__(self, d, m, k):
        """Initialize Maxout object.

        Args:
            d (int): (Unused)
            m (int): Number of features remeaining after Maxout.
            k (int): Pool Size
        """
        super(Maxout, self).__init__()
        self.d_in, self.d_out, self.pool_size = d, m, k
        self.lin = nn.Linear(d, m * k)

    def forward(self, inputs):
        """Apply Maxout to inputs.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, _ = out.view(*shape).max(dim=max_dim)
        return m


class MaxOut_MLP(nn.Module):
    """Implements Maxout w/ MLP."""
    
    def __init__(
            self, num_outputs, first_hidden=64, number_input_feats=300, second_hidden=None, linear_layer=True):
        """Instantiate MaxOut_MLP Module.

        Args:
            num_outputs (int): Output dimension
            first_hidden (int, optional): First hidden layer dimension. Defaults to 64.
            number_input_feats (int, optional): Input dimension. Defaults to 300.
            second_hidden (_type_, optional): Second hidden layer dimension. Defaults to None.
            linear_layer (bool, optional): Whether to include an output hidden layer or not. Defaults to True.
        """
        super(MaxOut_MLP, self).__init__()

        if second_hidden is None:
            second_hidden = first_hidden
        self.op0 = nn.BatchNorm1d(number_input_feats, 1e-4)
        self.op1 = Maxout(number_input_feats, first_hidden, 2)
        self.op2 = nn.Sequential(nn.BatchNorm1d(first_hidden), nn.Dropout(0.3))
        #self.op2 = nn.BatchNorm1d(first_hidden)
        #self.op3 = Maxout(first_hidden, first_hidden * 2, 5)
        self.op3 = Maxout(first_hidden, second_hidden, 2)
        self.op4 = nn.Sequential(nn.BatchNorm1d(
            second_hidden), nn.Dropout(0.3))
        #self.op4 = nn.BatchNorm1d(second_hidden)

        # The linear layer that maps from hidden state space to output space
        if linear_layer:
            self.hid2val = nn.Linear(second_hidden, num_outputs)
        else:
            self.hid2val = None

    def forward(self, x):
        """Apply module to layer input

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        o0 = self.op0(x)
        o1 = self.op1(o0)
        o2 = self.op2(o1)
        o3 = self.op3(o2)
        o4 = self.op4(o3)
        if self.hid2val is None:
            return o4
        o5 = self.hid2val(o4)

        return o5


class GlobalPooling2D(nn.Module):
    """Implements 2D Global Pooling."""
    
    def __init__(self):
        """Initializes GlobalPooling2D Module."""
        super(GlobalPooling2D, self).__init__()

    def forward(self, x):
        """Apply 2D Global Pooling to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        # apply global average pooling
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        x = x.view(x.size(0), -1)

        return x


class Constant(nn.Module):
    """Implements a module that returns a constant no matter the input."""
    
    def __init__(self, out_dim):
        """Initialize Constant Module.

        Args:
            out_dim (int): Output Dimension.
        """
        super(Constant, self).__init__()
        self.out_dim = out_dim

    def forward(self, x):
        """Apply Constant to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return torch.zeros(self.out_dim).to(x.device)


class Identity(nn.Module):
    """Identity Module."""
    
    def __init__(self):
        """Initialize Identity Module."""
        super().__init__()

    def forward(self, x):
        """Apply Identity to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return x



class DAN(torch.nn.Module):
    """
    Deep Averaging Network: https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf
    Deep Sets: https://arxiv.org/abs/1703.06114
    """
    def __init__(self, indim, hiddim, dropout=False, dropoutp=0.25, nlayers=3, has_padding=False):
        """Initialize DAN Object.

        Args:
            indim (int): Input Dimension
            hiddim (int): Hidden Dimension
            dropout (bool, optional): Whether to apply dropout to layer output. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.25.
            nlayers (int, optional): Number of layers. Defaults to 3.
            has_padding (bool, optional): Whether the input has padding. Defaults to False.
        """
        super(DAN, self).__init__()
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.nlayers = nlayers
        self.has_padding = has_padding

        self.embedding = nn.Linear(indim, hiddim)

        mlp = []
        for _ in range(nlayers):
            mlp.append(nn.Linear(hiddim, hiddim))
        self.mlp = nn.ModuleList(mlp)

    def forward(self, x):
        """Apply DAN to input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        # x_vals: B x S x P
        if self.has_padding:
            x_vals = x[0]
            x_lens = x[1]
        else:
            x_vals = x
        # embedded: B x S x H
        embedded = self.embedding(x_vals)
        if self.dropout:
            embedded = self.dropout_layer(embedded)
        if self.has_padding:
            # mask out padded values
            # mask: B x S
            mask = torch.arange(embedded.shape[1], device=embedded.device).repeat(
                embedded.shape[0], 1) < x_lens.repeat(-1, 1).repeat(1, embedded.shape[1])
            embedded[~mask] = 0
        # sum pooling
        # pool: B x H
        pooled = embedded.sum(dim=1)
        for layer in self.mlp:
            pooled = layer(pooled)
            if self.dropout:
                pooled = self.dropout_layer(pooled)
        return pooled


class ResNetLSTMEnc(torch.nn.Module):
    """Implements an encoder which applies as ResNet first, and then an LSTM."""
    
    def __init__(self, hiddim, dropout=False, dropoutp=0.1):
        """Instantiates ResNetLSTMEnc Module

        Args:
            hiddim (int): Hidden dimension size of LSTM.
            dropout (bool, optional): Whether to apply dropout or not.. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(ResNetLSTMEnc, self).__init__()
        self.enc = torchvision.models.resnet18(pretrained=True)
        self.lstm = nn.LSTM(1000, hiddim, batch_first=True)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout

    def forward(self, x):  # x is (cbatch_size, 3, 150, 112, 112)
        """Apply ResNetLSTMEnc Module to Input

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        cbatch_size = x.shape[0]
        x = x.permute([0, 2, 1, 3, 4])  # (cbatch_size, 150, 3, 112, 112)
        x = x.reshape(-1, 3, 112, 112)  # (cbatch_size*150, 3, 112, 112)
        x = self.enc(x)  # (cbatch_size*150, 1000)
        x = x.reshape(cbatch_size, -1, 1000)
        hidden = self.lstm(x)[1][0]
        hidden = hidden.permute([1, 2, 0])
        hidden = hidden.reshape([hidden.size()[0], -1])
        if self.dropout:
            hidden = self.dropout_layer(hidden)
        return hidden


class Transformer(nn.Module):
    """Extends nn.Transformer."""
    
    def __init__(self, n_features, dim):
        """Initialize Transformer object.

        Args:
            n_features (int): Number of features in the input.
            dim (int): Dimension which to embed upon / Hidden dimension size.
        """
        super().__init__()
        self.embed_dim = dim
        self.conv = nn.Conv1d(n_features, self.embed_dim,
                              kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=5)
        self.transformer = nn.TransformerEncoder(layer, num_layers=5)

    def forward(self, x):
        """Apply Transformer to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if type(x) is list:
            x = x[0]
        x = self.conv(x.permute([0, 2, 1]))
        x = x.permute([2, 0, 1])
        x = self.transformer(x)[-1]
        return x
