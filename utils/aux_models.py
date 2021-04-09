#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanma
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# %%

class Identity(nn.Module):
    def forward(self, inputs):
        return inputs


class Tensor1DLateralPadding(nn.Module):

    def __init__(self, pad):
        super(Tensor1DLateralPadding, self).__init__()
        self.pad = pad

    def forward(self, inputs):
        sz = inputs.size()
        padding = torch.autograd.Variable(torch.zeros(sz[0], self.pad), requires_grad=False)
        if inputs.is_cuda:
            padding = padding.cuda()

        padded = torch.cat((inputs, padding), 1)
        return padded


class ChannelPadding(nn.Module):

    def __init__(self, pad):
        super(ChannelPadding, self).__init__()
        self.pad = pad

    def forward(self, inputs):
        sz = inputs.size()
        padding = torch.autograd.Variable(torch.zeros(sz[0], self.pad, sz[2], sz[3]), requires_grad=False)
        if inputs.is_cuda:
            padding = padding.cuda()

        padded = torch.cat((inputs, padding), 1)
        return padded


class GlobalPooling2D(nn.Module):
    def __init__(self):
        super(GlobalPooling2D, self).__init__()

    def forward(self, x):
        # apply global average pooling
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        x = x.view(x.size(0), -1)

        return x


class GlobalPooling1D(nn.Module):
    def __init__(self):
        super(GlobalPooling1D, self).__init__()

    def forward(self, x):
        # apply global average pooling
        x = torch.mean(x, 2)

        return x


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


class AlphaScalarMultiplication(nn.Module):
    def __init__(self, size_alpha_x, size_alpha_y):
        super(AlphaScalarMultiplication, self).__init__()
        self.size_alpha_x = size_alpha_x
        self.size_alpha_y = size_alpha_y

        # self.alpha_x = torch.tensor([float(1)], requires_grad=True)
        self.alpha_x = nn.Parameter(torch.from_numpy(np.zeros((1), np.float32)))

    def forward(self, x, y):
        bsz = x.size()[0]
        factorx = torch.sigmoid(self.alpha_x.expand(bsz, self.size_alpha_x))
        factory = 1.0 - torch.sigmoid(self.alpha_x.expand(bsz, self.size_alpha_y))

        x = x * factorx
        y = y * factory

        return x, y


class AlphaVectorMultiplication(nn.Module):
    def __init__(self, size_alpha):
        super(AlphaVectorMultiplication, self).__init__()
        self.size_alpha = size_alpha

        self.alpha = nn.Parameter(torch.from_numpy(np.zeros((1, size_alpha), np.float32)))

    def forward(self, x):
        bsz = x.size()[0]
        x = x * torch.sigmoid(self.alpha.expand(bsz, -1))

        return x


# %%
class WeightedCrossEntropyWithLogits(nn.Module):

    def __init__(self, pos_weight):
        super(WeightedCrossEntropyWithLogits, self).__init__()
        self.w = pos_weight

    def forward(self, logits, targets):
        q = [self.w] * logits.size()[0]
        q = torch.from_numpy(np.asarray(q, np.float32)).to(logits.device)

        x = torch.sigmoid(logits)
        z = targets

        L = q * z * -torch.log(x) + (1 - z) * -torch.log(1 - x)
        # l = (1 + (q - 1) * z)
        # L = (1 - z) * x + l * (torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(-x, 0)[0])

        totloss = torch.mean(torch.mean(L))
        return totloss

    # %%


class CellBlock(nn.Module):

    def __init__(self, op1_type, op2_type, args):
        super(CellBlock, self).__init__()

        self.args = args
        self.op1 = CreateOp(op1_type, args.planes, args.planes)
        self.op2 = CreateOp(op2_type, args.planes, args.planes)

        self.op1_type = op1_type
        self.op2_type = op2_type

        self.dp1 = DropPath(1.0 - self.args.drop_path)
        self.dp2 = DropPath(1.0 - self.args.drop_path)

    def forward(self, x1, x2):
        xa, xa_dropped = self.dp1(self.op1(x1))

        xb, xb_dropped = self.dp2(self.op2(x2), xa_dropped)

        # print('dropped: {}, {}'.format(xa_dropped, xb_dropped))

        return xa + xb


# %%

class Cell(nn.Module):

    def __init__(self, operation_labels, configuration_indexes, connections, args):
        super(Cell, self).__init__()

        self._args = args
        self._configuration = configuration_indexes
        self._connections = connections
        self._operation_labels = operation_labels
        self._planes = args.planes

        self.blocks, self.block_used = self._create_blocks()
        self.num_concatenations = len([bu for bu in self.block_used if not bu])

        self.bn = nn.BatchNorm2d(self._planes, eps=1e-3)

    def forward(self, x1, x2):

        block_outputs = list([x1, x2])

        # apply blocks according to the connections
        for block_index, block_connection in enumerate(self._connections):
            conn = self._conn(block_connection)
            block_outputs.append(self.blocks[block_index](block_outputs[conn[0]], block_outputs[conn[1]]))

        # check which blocks were not used and concatenate the outputs (first two outputs are not blocks, hence the :2)
        output = [block_output for b_i, block_output in enumerate(block_outputs[2:]) if not self.block_used[b_i]]

        # sum during search for some reason. for fixedcell they are concated
        output = sum(output)
        output = self.bn(output)

        return output

    def _conn(self, conn):
        return conn + 2

    def _create_blocks(self):

        block_array = nn.ModuleList()
        block_used = len(self._connections) * [False]

        for b_i, block_conf in enumerate(self._configuration):
            op1_type = self._operation_labels[block_conf[0]]
            op2_type = self._operation_labels[block_conf[1]]
            block_array.append(CellBlock(op1_type, op2_type, self._args))

            block_connection = self._connections[b_i]
            if block_connection[0] >= 0:
                block_used[block_connection[0]] = True
            if block_connection[1] >= 0:
                block_used[block_connection[1]] = True

        return block_array, block_used


class FixedCell(nn.Module):

    def __init__(self, operation_labels, configuration_indexes, connections, args):
        super(FixedCell, self).__init__()

        self._args = args
        self._configuration = configuration_indexes
        self._connections = connections
        self._operation_labels = operation_labels
        self._planes = args.planes

        self.blocks, self.block_used = self._create_blocks()
        self.num_concatenations = len([bu for bu in self.block_used if not bu])

        in_planes = self.num_concatenations * self._args.planes
        self.dim_reduc = nn.Sequential(
            nn.Conv2d(in_planes, self._args.planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(self._args.planes))

    def forward(self, x1, x2):

        block_outputs = list([x1, x2])

        # apply blocks according to the connections
        for block_index, block_connection in enumerate(self._connections):
            conn = self._conn(block_connection)
            block_outputs.append(self.blocks[block_index](block_outputs[conn[0]], block_outputs[conn[1]]))

        # check which blocks were not used and concatenate the outputs (first two outputs are not blocks, hence the :2)
        output = [block_output for b_i, block_output in enumerate(block_outputs[2:]) if not self.block_used[b_i]]

        # ToDO: use this only for final network
        # if output:
        output = torch.cat(output, dim=1)  # concatenate all selected outputs depthwise
        # else:
        #    raise TypeError("Something went wrong. No outputs!")
        output = self.dim_reduc(output)

        return output

    def _conn(self, conn):
        return conn + 2

    def _create_blocks(self):

        block_array = nn.ModuleList()
        block_used = len(self._connections) * [False]

        for b_i, block_conf in enumerate(self._configuration):
            op1_type = self._operation_labels[block_conf[0]]
            op2_type = self._operation_labels[block_conf[1]]
            block_array.append(CellBlock(op1_type, op2_type, self._args))

            block_connection = self._connections[b_i]
            if block_connection[0] >= 0:
                block_used[block_connection[0]] = True
            if block_connection[1] >= 0:
                block_used[block_connection[1]] = True

        return block_array, block_used

    # %%


class FactorizedReduction(nn.Module):
    '''
    Reduce both spatial dimensions (width and height) by a factor of 2, and 
    potentially to change the number of output filters
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L129
    '''

    def __init__(self, in_planes, out_planes, stride=2):
        super(FactorizedReduction, self).__init__()

        assert out_planes % 2 == 0, (
            "Need even number of filters when using this factorized reduction.")

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        if stride == 1:
            self.fr = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes))
        else:
            self.path1 = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.Conv2d(in_planes, out_planes // 2, kernel_size=1, bias=False))

            self.path2 = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.Conv2d(in_planes, out_planes // 2, kernel_size=1, bias=False))
            self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        if self.stride == 1:
            return self.fr(x)
        else:
            path1 = self.path1(x)

            # pad the right and the bottom, then crop to include those pixels
            path2 = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0.)
            path2 = path2[:, :, 1:, 1:]
            path2 = self.path2(path2)

            out = torch.cat([path1, path2], dim=1)
            out = self.bn(out)
            return out


# %%

class PoolBranch(nn.Module):
    """
    Max pooling operations with 1x1 convolutions to fix output size
    """

    def __init__(self, in_planes, out_planes, avg_or_max):
        super(PoolBranch, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.avg_or_max = avg_or_max

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())

        if avg_or_max == 'avg':
            self.pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        elif avg_or_max == 'max':
            self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError("Unknown pool {}".format(avg_or_max))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        return out


# %%

class ConvBranch(nn.Module):
    '''
    Conv branch
    '''

    def __init__(self, in_planes, out_planes, kernel_size, separable=False):
        super(ConvBranch, self).__init__()
        assert kernel_size in [3, 5, 7], "Kernel size must be either 3, 5 or 7"

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.separable = separable

        self.inp_conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())

        if separable:
            self.out_conv = nn.Sequential(
                SeparableConvOld(out_planes, out_planes, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU())
        else:
            padding = (kernel_size - 1) // 2
            self.out_conv = nn.Sequential(
                nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size,
                          padding=padding, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU())

    def forward(self, x):
        out = self.inp_conv1(x)
        out = self.out_conv(out)
        return out


# %%

class SeparableConvOld(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, bias=False):
        super(SeparableConvOld, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size,
                                   padding=padding, groups=in_planes, bias=bias)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# %%

class SeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, bias=False):
        super(SeparableConv, self).__init__()
        padding = (kernel_size - 1) // 2

        self.op = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size,
                      padding=padding, groups=in_planes, bias=bias),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_planes, eps=1e-3),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=1,
                      padding=padding, groups=out_planes, bias=bias),
            nn.Conv2d(out_planes, out_planes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes, eps=1e-3),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.op(x)
        return out


# %%

class IdentityModule(nn.Module):
    def forward(self, inputs):
        return inputs


# %%

def CreateOp(conv_type, input_planes=64, output_planes=64):
    if conv_type == 0 or conv_type == 'I':
        inp_conv = nn.Sequential(
            nn.Conv2d(input_planes, output_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_planes),
            nn.ReLU())
        op = nn.Sequential(inp_conv, IdentityModule())
    elif conv_type == 1 or conv_type == '1x1 conv':
        op = ConvBranch(input_planes, output_planes, kernel_size=1, separable=False)
    elif conv_type == 2 or conv_type == '3x3 conv':
        op = ConvBranch(input_planes, output_planes, kernel_size=3, separable=False)
    elif conv_type == 3 or conv_type == '5x5 conv':
        op = ConvBranch(input_planes, output_planes, kernel_size=5, separable=False)
    elif conv_type == 4 or conv_type == '7x7 conv':
        op = ConvBranch(input_planes, output_planes, kernel_size=7, separable=False)
    elif conv_type == 5 or conv_type == '3x3 depthconv':
        op = ConvBranch(input_planes, output_planes, kernel_size=3, separable=True)
    elif conv_type == 6 or conv_type == '5x5 depthconv':
        op = ConvBranch(input_planes, output_planes, kernel_size=5, separable=True)
    elif conv_type == 7 or conv_type == '7x7 depthconv':
        op = ConvBranch(input_planes, output_planes, kernel_size=7, separable=True)
    elif conv_type == 8 or conv_type == '3x3 maxpool':
        op = PoolBranch(input_planes, output_planes, 'max')
    elif conv_type == 9 or conv_type == '3x3 avgpool':
        op = PoolBranch(input_planes, output_planes, 'avg')
    else:
        raise NotImplementedError(conv_type)

    return op


# %%
class AuxiliaryHead(nn.Module):

    def __init__(self, num_classes, filters=96):
        super(AuxiliaryHead, self).__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(filters, filters * 2, 1, bias=False),
            nn.BatchNorm2d(filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters * 2, filters * 6, 2, bias=False),
            nn.BatchNorm2d(filters * 6),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(filters * 6, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(self.global_avg_pool(x).view(x.size(0), -1))
        return x


# %%
from random import random


class DropPath(nn.Module):
    def __init__(self, keep_prob=0.9):
        super(DropPath, self).__init__()
        self.keep_prob = keep_prob

    def forward(self, x, other_dropped=False):
        if self.training:
            p = random()
            if p <= self.keep_prob or other_dropped:
                return x / (self.keep_prob), False  # Inverted scaling
            else:
                return torch.zeros_like(x, requires_grad=False), True
        else:
            return x, False
