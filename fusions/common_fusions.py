import torch
from torch import nn
from torch.nn import functional as F
import pdb
from torch.autograd import Variable



class Concat(nn.Module):
    """
    Concatenation of input data on dimension 1.
    """
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of Concat.
        
        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)



class ConcatEarly(nn.Module):
    """
    Concatenation of input data on dimension 2.
    """
    def __init__(self):
        super(ConcatEarly, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of ConcatEarly.
        
        :param modalities: An iterable of modalities to combine
        """
        return torch.cat(modalities, dim=2)


# Stacking modalities
class Stack(nn.Module):
    """
    Stacking modalities together on dimension 1.
    """
    def __init__(self):
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of Stack.
        
        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.stack(flattened, dim=2)


class ConcatWithLinear(nn.Module):
    """
    Concatenation with a linear layer.
    """
    def __init__(self, input_dim, output_dim, concat_dim=1):
        """
        Initialize ConcatWithLinear Module
        
        :param input_dim: The input dimension for the linear layer
        :param output_dim: The output dimension for the linear layer
        :concat_dim: The concatentation dimension for the modalities.
        """
        super(ConcatWithLinear, self).__init__()
        self.concat_dim = concat_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, modalities):
        """
        Forward Pass of Stack.
        
        :param modalities: An iterable of modalities to combine
        """
        return self.fc(torch.cat(modalities, dim=self.concat_dim))


class FiLM(nn.Module):
    """
    Implementation of FiLM - Feature-Wise Affine Transformations of the Input.
    
    See https://arxiv.org/pdf/1709.07871.pdf for more details.
    """
    def __init__(self, gamma_generation_network, beta_generation_network, base_modal=0, gamma_generate_modal=1, beta_generate_modal=1):
        """
        Initialize FiLM layer. 
        
        :param gamma_generation_network: Network which generates gamma_parameters from gamma_generation_modal data.
        :param beta_generation_network: Network which generates beta_parameters from beta_generation_modal data.
        :param base_modal: Modality to apply affine transformation to.
        :param gamma_generate_modal: Modality to generate gamma portion of affine transformation from.
        :param beta_generate_modal: Modality to generate beta portion of affine transformation from.
        """
        super(FiLM, self).__init__()
        self.g_net = gamma_generation_network
        self.b_net = beta_generation_network
        self.base_modal = base_modal
        self.ggen_modal = gamma_generate_modal
        self.bgen_modal = beta_generate_modal

    def forward(self, modalities):
        """
        Forward Pass of FiLM.
        
        :param modalities: An iterable of modalities to combine. 
        """
        gamma = self.g_net(modalities[self.ggen_modal])
        beta = self.b_net(modalities[self.bgen_modal])
        return gamma * modalities[self.base_modal] + beta



class MultiplicativeInteractions3Modal(nn.Module):
    """
    Implementation of 3-Way Modal Multiplicative Interactions
    """
    def __init__(self, input_dims, output_dim, task=None):
        """
        :param input_dims: list or tuple of 3 integers indicating sizes of input
        :param output_dim: size of outputs
        :param task: Set to "affect" when working with social data.
        """
        super(MultiplicativeInteractions3Modal, self).__init__()
        self.a = MultiplicativeInteractions2Modal([input_dims[0], input_dims[1]],
                                                  [input_dims[2], output_dim], 'matrix3D')
        self.b = MultiplicativeInteractions2Modal([input_dims[0], input_dims[1]],
                                                  output_dim, 'matrix')
        self.task = task

    def forward(self, modalities):
        """
        Forward Pass of MultiplicativeInteractions3Modal.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if self.task == 'affect':
            return torch.einsum('bm, bmp -> bp', modalities[2], self.a(modalities[0:2])) + self.b(modalities[0:2])
        return torch.matmul(modalities[2], self.a(modalities[0:2])) + self.b(modalities[0:2])


class MultiplicativeInteractions2Modal(nn.Module):
    """
    Implementation of 2-way Modal Multiplicative Interactions
    """
    def __init__(self, input_dims, output_dim, output, flatten=False, clip=None, grad_clip=None, flip=False):
        """
        :param input_dims: list or tuple of 2 integers indicating input dimensions of the 2 modalities
        :param output_dim: output dimension
        :param output: type of MI, options from 'matrix3D','matrix','vector','scalar'
        :param flatten: whether we need to flatten the input modalities
        :param clip: clip parameter values, None if no clip
        :param grad_clip: clip grad values, None if no clip
        :param flip: whether to swap the two input modalities in forward function or not
        """
        super(MultiplicativeInteractions2Modal, self).__init__()
        self.input_dims = input_dims
        self.clip = clip
        self.output_dim = output_dim
        self.output = output
        self.flatten = flatten
        if output == 'matrix3D':
            self.W = nn.Parameter(torch.Tensor(
                input_dims[0], input_dims[1], output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(
                input_dims[0], output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(
                input_dims[1], output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.b)

        # most general Hypernetworks as Multiplicative Interactions.
        elif output == 'matrix':
            self.W = nn.Parameter(torch.Tensor(
                input_dims[0], input_dims[1], output_dim))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0], output_dim))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(input_dims[1], output_dim))
            nn.init.xavier_normal(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim))
            nn.init.normal_(self.b)
        # Diagonal Forms and Gating Mechanisms.
        elif output == 'vector':
            self.W = nn.Parameter(torch.Tensor(input_dims[0], input_dims[1]))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(
                self.input_dims[0], self.input_dims[1]))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.b)
        # Scales and Biases.
        elif output == 'scalar':
            self.W = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.b)
        self.flip = flip
        if grad_clip is not None:
            for p in self.parameters():
                p.register_hook(lambda grad: torch.clamp(
                    grad, grad_clip[0], grad_clip[1]))

    def repeatHorizontally(self, tensor, dim):
        return tensor.repeat(dim).view(dim, -1).transpose(0, 1)

    def forward(self, modalities):
        """
        Forward Pass of MultiplicativeInteractions2Modal.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if len(modalities) == 1:
            return modalities[0]
        elif len(modalities) > 2:
            assert False
        m1 = modalities[0]
        m2 = modalities[1]
        if self.flip:
            m1 = modalities[1]
            m2 = modalities[0]

        if self.flatten:
            m1 = torch.flatten(m1, start_dim=1)
            m2 = torch.flatten(m2, start_dim=1)
        if self.clip is not None:
            m1 = torch.clip(m1, self.clip[0], self.clip[1])
            m2 = torch.clip(m2, self.clip[0], self.clip[1])

        if self.output == 'matrix3D':
            Wprime = torch.einsum('bn, nmpq -> bmpq', m1,
                                  self.W) + self.V  # bmpq
            bprime = torch.einsum('bn, npq -> bpq', m1,
                                  self.U) + self.b    # bpq
            output = torch.einsum('bm, bmpq -> bpq', m2,
                                  Wprime) + bprime   # bpq

        # Hypernetworks as Multiplicative Interactions.
        elif self.output == 'matrix':
            Wprime = torch.einsum('bn, nmd -> bmd', m1,
                                  self.W) + self.V      # bmd
            bprime = torch.matmul(m1, self.U) + self.b      # bmd
            output = torch.einsum('bm, bmd -> bd', m2,
                                  Wprime) + bprime             # bmd

        # Diagonal Forms and Gating Mechanisms.
        elif self.output == 'vector':
            Wprime = torch.matmul(m1, self.W) + self.V      # bm
            bprime = torch.matmul(m1, self.U) + self.b      # b
            output = Wprime*m2 + bprime             # bm

        # Scales and Biases.
        elif self.output == 'scalar':
            Wprime = torch.matmul(m1, self.W.unsqueeze(1)).squeeze(1) + self.V
            bprime = torch.matmul(m1, self.U.unsqueeze(1)).squeeze(1) + self.b
            output = self.repeatHorizontally(
                Wprime, self.input_dims[1]) * m2 + self.repeatHorizontally(bprime, self.input_dims[1])
        return output


class TensorFusion(nn.Module):
    """
    Implementation of TensorFusion Networks.
    
    See https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py for more and the original code.
    """
    def __init__(self):
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if len(modalities) == 1:
            return modalities[0]

        mod0 = modalities[0]
        nonfeature_size = mod0.shape[:-1]

        m = torch.cat((Variable(torch.ones(
            *nonfeature_size, 1).type(mod0.dtype).to(mod0.device), requires_grad=False), mod0), dim=-1)
        for mod in modalities[1:]:
            mod = torch.cat((Variable(torch.ones(
                *nonfeature_size, 1).type(mod.dtype).to(mod.device), requires_grad=False), mod), dim=-1)
            fused = torch.einsum('...i,...j->...ij', m, mod)
            m = fused.reshape([*nonfeature_size, -1])

        return m


class LowRankTensorFusion(nn.Module):
    """
    Implementation of Low-Rank Tensor Fusion.
    
    See https://github.com/Justin1904/Low-rank-Multimodal-Fusion for more information.
    """

    def __init__(self, input_dims, output_dim, rank, flatten=True):
        """
        :param input_dims: list or tuple of integers indicating input dimensions of the modalities
        :param output_dim: output dimension
        :param rank: a hyperparameter of LRTF. See link above for details
        :param flatten: Boolean to dictate if output should be flattened or not. Default: True
        """
        super(LowRankTensorFusion, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        # low-rank factors
        self.factors = []
        for input_dim in input_dims:
            factor = nn.Parameter(torch.Tensor(
                self.rank, input_dim+1, self.output_dim)).cuda()
            nn.init.xavier_normal(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank)).cuda()
        self.fusion_bias = nn.Parameter(
            torch.Tensor(1, self.output_dim)).cuda()
        # init the fusion weights
        nn.init.xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities):
        """
        Forward Pass of Low-Rank TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        batch_size = modalities[0].shape[0]
        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        fused_tensor = 1
        for (modality, factor) in zip(modalities, self.factors):
            ones = Variable(torch.ones(batch_size, 1).type(
                modality.dtype), requires_grad=False).cuda()
            if self.flatten:
                modality_withones = torch.cat(
                    (ones, torch.flatten(modality, start_dim=1)), dim=1)
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(
            1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output


class NLgate(torch.nn.Module):
    """
    Implementation of NLgate Fusion.
    
    See section F4 of "What makes training MM classification networks hard" for details
    """
    def __init__(self, thw_dim, c_dim, tf_dim, q_linear=None, k_linear=None, v_linear=None):
        """
        q_linear, k_linear, v_linear are none if no linear layer applied before q,k,v.
        Otherwise, a tuple of (indim,outdim) is required for each of these 3 arguments.
        
        :param thw_dim: TODO
        :param c_dim: TODO 
        :param tf_dim: TODO
        :param q_linear: TODO
        :param k_linear: TODO
        :param v_linear: TODO
        """
        super(NLgate, self).__init__()
        self.qli = None
        if q_linear is not None:
            self.qli = nn.Linear(q_linear[0], q_linear[1])
        self.kli = None
        if k_linear is not None:
            self.kli = nn.Linear(k_linear[0], k_linear[1])
        self.vli = None
        if v_linear is not None:
            self.vli = nn.Linear(v_linear[0], v_linear[1])
        self.thw_dim = thw_dim
        self.c_dim = c_dim
        self.tf_dim = tf_dim
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """
        Forward Pass of Low-Rank TensorFusion.
        
        :param x: An iterable of modalities to combine. 
        """
        q = x[0]
        k = x[1]
        v = x[1]
        if self.qli is None:
            qin = q.view(-1, self.thw_dim, self.c_dim)
        else:
            qin = self.qli(q).view(-1, self.thw_dim, self.c_dim)
        if self.kli is None:
            kin = k.view(-1, self.c_dim, self.tf_dim)
        else:
            kin = self.kli(k).view(-1, self.c_dim, self.tf_dim)
        if self.vli is None:
            vin = v.view(-1, self.tf_dim, self.c_dim)
        else:
            vin = self.vli(v).view(-1, self.tf_dim, self.c_dim)
        matmulled = torch.matmul(qin, kin)
        finalout = torch.matmul(self.softmax(matmulled), vin)
        return torch.flatten(qin + finalout, 1)
