import torch
from torch import nn
from torch.nn import functional as F
import pdb
from torch.autograd import Variable

class Concat(nn.Module):
    def __init__(self):
        super(Concat,self).__init__()
    
    def forward(self, modalities, training=False):
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)


class ConcatWithLinear(nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Concat,self).__init__()
        self.fc = nn.Linear(input_dims, output_dim)
    
    def forward(self, modalities, training=False):
        return self.fc(torch.cat(modalities, dim=1))


class FiLM(nn.Module):
    #See https://arxiv.org/pdf/1709.07871.pdf
    def __init__(self, gamma_generation_network, beta_generation_network, base_modal=0, gamma_generate_modal=1, beta_generate_modal=1):
        super(FiLM, self).__init__()
        self.g_net = gamma_generation_network
        self.b_net = beta_generation_network
        self.base_modal = base_modal
        self.ggen_modal = gamma_generate_modal
        self.bgen_modal = beta_generate_modal

    def forward(self, modalities, training=False):
        gamma = self.g_net(modalities[self.ggen_modal])
        beta = self.b_net(modalities[self.bgen_modal])
        return gamma * modalities[self.base_modal] + beta

class MultiplicativeInteractions3Modal(nn.Module):
    def __init__(self,input_dims, output_dim, output):
        super(MultiplicativeInteractions3Modal, self).__init__()
        if output == 'vector':
            self.a = MultiplicativeInteractions2Modal([input_dims[0],input_dims[1]],
                    [input_dims[2],output_dim],'matrix')
            self.b = MultiplicativeInteractions2Modal([input_dims[0],input_dims[1]],
                    output_dim,'vector')
        elif output == 'scalar':
            self.a = MultiplicativeInteractions2Modal([input_dims[0],input_dims[1]],
                    input_dims[2],'vector')
            self.b = MultiplicativeInteractions2Modal([input_dims[0],input_dims[1]],
                    1,'scalar')
        self.output = output
    def forward(self, modalities, training=False):
        if self.output == 'vector':
            return torch.matmul(modalities[2],self.a(modalities[0:2]))+self.b(modalities[0:2])
        elif self.output == 'scalar':
            return torch.sum(modalities[2]*self.a(modalities[0:2]),dim=1)+self.b(modalities[0:2])




class MultiplicativeInteractions2Modal(nn.Module):
    def __init__(self, input_dims, output_dim, output):
        super(MultiplicativeInteractions2Modal, self).__init__()
        self.input_dims = input_dims 
        self.output_dim = output_dim
        self.output = output

        # most general Hypernetworks as Multiplicative Interactions.
        if output == 'matrix':
            self.W = nn.Parameter(torch.Tensor(self.input_dims[0],self.input_dims[1], output_dim[0],output_dim[1]))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0], output_dim[0],output_dim[1]))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(input_dims[1], output_dim[0],output_dim[1]))
            nn.init.xavier_normal(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim[0],output_dim[1]))
        
        # Diagonal Forms and Gating Mechanisms.
        elif output == 'vector':
            self.W = nn.Parameter(torch.Tensor(self.input_dims[0],self.input_dims[1],output_dim))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0],output_dim))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(self.input_dims[1],output_dim))
            nn.init.xavier_normal(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim))
            
        # Scales and Biases.
        elif output == 'scalar':
            self.W = nn.Parameter(torch.Tensor(self.input_dims[0],self.input_dims[1]))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(self.input_dims[0]))
            nn.init.normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(1))


    def forward(self, modalities, training=False):
        if len(modalities) == 1:
            return modalities[0]
        elif len(modalities) > 2:
            assert False
        m1 = modalities[0]
        m2 = modalities[1]

        # Hypernetworks as Multiplicative Interactions.
        if self.output == 'matrix':
            Wprime = torch.einsum('bn, nmpq -> bmpq', m1, self.W) + self.V    # bmpq
            bprime = torch.einsum('bn, npq -> bpq', m1, self.U) + self.b      # bpq
            output = torch.einsum('bm, bmpq -> bpq', m2, Wprime) + bprime     # bpq

        # Diagonal Forms and Gating Mechanisms.
        elif self.output == 'vector':
            Wprime = torch.einsum('bn, nmd -> bmd', m1, self.W) + self.V      # bmd
            bprime = torch.einsum('bn, nd -> bd', m1, self.U) + self.b      # bmd
            output = torch.einsum('bm, bmd -> bd',m2, Wprime) + bprime             # bmd
            
        # Scales and Biases.
        elif self.output == 'scalar':
            Wprime = torch.einsum('bn, nm -> bm', m1, self.W) + self.V      # bm
            bprime = torch.einsum('bn, n -> b', m1, self.U) + self.b      # b
            output = torch.einsum('bm, bm -> b',Wprime, m2) + bprime             # bm
        return output

class TensorFusion(nn.Module):
    # https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py
    # only works for 1,2,3 modalities
    def __init__(self, input_dims):
        super(TensorFusion, self).__init__()
        self.input_dims = input_dims


    def forward(self, modalities, training=False):
        batch_size = modalities[0].shape[0]
        
        if len(modalities) == 1:
            fused_tensor = modalities[0]

        if len(modalities) == 2:
            m1_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modalities[0].dtype), requires_grad=False), modalities[0]), dim=1)
            m2_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modalities[1].dtype), requires_grad=False), modalities[1]), dim=1)
            fused_tensor = torch.bmm(m1_withones.unsqueeze(2), m2_withones.unsqueeze(1))
            fused_tensor = fused_tensor.view(batch_size, -1)

        if len(modalities) == 3:
            m1_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modalities[0].dtype), requires_grad=False), modalities[0]), dim=1)
            m2_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modalities[1].dtype), requires_grad=False), modalities[1]), dim=1)
            m3_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modalities[2].dtype), requires_grad=False), modalities[2]), dim=1)
            fused_tensor = torch.bmm(m1_withones.unsqueeze(2), m2_withones.unsqueeze(1))
            fused_tensor = fused_tensor.view(-1, (self.input_dims[0] + 1) * (self.input_dims[1] + 1), 1)
            fused_tensor = torch.bmm(fused_tensor, m3_withones.unsqueeze(1)).view(batch_size, -1)

        return fused_tensor


class LowRankTensorFusion(nn.Module):
    # https://github.com/Justin1904/Low-rank-Multimodal-Fusion
    def __init__(self, input_dims, output_dim, rank):
        super(LowRankTensorFusion, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank

        # low-rank factors
        self.factors = []
        for input_dim in input_dims:
            factor = nn.Parameter(torch.Tensor(self.rank, input_dim+1, self.output_dim))
            nn.init.xavier_normal(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))

        # init the fusion weights
        nn.init.xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities, training=False):
        batch_size = modalities[0].shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        fused_tensor = 1
        for (modality, factor) in zip(modalities, self.factors):
            modality_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False), modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output


class NLgate(torch.nn.Module):
    # q_linear, k_linear, v_linear are none of no linear layer applied before q,k,v;
    # otherwise, a tuple of (indim,outdim) is inputted for each of these 3 arguments
    # See section F4 of "What makes training MM classification networks hard for details"
    def __init__(self, thw_dim, c_dim, tf_dim, q_linear=None, k_linear=None, v_linear=None):
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
    
    def forward(self, x, training=False):
        q = x[0]
        k = x[1]
        v = x[1]
        if self.qli is None:
          qin = q.view(-1, self.thw_dim, self.c_dim)
        else:
          qin = self.qli(q).view(-1, self.thw_dim, self.c_dim)
        if self.kli is None:
          kin = k.view(-1, c_dim, tf_dim)
        else:
          kin = self.kli(k).view(-1, self.c_dim, self.tf_dim)
        if self.vli is None:
          vin = v.view(-1, tf_dim, c_dim)
        else:
          vin = self.vli(v).view(-1, self.tf_dim, self.c_dim)
        matmulled = torch.matmul(qin, kin)
        finalout = torch.matmul(self.softmax(matmulled), vin)
        return torch.flatten(qin + finalout,1)
