import torch
from torch import nn
from torch.nn import functional as F
import pdb


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
    # TODO
    def __init__(self, input_dims, output_dims):
        '''
        Args:
            TODO
        Output:
            TODO
        '''
        super(FiLM, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

    def forward(self, modalities, training=False):
        return


class MultiplicativeInteractions(nn.Module):
    # TODO
    # see https://openreview.net/pdf?id=rylnK6VtDH
    def __init__(self, input_dims, output_dim):
        '''
        Args:
            TODO
        Output:
            TODO
        '''
        super(MultiplicativeInteractions, self).__init__()
        self.input_dims = input_dims 
        self.output_dim = output_dim

        W_dims = self.input_dims + [output_dim]
        self.W = nn.Parameter(torch.Tensor(W_dims))
        nn.init.xavier_normal(self.W)
        self.U = nn.Parameter(torch.Tensor(self.input_dims[0], output_dim))
        nn.init.xavier_normal(self.U)
        self.V = nn.Parameter(torch.Tensor(self.input_dims[1], output_dim))
        nn.init.xavier_normal(self.V)
        self.b = nn.Parameter(torch.Tensor(self.output_dim))


    def forward(self, modalities, training=False):

        # TODO: extend to more than 2 modalities
        m1 = modalities[0]
        m2 = modalities[1]
        return


class TensorFusion(nn.Module):
    # https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py
    def __init__(self, input_dims):
        '''
        Args:
            TODO
        Output:
            TODO
        '''
        super(TensorFusion, self).__init__()
        self.input_dims = input_dims


    def forward(self, modalities, training=False):
        '''
        Args:
            TODO
        '''
        batch_size = modalities[0].shape[0]
        
        if len(modalities) == 1:
            fused_tensor = modalities[0]

        if len(modalities) == 2:
            m1_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False), modalities[0]), dim=1)
            m2_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False), modalities[1]), dim=1)
            fused_tensor = torch.bmm(m1_withones.unsqueeze(2), m2_withones.unsqueeze(1))
            fused_tensor = fused_tensor.view(batch_size, -1)

        if len(modalities) == 3:
            m1_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False), modalities[0]), dim=1)
            m2_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False), modalities[1]), dim=1)
            m3_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False), modalities[2]), dim=1)
            fused_tensor = torch.bmm(m1_withones.unsqueeze(2), m2_withones.unsqueeze(1))
            fused_tensor = fused_tensor.view(-1, (self.input_dims[0] + 1) * (self.input_dims[1] + 1), 1)
            fused_tensor = torch.bmm(fused_tensor, m3_withones.unsqueeze(1)).view(batch_size, -1)

        return fused_tensor


class LowRankTensorFusion(nn.Module):
    # https://github.com/Justin1904/Low-rank-Multimodal-Fusion
    def __init__(self, input_dims, output_dim, rank):
        '''
        Args:
            TODO
        Output:
            TODO
        '''
        super(LowRankTensorFusion, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank

        # low-rank factors
        self.factors = []
        for input_dim in range(input_dims):
            factor = nn.Parameter(torch.Tensor(self.rank, input_dim+1, self.output_dim))
            nn.init.xavier_normal(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))

        # init the fusion weights
        nn.init.xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities, training=False):
        '''
        Args:
            TODO
        '''
        batch_size = modalities[0].shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        fused_tensor = torch.ones(batch_size, 1).type(modality.dtype)
        for (modality, factor) in zip(modalities, self.factors):
            modality_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False), modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output


class NLgate(torch.nn.Module):
    # q_linear,k_liear,v_linear are none of no linear layer applied before q,k,v; otherwise, a tuple of (indim,outdim)
    # is inputted for each of these 3 arguments
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
