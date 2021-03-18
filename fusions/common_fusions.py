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


class TensorFusion(nn.Module):
    # # https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py
    def __init__(self, input_dims, output_dim):
        '''
        Args:
            TODO
        Output:
            TODO
        '''
        super(TensorFusion, self).__init__()

        self.input_dims = input_dims
        self.output_dim = output_dim


    def forward(self, modalities, training=False):
        '''
        Args:
            TODO
        '''
        batch_size = modalities[0].shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        fused_tensor = torch.ones(batch_size, 1).type(modality.dtype)
        for modality in modalities:
            modality_withones = torch.cat((Variable(torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False), modality), dim=1)
            torch.matmul(modality_withones, factor)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fused_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        
        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fused_tensor = fused_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
        fused_tensor = torch.bmm(fused_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)
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
        super(NLgate,self).__init__()
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
    
    def forward(self,q,k,v,training=False):
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
        return qin + finalout
