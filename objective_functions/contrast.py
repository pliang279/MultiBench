"""Implement objectives for contrastive loss."""
import torch
from torch import nn
import math


eps = 1e-7


class AliasMethod(object):
    """
    Initializes a generic method to sample from arbritrary discrete probability methods.
    
    Sourced From https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/.
    Alternatively, look here for more details: http://cgi.cs.mcgill.ca/~enewel3/posts/alias-method/index.html.
    """

    def __init__(self, probs):
        """Initialize AliasMethod object.

        Args:
            probs (list[int]): List of probabilities for each object. Can be greater than 1, but will be normalized.
        """
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        """Generate CUDA version of self, for GPU-based sampling."""
        self.prob = self.prob.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.alias = self.alias.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def draw(self, N):
        """
        Draw N samples from multinomial dkstribution, based on given probability array.
        
        :param N: number of samples
        :return: samples
        """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long,
                         device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj


class NCEAverage(nn.Module):
    """Implements NCEAverage Loss Function."""
    
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        """Instantiate NCEAverage Loss Function.

        Args:
            inputSize (int): Input Size
            outputSize (int): Output Size
            K (float): K Value. See paper for more.
            T (float, optional): T Value. See paper for more. Defaults to 0.07.
            momentum (float, optional): Momentum for NCEAverage Loss. Defaults to 0.5.
            use_softmax (bool, optional): Whether to use softmax or not. Defaults to False.
        """
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(
            outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(
            outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        """Apply NCEAverage Module.

        Args:
            l (torch.Tensor): Labels
            ab (torch.Tensor): See paper for more.
            y (torch.Tensor): True values.
            idx (torch.Tensor, optional): See paper for more. Defaults to None.

        Returns:
            _type_: _description_
        """
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(
                batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.view(-1))
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        # sample
        weight_ab = torch.index_select(
            self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print(
                    "normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print(
                    "normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y.view(-1), updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y.view(-1), updated_ab)

        return out_l, out_ab


class NCECriterion(nn.Module):
    """
    Implements NCECriterion Loss.
    
    Eq. (12): L_{NCE}
    """

    def __init__(self, n_data):
        """Instantiate NCECriterion Loss."""
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        """Apply NCECriterion to Tensor Input.

        Args:
            x (torch.Tensor): Tensor Input

        Returns:
            torch.Tensor: Loss
        """
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn),
                           P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class NCESoftmaxLoss(nn.Module):
    """Implements Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)."""

    def __init__(self):
        """Instantiate NCESoftmaxLoss Module."""
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """Apply NCESoftmaxLoss to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")).long()
        loss = self.criterion(x, label)
        return loss


class MultiSimilarityLoss(nn.Module):
    """Implements MultiSimilarityLoss."""
    
    def __init__(self,):
        """Initialize MultiSimilarityLoss Module."""
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 5e-2
        self.scale_neg = 2e-3

    def forward(self, feats, labels):
        """Apply MultiSimilarityLoss to Tensor Inputs.

        Args:
            feats (torch.Tensor): Features
            labels (torch.Tensor): Labels

        Returns:
            torch.Tensor: Loss output.
        """
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()
        total = 0

        for i in range(batch_size):
            for k in range(labels[i].size(0)):
                if labels[i][k] == 1:
                    total += 1
                    pos_pair_ = sim_mat[i][labels[:, k] == 1]
                    pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
                    neg_pair_ = sim_mat[i][labels[:, k] == 0]

                    if pos_pair_.size(0) == 0:
                        neg_pair = neg_pair_
                    else:
                        neg_pair = neg_pair_[neg_pair_ +
                                             self.margin > min(pos_pair_)]
                    pos_pair = pos_pair_[pos_pair_ -
                                         self.margin < max(neg_pair_)]

                    if len(neg_pair) < 1 or len(pos_pair) < 1:
                        continue
                    
                    # weighting step
                    pos_loss = 1.0 / self.scale_pos * torch.log(
                        1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
                    neg_loss = 1.0 / self.scale_neg * torch.log(
                        1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
                    loss.append(pos_loss + neg_loss)
                    assert math.isinf(pos_loss) == False
                    assert math.isinf(neg_loss) == False
        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / total
        return loss
