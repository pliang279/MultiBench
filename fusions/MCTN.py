"""Implements MCTN for Fusions."""

from torch.autograd import Variable
import random
import math
from torch.nn import functional as F
from torch import nn
import torch
import sys
import os

from torch.serialization import save

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


class Encoder(nn.Module):
    """
    Apply a gated GRU to encode the input vector.
    
    Paper/Code Sourced From: https://arxiv.org/pdf/1812.07809.pdf.
    """

    def __init__(self, input_size, hidden_size,
                 n_layers=1, dropout=0.2):
        """
        Create Encoder.

        :param input_size: Encoder input size
        :param hidden_size: Hidden state size for internal GRU
        :param n_layers: Number of layers in recurrent unit
        :param dropout: Dropout Probability
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        """
        Apply Encoder to input.

        :param src: Encoder Input
        :param hidden: Encoder Hidden State
        """
        outputs, hidden = self.gru(src, hidden)

        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    """Implements Attention Mechanism for MCTN."""

    def __init__(self, hidden_size):
        """Initialize Attention Mechanism for MCTN.

        Args:
            hidden_size (int): Hidden Size of Attention Layer.
        
        """
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        """Apply Attention to Input, with Hidden Layers.

        Args:
            hidden : Initial hidden state.
            encoder_outputs : Outputs of Encoder object.

        Returns:
            output: Output of Attention layer

        """
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self._score(h, encoder_outputs)
        return F.relu(attn_energies).unsqueeze(1)

    def _score(self, hidden, encoder_outputs):
        energy = F.softmax(
            self.attn(torch.cat([hidden, encoder_outputs], 2)), dim=1)
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    """
    Apply a gated GRU to decode the input vector.
    
    Paper/Code Sourced From: https://arxiv.org/pdf/1812.07809.pdf.
    """

    def __init__(self, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        """Initialize Decoder Mechanism for MCTN.

        Args:
            hidden_size (int): Size of hidden layer.
            output_size (int): Size of output layer
            n_layers (int, optional): Number of layers in encoder. Defaults to 1.
            dropout (float, optional): Dropout percentage. Defaults to 0.2.
        
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + output_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        """Apply Decoder Mechanism for MCTN.

        Args:
            input: Input to MCTN Mechanism.
            last_hidden: Last hidden layer input.
            encoder_outputs: Output of Encoder object.

        Returns:
            output: Output of this module.

        """
        # Get the embedding of the current input word (last output word)
        embedded = input.unsqueeze(0)  # (1,B,N)
        

        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    """Implements a Seq2Seq Layer for MCTN."""

    def __init__(self, encoder, decoder):
        """Initialize Seq2Seq Module.

        Args:
            encoder (nn.Module): Encoder for the Seq2Seq Layer.
            decoder (nn.Module): Decoder for the Seq2Seq Layer.
        
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """Apply Seq2Seq Module to Input.

        Args:
            src (torch.Tensor): Seq2Seq Input
            trg (torch.Tensor): Seq2Seq Output for Teacher Forcing.
            teacher_forcing_ratio (float, optional): Teacher Forcing Ratio. Set to 0 when evaluating. Defaults to 0.5.

        Returns:
            _type_: _description_
        
        """
        batch_size = src.size(1)
        max_len = src.size(0)

        output_size = self.decoder.output_size
        outputs = Variable(torch.zeros(
            max_len, batch_size, output_size)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]

        
        

        if self.training:
            output = Variable(
                torch.zeros_like(trg.data[0, :]))  # solve the bug of input.size must be equal to input_size
        else:
            output = Variable(torch.zeros_like(src.data[0, :]))
        
        
        for t in range(0, max_len):
            output, hidden, attn_weights = self.decoder(
                output, hidden, encoder_output)
            outputs[t] = output

            is_teacher = random.random() < teacher_forcing_ratio
            if is_teacher:
                output = Variable(trg.data[t]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        return outputs, encoder_output


class MCTN(nn.Module):
    """Implements MCTN."""

    def __init__(self, seq2seq, regression_encoder, head, p=0.2):
        """Initialize MCTN object.

        Args:
            seq2seq (nn.Module): Seq2Seq module for MCTN.
            regression_encoder (nn.Module): Encoder module for MCTN.
            head (nn.Module): Head for MCTN.
            p (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(MCTN, self).__init__()
        self.seq2seq = seq2seq
        # self.regression = nn.GRU(embedd_dim, reg_hidden_dim,
        #                   n_layers, dropout=dropout)
        self.dropout = nn.Dropout(p)
        self.regression = regression_encoder
        self.head = head

    def forward(self, src, trg=None):
        """Apply Cyclic Joint Embedding in MCTN.

        Args:
            src (torch.Tensor): Input Tensor
            trg (torch.Tensor, optional): Output Tensor for Teacher-Forcing. Defaults to None.

        Returns:
            torch.Tensor: Output after applying MCTN.
        """
        # get the cyclic joint embedding!
        reout = None
        
        if self.training:
            out, _ = self.seq2seq(src, trg)
            
            reout, joint_embbed = self.seq2seq(out, src)
        else:
            # Set teacher_forcing_ratio to zero to get rid of the input of target during inference stage
            out, _ = self.seq2seq(src, trg, teacher_forcing_ratio=0.0)
            joint_embbed, _ = self.seq2seq.encoder(out)
        _, reg = self.regression(joint_embbed)
        
        reg = self.dropout(reg)
        head_out = self.head(reg)[0]
        head_out = self.dropout(head_out)
        return out, reout, head_out


class L2_MCTN(nn.Module):
    """Implements 2-Output MCTN."""

    def __init__(self, seq2seq_0, seq2seq_1, regression_encoder, head, p=0.2):
        """Initialize L2_MCTN.

        Args:
            seq2seq_0 (nn.Module): Seq2Seq Module converting input to target 1.
            seq2seq_1 (nn.Module): Seq2Seq Module converting input to target 2.
            regression_encoder (nn.Module): Encoder applied to joint embedding.
            head (nn.Module): Head module.
            p (float, optional): Dropout percentage. Defaults to 0.2.
        
        """
        super(L2_MCTN, self).__init__()
        self.seq2seq0 = seq2seq_0
        self.seq2seq1 = seq2seq_1

        self.dropout = nn.Dropout(p)
        self.regression = regression_encoder
        self.head = head

    def forward(self, src, trg0=None, trg1=None):
        """Apply L2_MCTN to input.

        Args:
            src (torch.Tensor): Input tensor.
            trg0 (torch.Tensor, optional): Target output for Seq2Seq Module 1 Teacher Forcing. Defaults to None.
            trg1 (torch.Tensor, optional): Target output for Seq2Seq Module 2 Teacher Forcing. Defaults to None.

        Returns:
            torch.Tensor: Output for L2_MCTN instance.
        """
        reout = None
        rereout = None
        if self.training:
            out, _ = self.seq2seq0(src, trg0)
            
            reout, joint_embbed0 = self.seq2seq0(out, src)
            
            
            rereout, joint_embbed1 = self.seq2seq1(joint_embbed0, trg1)
        else:
            out, _ = self.seq2seq0(src, trg0, teacher_forcing_ratio=0.0)
            _, joint_embbed0 = self.seq2seq0.encoder(out)
            _, joint_embbed1 = self.seq2seq1.encoder(joint_embbed0)
        _, reg = self.regression(joint_embbed1)
        reg = self.dropout(reg)
        head_out = self.head(reg)[0]
        head_out = self.dropout(head_out)
        return out, reout, rereout, head_out


""" def process_input(inputs, max_seq=20):
    src = inputs[0][2][:, :max_seq, :]
    trg = inputs[0][1][:, :max_seq, :]
    feature_dim = max(src.size(-1), trg.size(-1))
    

    if src.size(-1) > trg.size(-1):
        trg = F.pad(trg, (0, src.size(-1) - trg.size(-1)))
        # src = F.pad(src, (1, 0))
    else:
        src = F.pad(src, (0, trg.size(-1) - src.size(-1)))
        # trg = F.pad(trg, (1, 0))
    src = src.transpose(1, 0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    trg = trg.transpose(1, 0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    labels = inputs[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    return src, trg, labels, feature_dim

 """

