"""Implements the MultimodalTransformer Model. See https://github.com/yaohungt/Multimodal-Transformer for more."""
import math
import torch
import torch.nn.functional as F
from torch import nn


class MULTModel(nn.Module):
    """
    Implements the MultimodalTransformer Model.
    
    See https://github.com/yaohungt/Multimodal-Transformer for more.
    """
    
    class DefaultHyperParams():
        """Set default hyperparameters for the model."""
        
        num_heads = 3
        layers = 3
        attn_dropout = 0.1
        attn_dropout_modalities = [0.0] * 1000
        relu_dropout = 0.1
        res_dropout = 0.1
        out_dropout = 0.0
        embed_dropout = 0.25
        embed_dim = 9
        attn_mask = True
        output_dim = 1
        all_steps = False

    def __init__(self, n_modalities, n_features, hyp_params=DefaultHyperParams):
        """Construct a MulT model."""
        super().__init__()
        self.n_modalities = n_modalities
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_modalities = hyp_params.attn_dropout_modalities
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.all_steps = hyp_params.all_steps

        combined_dim = self.embed_dim * n_modalities * n_modalities

        # This is actually not a hyperparameter :-)
        output_dim = hyp_params.output_dim

        # 1. Temporal convolutional layers
        self.proj = [nn.Conv1d(n_features[i], self.embed_dim, kernel_size=1,
                               padding=0, bias=False) for i in range(n_modalities)]
        self.proj = nn.ModuleList(self.proj)

        # 2. Crossmodal Attentions
        self.trans = [nn.ModuleList([self.get_network(i, j, mem=False) for j in range(
            n_modalities)]) for i in range(n_modalities)]
        self.trans = nn.ModuleList(self.trans)

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_mems = [self.get_network(
            i, i, mem=True, layers=3) for i in range(n_modalities)]
        self.trans_mems = nn.ModuleList(self.trans_mems)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, mod1, mod2, mem, layers=-1):
        """Create TransformerEncoder network from layer information."""
        if not mem:
            embed_dim = self.embed_dim
            attn_dropout = self.attn_dropout_modalities[mod2]
        else:
            embed_dim = self.n_modalities * self.embed_dim
            attn_dropout = self.attn_dropout

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x):
        """
        Apply MultModel Module to Layer Input.
        
        Args:
            x: layer input. Has size n_modalities * [batch_size, seq_len, n_features]
        """
        x = [v.permute(0, 2, 1)
             for v in x]  # n_modalities * [batch_size, n_features, seq_len]

        # Project the textual/visual/audio features
        proj_x = [self.proj[i](x[i]) for i in range(self.n_modalities)]
        proj_x = torch.stack(proj_x)
        # [n_modalities, seq_len, batch_size, proj]
        proj_x = proj_x.permute(0, 3, 1, 2)

        hs = []
        last_hs = []
        for i in range(self.n_modalities):
            h = []
            for j in range(self.n_modalities):
                h.append(self.trans[i][j](proj_x[i], proj_x[j], proj_x[j]))
            h = torch.cat(h, dim=2)
            h = self.trans_mems[i](h)
            # if type(h) == tuple:
            #     h = h[0]
            if self.all_steps:
                hs.append(h)
            else:
                last_hs.append(h[-1])

        if self.all_steps:
            out = torch.cat(hs, dim=2)  # [seq_len, batch_size, out_features]
            out = out.permute(1, 0, 2)  # [batch_size, seq_len, out_features]
        else:
            out = torch.cat(last_hs, dim=1)

        # A residual block
        out_proj = self.proj2(
            F.dropout(F.relu(self.proj1(out)), p=self.out_dropout, training=self.training))
        out_proj += out

        out = self.out_layer(out_proj)
        return out


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers.
    
    Each layer is a :class:`TransformerEncoderLayer`.
    
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        """Initialize Transformer Encoder.

        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of heads
            layers (int): Number of layers
            attn_dropout (float, optional): Probability of dropout in attention mechanism. Defaults to 0.0.
            relu_dropout (float, optional): Probability of dropout after ReLU. Defaults to 0.0.
            res_dropout (float, optional): Probability of dropout in residual layer. Defaults to 0.0.
            embed_dropout (float, optional): Probability of dropout in embedding layer. Defaults to 0.0.
            attn_mask (bool, optional): Whether to apply a mask to the attention or not. Defaults to False.
        """
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for _ in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """
        Apply Transformer Encoder to layer input.
        
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            # Add positional embedding
            x += self.embed_positions(x_in.transpose(0, 1)
                                      [:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                # Add positional embedding
                x_k += self.embed_positions(x_in_k.transpose(0, 1)
                                            [:, :, 0]).transpose(0, 1)
                # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)
                                            [:, :, 0]).transpose(0, 1)
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """Implements encoder layer block.
    
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        """Instantiate TransformerEncoderLayer Module.

        Args:
            embed_dim (int): Embedding dimension
            num_heads (int, optional): Number of heads. Defaults to 4.
            attn_dropout (float, optional): Dropout for attention mechanism. Defaults to 0.1.
            relu_dropout (float, optional): Dropout after ReLU. Defaults to 0.1.
            res_dropout (float, optional): Dropout after residual layer. Defaults to 0.1.
            attn_mask (bool, optional): Whether to apply an attention mask or not. Defaults to False.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        # The "Add & Norm" part in the paper
        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList(
            [LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Apply TransformerEncoderLayer to Layer Input.
        
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self._maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self._maybe_layer_norm(0, x_k, before=True)
            x_v = self._maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._maybe_layer_norm(0, x, after=True)

        residual = x
        x = self._maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._maybe_layer_norm(1, x, after=True)
        return x

    def _maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    """Generate buffered future mask.

    Args:
        tensor (torch.Tensor): Tensor to initialize mask from.
        tensor2 (torch.Tensor, optional): Tensor to initialize target mask from. Defaults to None.

    Returns:
        torch.Tensor: Buffered future mask.
    """
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(
        torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    """Generate Linear Layer with given parameters and Xavier initialization.

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool, optional): Whether to include a bias term or not. Defaults to True.

    Returns:
        nn.Module: Initialized Linear Module.
    """
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    """Generate LayerNorm Layer with given parameters.

    Args:
        embedding_dim (int): Embedding dimension

    Returns:
        nn.Module: Initialized LayerNorm Module
    """
    m = nn.LayerNorm(embedding_dim)
    return m


"""Implements Positional Encoding.

Adapted from fairseq repo.
"""
def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    
    Args:
        tensor (torch.Tensor): Tensor to generate padding on.   
        padding_idx (int): Position numbers start at padding_idx + 1
        left_pad (bool): Whether to pad from the left or from the right.

    Returns:
        torch.Tensor: Padded output
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(
        make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos,
                     out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[
        :tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - \
            mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """
    This module produces sinusoidal positional embeddings of any length.
    
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0):
        """Instantiate SinusoidalPositionalEmbedding Module.

        Args:
            embedding_dim (int): Embedding dimension
            padding_idx (int, optional): Padding index. Defaults to 0.
            left_pad (int, optional): Whether to pad from the left or not. Defaults to 0.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        # device --> actual weight; due to nn.DataParallel :-(
        self.weights = dict()
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Apply PositionalEncodings to Input.
        
        Input is expected to be of size [bsz x seqlen].

        Args:
            input (torch.Tensor): Layer input

        Returns:
            torch.Tensor: Layer output
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.reshape(-1)).reshape((bsz, seq_len, -1)).detach()
