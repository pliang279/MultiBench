"""Implements Positional Encoding.

Adapted from fairseq repo.
"""

import math

import torch
import torch.nn as nn



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

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        """Instantiate SinusoidalPositionalEmbedding Module.

        Args:
            embedding_dim (int): Embedding dimension
            padding_idx (int, optional): Padding index. Defaults to 0.
            left_pad (int, optional): Whether to pad from the left or not. Defaults to 0.
            init_size (int, optional): Initial size. Defaults to 128.
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

    def max_positions(self):
        """Get the maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
