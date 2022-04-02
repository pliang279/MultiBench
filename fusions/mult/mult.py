"""Implements the MultimodalTransformer Model. See https://github.com/yaohungt/Multimodal-Transformer for more."""
import torch
import torch.nn.functional as F
from torch import nn
from .modules.transformer import TransformerEncoder


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
            if type(h) == tuple:
                h = h[0]
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
