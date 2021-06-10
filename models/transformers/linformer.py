


import torch
import torch.nn as nn
from einops import rearrange,repeat
from models.transformers.transformer import TransformerBlock,compute_mhsa


def project_vk_linformer(v, k, E):
    # project k,v
    v = torch.einsum('b h j d , j k -> b h k d', v, E)
    k = torch.einsum('b h j d , j k -> b h k d', k, E)
    return v, k


class LinformerAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, shared_projection=True, proj_shape=None, trainable_proj=True):
        """
        Based on the Linformer paper
        Link: https://arxiv.org/pdf/2006.04768.pdf
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head.
            shared_projection: if the projection matrix will be shared among layers
            (it will have to be passed in the forward that way)
            trainable_proj: if the projection matrix E matrix is not shared,
            you can enable this option to make it trainable (non trainable in the paper)
            proj_shape: 2-tuple (tokens,k), where k is the projection dimension of the linformer
            """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5
        self.shared_projection = shared_projection

        if not shared_projection:
            self.E = torch.nn.Parameter(torch.randn(proj_shape), requires_grad=trainable_proj)
            self.k = proj_shape[1]

    def forward(self, x, proj_mat=None):
        assert x.dim() == 3
        E = proj_mat if (self.shared_projection and proj_mat is not None) else self.E
        assert x.shape[1] == E.shape[0], f'{x.shape[1]} Token in the input sequence while' \
                                         f' {E.shape[0]} were provided in the E proj matrix'

        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        v, k = project_vk_linformer(v, k, E)

        out = compute_mhsa(q, k, v, scale_factor=self.scale_factor)
        # re-compose: merge heads with dim_head

        out = rearrange(out, "b h i d -> b i (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)

class LinformerBlock(TransformerBlock):
    def __init__(self, dim, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1,
                 shared_projection=False, proj_shape=None,
                 trainable_proj=False, activation=nn.GELU):
        super().__init__(dim=dim, dim_linear_block=dim_linear_block, dropout=dropout, activation=activation)
        self.mhsa = LinformerAttention(dim=dim,
                                       heads=heads,
                                       dim_head=dim_head,
                                       shared_projection=shared_projection,
                                       proj_shape=proj_shape,
                                       trainable_proj=trainable_proj)

    def forward(self, x, proj_mat=None):
        return super().forward(x, proj_mat)


class LinformerEncoder(nn.Module):
    def __init__(self, dim, tokens, k=None, blocks=4, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1,
                 shared_projection=True,
                 trainable_proj=True, activation=nn.GELU):
        """
        Based on the Linformer paper
        Link: https://arxiv.org/pdf/2006.04768.pdf
        Args:
            dim: token's dimension, i.e. word embedding vector size
            tokens: sequence length
            blocks: number of sequential blocks
            heads: the number of distinct representations to learn per block
            dim_head: the dim of the head.
            dim_linear_block: reprojection dim. usually multiple of dim
            dropout: dropout in mhsa block
            activation: MHSA block activation
            Specific parameters for Linformer:
            Headwise sharing and Key-value sharing are on by default!
            shared_projection: if the projection matrix E will be shared among layers
            (it will have to be passed in the forward that way)
            trainable_proj: you can enable this option to make it trainable (non-trainable in the paper)
            tokens: (tokens, projection dimension k), where k is the projection dimension of the linformer
            Choice of k for sequences of length n so that
            Linformer’s performance is nearly on par with the original Transformer:
            Practical:
            k = 128 for n = 512
            k = 256 for n = 1024
            Default is n/4 as in most of the paper's experiments
            """
        super().__init__()
        self.shared_projection = shared_projection
        self.k = k if k is not None else tokens // 4
        proj_shape = [tokens, self.k]

        if self.shared_projection:
            self.E = torch.nn.Parameter(torch.randn(proj_shape), requires_grad=trainable_proj)

        self.block_list = [LinformerBlock(dim=dim, heads=heads, dim_head=dim_head,
                                          dim_linear_block=dim_linear_block, dropout=dropout,
                                          shared_projection=shared_projection, proj_shape=proj_shape,
                                          trainable_proj=trainable_proj, activation=activation)
                           for _ in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x):
        for layer in self.layers:
            if self.shared_projection:
                x = layer(x, self.E)
            else:
                x = layer(x)
        return x