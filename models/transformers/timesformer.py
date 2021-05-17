import numpy as np
import torch
from einops import rearrange
# from self_attention_cv.linformer.linformer import project_vk_linformer
# from self_attention_cv.transformer_vanilla.mhsa import compute_mhsa
# from ..common import expand_to_batch
#
from einops import repeat
from torch import nn

def project_vk_linformer(v, k, E):
    # project k,v
    v = torch.einsum('b h j d , j k -> b h k d', v, E)
    k = torch.einsum('b h j d , j k -> b h k d', k, E)
    return v, k

def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)

def split_cls(x):
    """
    split the first token of the sequence in dim 2
    """
    # Note by indexing the first element as 0:1 the dim is kept in the tensor's shape
    return x[:, :, 0:1, ...], x[:, :, 1:, ...]

def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * scale_factor

    if mask is not None:
        assert mask.shape == scaled_dot_prod.shape[2:]
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

    attention = torch.softmax(scaled_dot_prod, dim=-1)
    # calc result per head
    return torch.einsum('... i j , ... j d -> ... i d', attention, v)

def time_att_rearrange(x, frames):
    return rearrange(x, 'b h (f p) d -> (b p) h f d', f=frames)


def space_att_rearrange(x, patches):
    return rearrange(x, 'b h (f p) d -> (b f) h p d', p=patches)


def merge_timespace(x, batch, space=False):
    out_indices = 'b h (k t) d' if space else 'b h (t k) d'
    return rearrange(x, f'(b k) h t d -> {out_indices}', b=batch)


class SpacetimeMHSA(nn.Module):
    def __init__(self, dim, tokens_to_attend, space_att, heads=8,
                 dim_head=None, classification=True,
                 linear_spatial_attention=False, k=None):
        """
        Attention through time and space to process videos
        choose mode (whether to operate in space and time with space_att (bool) )
        CLS token is used for video classification, which will attend all tokens in both
        space and time before attention only in time or space.
        Code is based on lucidrains repo: https://github.com/lucidrains/TimeSformer-pytorch
        Args:
            dim: token's dimension, i.e. word embedding vector size
            tokens_to_attend: space (patches) or time (frames) tokens that we will attend to
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head=dim//heads.
            space_att: whether to use space or time attention in this block
            classification: when True a classification token is expected in the forward call
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5
        self.space_att = space_att
        self.reshape_timespace = space_att_rearrange if self.space_att else time_att_rearrange
        self.tokens_to_attend = tokens_to_attend
        self.classification = classification
        self.linear_spatial_attention = linear_spatial_attention and self.space_att
        self.k = k if k is not None else 256

        if self.linear_spatial_attention:
            proj_shape = tuple((self.tokens_to_attend + 1, k))
            self.E = torch.nn.Parameter(torch.randn(proj_shape))

    def forward(self, x):
        """
        Expects input x with merged tokens in both space and time
        Args:
            x: [batch, tokens_timespace+ cls_token, dim*3*heads ]
        """
        assert x.dim() == 3
        batch, token_dim = x.shape[0], 2
        qkv = self.to_qvk(x)

        # decomposition to q,v,k and cast to tuple
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        if self.classification:
            # only the cls token will to ALL tokens in both space+time: tokens_timespace
            (cls_q, q_3D) = split_cls(q)
            out_cls = compute_mhsa(cls_q, k, v, scale_factor=self.scale_factor)

            # reshape for space or time attention here only
            (cls_k, k_3D), (cls_v, v_3D) = map(split_cls, (k, v))

            # this is where we decompose/separate the tokens for attention in time or space only
            q_sep, k_sep, v_sep = map(self.reshape_timespace, [q_3D, k_3D, v_3D], [self.tokens_to_attend] * 3)

            # we have to expand/repeat the cls_k, and cls_v to k,v
            cls_k, cls_v = map(expand_to_batch, (cls_k, cls_v), (k_sep.shape[0], v_sep.shape[0]))

            k = torch.cat((cls_k, k_sep), dim=token_dim)
            v = torch.cat((cls_v, v_sep), dim=token_dim)

            if self.linear_spatial_attention:
                v, k = project_vk_linformer(v, k, self.E)

            # finally the conventional attention only through space/time
            out_mhsa = compute_mhsa(q_sep, k, v, scale_factor=self.scale_factor)

            # merge tokens from space and time
            out_mhsa = merge_timespace(out_mhsa, batch, self.space_att)
            # and spacetime cls token
            out = torch.cat((out_cls, out_mhsa), dim=token_dim)
        else:
            out = compute_mhsa(q, k, v)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)


class TimeSformerBlock(nn.Module):
    def __init__(self, *, frames, patches, dim=512,
                 heads=None, dim_linear_block=1024,
                 activation=nn.GELU,
                 dropout=0.1, classification=True,
                 linear_spatial_attention=False, k=None):
        """
        Args:
            dim: token's dim
            heads: number of heads
            linear_spatial_attention: if True Linformer-based attention is applied
        """
        super().__init__()
        self.frames = frames
        self.patches = patches
        self.classification = classification

        self.time_att = nn.Sequential(nn.LayerNorm(dim),
                                      SpacetimeMHSA(dim, tokens_to_attend=self.frames, space_att=False,
                                                    heads=heads, classification=self.classification))

        self.space_att = nn.Sequential(nn.LayerNorm(dim),
                                       SpacetimeMHSA(dim, tokens_to_attend=self.patches, space_att=True,
                                                     heads=heads, classification=self.classification,
                                                     linear_spatial_attention=linear_spatial_attention, k=k))

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_linear_block),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        assert x.dim() == 3
        x = self.time_att(x) + x
        x = self.space_att(x) + x
        x = self.mlp(x) + x
        return x


class Timesformer(nn.Module):
    def __init__(self, *,
                 img_dim, frames,
                 num_classes=None,
                 in_channels=3,
                 patch_dim=16,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 activation=nn.GELU,
                 dropout=0,
                 linear_spatial_attention=False, k=None):
        """
        Adapting ViT for video classification.
        Best strategy to handle multiple frames so far is
        Divided Space-Time Attention (T+S). We apply attention to projected
        image patches, first in time and then in both spatial dims.
        Args:
            img_dim: the spatial image size
            frames: video frames
            num_classes: classification task classes
            in_channels: number of img channels
            patch_dim: desired patch dim
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            linear_spatial_attention: for Linformer linear attention
            k: for Linformer linear attention
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible by img dim {img_dim}'
        self.p = patch_dim
        # classification: creates an extra CLS token that we will index in the final classification layer
        self.classification = True if num_classes is not None else False
        img_patches = (img_dim // patch_dim) ** 2
        # tokens = number of img patches * number of frames
        tokens_spacetime = frames * img_patches
        self.token_dim = in_channels * (patch_dim ** 2)
        self.dim = dim
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head

        # Projection and pos embeddings
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb1D = nn.Parameter(torch.randn(tokens_spacetime + 1, dim))
        if self.classification:
            self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        transormer_blocks = [TimeSformerBlock(
            frames=frames, patches=img_patches, dim=dim,
            heads=heads, dim_linear_block=dim_linear_block,
            activation=activation,
            dropout=dropout,
            linear_spatial_attention=linear_spatial_attention, k=k)
            for _ in range(blocks)]

        self.transformer = nn.Sequential(*transormer_blocks)

    def forward(self, vid):
        # Create patches as in ViT wherein frames are merged with patches
        # from [batch, frames, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        img_patches = rearrange(vid,
                                'b f c (patch_x x) (patch_y y) -> b (f x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)

        batch_size, tokens_spacetime, _ = img_patches.shape

        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        img_patches = torch.cat((expand_to_batch(self.cls_token, batch_size), img_patches), dim=1)
        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches = img_patches + self.pos_emb1D[:tokens_spacetime + 1, :]
        patch_embeddings = self.emb_dropout(img_patches)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings)

        # we index only the cls token for classification. nlp tricks :P
        return self.mlp_head(y[:, 0, :]) if self.classification else y[:, 0, :]
