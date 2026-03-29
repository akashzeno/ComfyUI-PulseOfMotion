"""Attention modules for the Visual Chronometer video autoencoder."""

from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class GroupNormSpecific(nn.GroupNorm):
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            return super().forward(x).type(x.dtype)
        else:
            return super().forward(x.float()).type(x.dtype)


def normalization(channels, num_groups=32):
    return GroupNormSpecific(num_groups, channels)


class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, num_units)
        )
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, rp=None, mask=None, use_tempoal_causal_attn=False):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )

        if rp is not None:
            k_rp, v_rp = rp
            weight2 = torch.einsum(
                "bct,tsc->bst", (q * scale).view(bs * self.n_heads, ch, length), k_rp
            )
            weight += weight2

        if mask is not None:
            INF = -1e8
            weight = weight.float().masked_fill(mask == 0, INF)

        weight = F.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )

        if rp is not None:
            a2 = torch.einsum("bts,tsc->btc", weight, v_rp).transpose(1, 2)
            a += a2

        return a.reshape(bs, -1, length)


class CrossAttention(nn.Module):
    """Cross-attention used in the FPS predictor's attention pooling."""

    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        b = x.shape[0]

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        attn_mask = None
        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            attn_mask = repeat(mask, "b j -> b h () j", h=h)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
