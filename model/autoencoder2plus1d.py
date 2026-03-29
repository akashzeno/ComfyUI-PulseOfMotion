"""2+1D Video Autoencoder with temporal 1D CNN - encoder path only for inference."""

import math
import torch
import torch.nn as nn
from einops import rearrange

from .ae_modules import Normalize, nonlinearity
from .attention import (
    normalization, zero_module, conv_nd,
    RelativePosition, QKVAttention, CrossAttention as CrossAttentionBase,
    default, exists,
)
from .distributions import DiagonalGaussianDistribution


class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.conv = torch.nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1),
        )
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        h = self.norm(x)
        h = nonlinearity(h)
        h = self.conv(h)
        return h


class ResnetBlock2plus1D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv1_tmp = TemporalConvLayer(out_channels, out_channels)

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv2_tmp = TemporalConvLayer(out_channels, out_channels)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3_tmp = TemporalConvLayer(out_channels, out_channels)

    def forward(self, x, temb=None, mask_temporal=False):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if not mask_temporal:
            h = self.conv1_tmp(h) + h

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if not mask_temporal:
            h = self.conv2_tmp(h) + h

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
            if not mask_temporal:
                x = self.conv3_tmp(x) + x

        return x + h


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        # SDPA expects (batch, num_heads, seq_len, head_dim) — use 1 head with dim=c
        q = rearrange(q, "b c t h w -> (b t) 1 (h w) c")
        k = rearrange(k, "b c t h w -> (b t) 1 (h w) c")
        v = rearrange(v, "b c t h w -> (b t) 1 (h w) c")

        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        h_ = rearrange(h_, "(b t) 1 (h w) c -> b c t h w", b=b, t=t, h=h)
        h_ = self.proj_out(h_)
        return x + h_


class TemporalAttention(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1, max_temporal_length=64):
        super().__init__()
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            self.num_heads = channels // num_head_channels

        self.norm = normalization(channels)
        self.qkv = zero_module(conv_nd(1, channels, channels * 3, 1))
        self.attention = QKVAttention(self.num_heads)
        self.relative_position_k = RelativePosition(
            num_units=channels // self.num_heads, max_relative_position=max_temporal_length
        )
        self.relative_position_v = RelativePosition(
            num_units=channels // self.num_heads, max_relative_position=max_temporal_length
        )
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, mask=None):
        b, c, t, h, w = x.shape
        out = rearrange(x, "b c t h w -> (b h w) c t")
        qkv = self.qkv(self.norm(out))
        len_q = qkv.size()[-1]
        k_rp = self.relative_position_k(len_q, len_q)
        v_rp = self.relative_position_v(len_q, len_q)
        out = self.attention(qkv, rp=(k_rp, v_rp))
        out = self.proj_out(out)
        out = rearrange(out, "(b h w) c t -> b c t h w", b=b, h=h, w=w)
        return x + out


class Downsample2plus1D(nn.Module):
    def __init__(self, in_channels, with_conv, temp_down):
        super().__init__()
        self.with_conv = with_conv
        self.temp_down = temp_down
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)

    def forward(self, x, mask_temporal=False):
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 0)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        return x


class Encoder2plus1D(nn.Module):
    def __init__(
        self, *, ch, out_ch, temporal_down_factor, ch_mult=(1, 2, 4, 8),
        num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True,
        in_channels, resolution, z_channels, double_z=True, **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.n_temporal_down = int(math.log2(temporal_down_factor))
        self.num_res_blocks = num_res_blocks

        self.conv_in = torch.nn.Conv3d(in_channels, self.ch, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock2plus1D(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout)
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                temp_down = i_level <= self.n_temporal_down - 1
                down.downsample = Downsample2plus1D(block_in, resamp_with_conv, temp_down)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2plus1D(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.attn_1_tmp = TemporalAttention(block_in, num_heads=1)
        self.mid.block_2 = ResnetBlock2plus1D(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in, 2 * z_channels if double_z else z_channels,
            kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
        )

    def forward(self, x, text_embeddings=None, text_attn_mask=None, mask_temporal=False):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb, mask_temporal)
                if len(self.down[i_level].attn) > 0:
                    h = h + self.down[i_level].attn[i_block](h, context=text_embeddings, mask=text_attn_mask)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1], mask_temporal))
        h = hs[-1]
        h = self.mid.block_1(h, temb, mask_temporal)
        h = self.mid.attn_1(h)
        if not mask_temporal:
            h = self.mid.attn_1_tmp(h)
        h = self.mid.block_2(h, temb, mask_temporal)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class AutoencoderKL2plus1D_1dcnn(nn.Module):
    """2+1D Video Autoencoder with temporal 1D CNN. Inference-only version."""

    def __init__(
        self,
        ddconfig,
        ppconfig,
        lossconfig=None,
        embed_dim=0,
        use_quant_conv=True,
        ckpt_path=None,
        **kwargs,
    ):
        super().__init__()
        self.use_quant_conv = use_quant_conv

        # 2+1D encoder (the only component used for FPS prediction)
        self.encoder = Encoder2plus1D(**ddconfig)

        if use_quant_conv:
            assert embed_dim
            self.embed_dim = embed_dim
            self.quant_conv = torch.nn.Conv3d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)

        if ckpt_path is not None:
            self._init_from_ckpt(ckpt_path)

    def _init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        try:
            sd = sd["state_dict"]
        except (KeyError, TypeError):
            pass
        self.load_state_dict(sd, strict=False)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
