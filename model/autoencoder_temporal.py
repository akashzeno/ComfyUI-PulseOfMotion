"""Temporal 1D CNN encoder/decoder for the video autoencoder."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def silu(x):
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def forward(self, x):
        return silu(x)


def Normalize(in_channels, norm_type="group"):
    if norm_type == "group":
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == "batch":
        return torch.nn.SyncBatchNorm(in_channels)


class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type="replicate"):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type="group", padding_type="replicate"):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type)
        self.conv2 = SamePadConv3d(out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)
        return x + h


class EncoderTemporal1DCNN(nn.Module):
    def __init__(self, *, ch, out_ch, attn_temporal_factor=[], temporal_scale_factor=4, hidden_channel=128, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temporal_scale_factor = temporal_scale_factor

        self.conv_in = SamePadConv3d(ch, hidden_channel, kernel_size=3, padding_type="replicate")
        self.mid_blocks = nn.ModuleList()

        num_ds = int(math.log2(temporal_scale_factor))
        norm_type = "group"

        for i in range(num_ds):
            block = nn.Module()
            in_channels = hidden_channel * 2 ** i
            out_channels = hidden_channel * 2 ** (i + 1)
            temporal_stride = 2

            block.down = SamePadConv3d(
                in_channels, out_channels, kernel_size=3,
                stride=(temporal_stride, 1, 1), padding_type="replicate"
            )
            block.res = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.attn = nn.ModuleList()
            self.mid_blocks.append(block)

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type),
            SiLU(),
            SamePadConv3d(out_channels, out_ch * 2, kernel_size=3, padding_type="replicate"),
        )

    def forward(self, x, text_embeddings=None, text_attn_mask=None):
        h = self.conv_in(x)
        for block in self.mid_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class DecoderTemporal1DCNN(nn.Module):
    def __init__(self, *, ch, out_ch, attn_temporal_factor=[], temporal_scale_factor=4, hidden_channel=128, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temporal_scale_factor = temporal_scale_factor

        num_us = int(math.log2(temporal_scale_factor))
        norm_type = "group"

        enc_out_channels = hidden_channel * 2 ** num_us
        self.conv_in = SamePadConv3d(ch, enc_out_channels, kernel_size=3, padding_type="replicate")
        self.mid_blocks = nn.ModuleList()

        for i in range(num_us):
            block = nn.Module()
            in_channels = enc_out_channels if i == 0 else hidden_channel * 2 ** (num_us - i + 1)
            out_channels = hidden_channel * 2 ** (num_us - i)
            block.up = torch.nn.ConvTranspose3d(
                in_channels, out_channels,
                kernel_size=(3, 3, 3), stride=(2, 1, 1),
                padding=(1, 1, 1), output_padding=(1, 0, 0),
            )
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.attn1 = nn.ModuleList()
            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.attn2 = nn.ModuleList()
            self.mid_blocks.append(block)

        self.conv_last = SamePadConv3d(out_channels, out_ch, kernel_size=3)

    def forward(self, x, text_embeddings=None, text_attn_mask=None):
        h = self.conv_in(x)
        for block in self.mid_blocks:
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h
