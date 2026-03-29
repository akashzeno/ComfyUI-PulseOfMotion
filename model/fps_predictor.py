"""Visual Chronometer FPS Predictor - pure PyTorch inference module."""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .autoencoder2plus1d import AutoencoderKL2plus1D_1dcnn
from .attention import CrossAttention


class FPSPredictor(nn.Module):
    def __init__(
        self,
        ddconfig,
        ppconfig,
        lossconfig=None,
        embed_dim=4,
        use_quant_conv=True,
        ckpt_path=None,
        freeze_encoder=True,
        hidden_dim=1024,
        n_layers=1,
        **kwargs,
    ):
        super().__init__()
        self.freeze_encoder = freeze_encoder

        self.vae = AutoencoderKL2plus1D_1dcnn(
            ddconfig=ddconfig,
            ppconfig=ppconfig,
            lossconfig=lossconfig,
            embed_dim=embed_dim,
            use_quant_conv=use_quant_conv,
            ckpt_path=ckpt_path,
        )

        if self.freeze_encoder:
            self.vae.eval()
            self.vae.freeze()

        self.feat_dim = 2 * ddconfig["z_channels"] if use_quant_conv else ddconfig["z_channels"]
        self.probe_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.proj_in = nn.Linear(self.feat_dim, hidden_dim)
        self.n_layers = n_layers

        if n_layers == 1:
            self.attn_pool = CrossAttention(
                query_dim=hidden_dim, context_dim=hidden_dim,
                heads=8, dim_head=64, dropout=0.1,
            )
        else:
            self.attn_pool = nn.ModuleList([
                CrossAttention(
                    query_dim=hidden_dim, context_dim=hidden_dim,
                    heads=8, dim_head=64, dropout=0.1,
                ) for _ in range(n_layers)
            ])

        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        with torch.no_grad():
            latents = self.vae.encoder(x)
            if self.vae.use_quant_conv:
                latents = self.vae.quant_conv(latents)

        b, c, t, h, w = latents.shape
        latents = rearrange(latents, "b c t h w -> b (t h w) c")
        latents = self.proj_in(latents)

        probe = repeat(self.probe_token, "1 1 d -> b 1 d", b=b)

        pooled = probe
        if self.n_layers == 1:
            pooled = self.attn_pool(pooled, context=latents)
        else:
            for attn in self.attn_pool:
                pooled = attn(pooled, context=latents)

        pred_log_fps = self.mlp(pooled).squeeze(-1)
        return pred_log_fps
