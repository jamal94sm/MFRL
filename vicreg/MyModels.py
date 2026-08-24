"""MyModels.py -- VICReg encoder/expander for the classification-task setup.

Same trunk recipe as MFRL's Context_Encoder/Target_Encoder (patch conv,
2D sincos pos-embed, pre-norm TransformerEncoder, same
num_heads = max(4, embed_dim//32) / depth = min(6, embed_dim//64+2)
auto-sizing) so encoder capacity matches 1:1. No masking (VICReg always
sees the full image), no EMA target (VICReg doesn't use one).
"""

import numpy as np
import torch
import torch.nn as nn


def get_1d_sincos_pos_embed(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class Encoder(nn.Module):
    def __init__(self, image_size, num_patches, embed_dim,
                 depth=None, num_heads=None, mlp_ratio=4.0):
        super().__init__()
        H, W = image_size
        patch_h = H // num_patches
        patch_w = W // num_patches

        if num_heads is None:
            num_heads = max(4, embed_dim // 32)
        if depth is None:
            depth = min(6, embed_dim // 64 + 2)

        self.proj = nn.Conv2d(3, embed_dim,
                               kernel_size=(patch_h, patch_w),
                               stride=(patch_h, patch_w))
        pos = get_2d_sincos_pos_embed(embed_dim, num_patches)
        self.pos_embed = nn.Parameter(
            torch.tensor(pos).float().unsqueeze(0), requires_grad=False)
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        z = self.proj(x).flatten(2).transpose(1, 2)
        z = z + self.pos_embed
        z = self.encoder(z)
        z = self.norm(z)
        return z


class Expander(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FeatureExtractor(nn.Module):
    """forward(x) -> [B, embed_dim]. Matches the contract MFRL's own
    _extract_eval_features / periodic_eval expect (full-image mean pool)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x).mean(dim=1)