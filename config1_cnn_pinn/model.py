from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.norm = nn.BatchNorm1d(c_out)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


class CrackEncoder(nn.Module):
    def __init__(self, channels=(8, 16, 24, 32), latent: int = 32, dropout: float = 0.3):
        super().__init__()
        kernels = (15, 9, 7, 5)
        strides = (4, 4, 4, 2)
        blocks, c_in = [], 2
        for c_out, k, s in zip(channels, kernels, strides):
            blocks.append(ConvBlock(c_in, c_out, k, s))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        # Mean pooling tracks overall attenuation; max pooling the strongest scattered echo.
        self.project = nn.Sequential(
            nn.Linear(2 * c_in, latent), nn.GELU(), nn.Dropout(dropout)
        )
        self.latent_dim = latent

    def forward(self, x):
        h = self.blocks(x)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.project(pooled)


class CnnPinn(nn.Module):
    def __init__(
        self,
        log_c_prior: float,
        log_c_range: float = 1.0,
        channels=(8, 16, 24, 32),
        latent: int = 32,
        dropout: float = 0.3,
        n_features: int = 13,
    ):
        super().__init__()
        self.encoder = CrackEncoder(channels, latent, dropout)
        self.n_features = n_features
        # Filled from each run's training specimens so no T7/T8 statistics leak in.
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_std", torch.ones(n_features))
        # The coefficient head reads only the latent: C is a property of the specimen.
        self.rate_head = nn.Linear(latent + n_features + 2, 1)
        self.initial_head = nn.Linear(latent, 1)
        self.coefficient_head = nn.Linear(latent, 1)
        self.register_buffer("log_c_prior", torch.tensor(float(log_c_prior)))
        self.log_c_range = float(log_c_range)
        # softplus(-1) ~ 0.31 mm, the typical growth between consecutive measurements.
        nn.init.zeros_(self.rate_head.weight)
        nn.init.constant_(self.rate_head.bias, -1.0)
        nn.init.zeros_(self.initial_head.weight)
        nn.init.constant_(self.initial_head.bias, 0.5)
        nn.init.zeros_(self.coefficient_head.weight)
        nn.init.zeros_(self.coefficient_head.bias)

    def forward(self, x, feats, a_prev, log_dn, dn):
        return self.crack_mm(self.encoder(x), feats, a_prev, log_dn, dn)

    def crack_mm(self, z, feats, a_prev, log_dn, dn, cycles_scale: float = 1000.0):
        feats_std = (feats - self.feature_mean) / self.feature_std.clamp(min=1e-6)
        h = torch.cat([z, feats_std, a_prev.unsqueeze(-1), log_dn.unsqueeze(-1)], dim=-1)
        # Incremental with dN as an input: an absolute output stalls (LOSO 1.66 vs 0.97 mm)
        # and rate * dN amplifies errors by dN / 1000, up to 36x.
        incremental = a_prev + F.softplus(self.rate_head(h)).squeeze(-1)
        absoluta = F.softplus(self.initial_head(z)).squeeze(-1)
        return torch.where(dn > 0, incremental, absoluta)

    def log_c(self, z, specimen_index: torch.Tensor, n_specimens: int):
        pooled = torch.zeros(n_specimens, z.shape[1], device=z.device, dtype=z.dtype)
        counts = torch.zeros(n_specimens, device=z.device, dtype=z.dtype)
        pooled.index_add_(0, specimen_index, z)
        counts.index_add_(0, specimen_index, torch.ones_like(specimen_index, dtype=z.dtype))
        pooled = pooled / counts.clamp(min=1.0).unsqueeze(-1)
        u = self.coefficient_head(pooled).squeeze(-1)
        return self.log_c_prior + self.log_c_range * torch.tanh(u)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
