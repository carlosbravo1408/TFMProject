from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config1_cnn_pinn.model import CnnPinn


class CycleAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(seq, seq, seq, need_weights=False)
        h = self.norm1(seq + h)
        h = self.norm2(h + self.ff(h))
        resumen, pesos = nn.functional.multi_head_attention_forward(
            self.query.transpose(0, 1), h.transpose(0, 1), h.transpose(0, 1),
            h.shape[-1], 1,
            None, None, None, None, False, 0.0,
            torch.eye(h.shape[-1], device=h.device), None,
            use_separate_proj_weight=True,
            q_proj_weight=torch.eye(h.shape[-1], device=h.device),
            k_proj_weight=torch.eye(h.shape[-1], device=h.device),
            v_proj_weight=torch.eye(h.shape[-1], device=h.device),
            training=self.training,
        )
        self.last_weights = pesos.detach()
        return resumen.transpose(0, 1).squeeze(1)


class CnnAttentionPinn(CnnPinn):
    def __init__(self, *args, n_heads: int = 2, attn_dropout: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.encoder.latent_dim
        # Positions are cycle gaps, not indices: the sequence is not uniform in cycles.
        self.position = nn.Linear(2, d)
        self.attention = CycleAttention(d, n_heads, attn_dropout)
        nn.init.zeros_(self.position.weight)
        nn.init.zeros_(self.position.bias)

    def log_c(self, z, specimen_index, n_specimens, cycles=None, log_dn=None):
        if cycles is None:
            # Without sequence context it behaves as Configuration 1.
            return super().log_c(z, specimen_index, n_specimens)

        salidas = []
        for s in range(n_specimens):
            mask = specimen_index == s
            zs, cs, ds = z[mask], cycles[mask], log_dn[mask]
            # One token per cycle: the two repetitions are near-identical.
            unicos, inverso = torch.unique(cs, return_inverse=True)
            tokens = torch.zeros(len(unicos), zs.shape[1], device=z.device, dtype=z.dtype)
            cuenta = torch.zeros(len(unicos), device=z.device, dtype=z.dtype)
            tokens.index_add_(0, inverso, zs)
            cuenta.index_add_(0, inverso, torch.ones_like(inverso, dtype=z.dtype))
            tokens = tokens / cuenta.clamp(min=1.0).unsqueeze(-1)

            hueco = torch.zeros(len(unicos), device=z.device, dtype=z.dtype)
            hueco.index_add_(0, inverso, ds)
            hueco = hueco / cuenta.clamp(min=1.0)
            avance = torch.linspace(0.0, 1.0, len(unicos), device=z.device, dtype=z.dtype)
            tokens = tokens + self.position(torch.stack([hueco, avance], dim=-1))

            resumen = self.attention(tokens.unsqueeze(0))
            salidas.append(self.coefficient_head(resumen).squeeze(-1))
        u = torch.cat(salidas)
        return self.log_c_prior + self.log_c_range * torch.tanh(u)
