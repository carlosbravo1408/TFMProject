from __future__ import annotations

import torch
import torch.nn.functional as F

_OVER, _UNDER = 0.5, 0.2
_MAX_EXPONENT = 6.0


def asymmetric_penalty_loss(pred_mm, true_mm, norm_mm, weight=None):
    x_true = true_mm / norm_mm
    x_hat = pred_mm / norm_mm
    diff = x_hat - x_true
    scale = torch.where(diff >= 0, torch.full_like(diff, _OVER), torch.full_like(diff, _UNDER))
    # Clamped: early in training exp(|dx| / 0.2) overflows.
    a = torch.exp(torch.clamp(diff.abs() / scale, max=_MAX_EXPONENT)) - 1.0
    w = (2.0 + 10.0 * x_true) if weight is None else weight
    return (w * a).mean()


def data_loss(pred_mm, true_mm, norm_mm, kind: str = "asym", weight=None):
    if kind == "mse":
        if weight is None:
            return F.mse_loss(pred_mm, true_mm)
        return (weight * (pred_mm - true_mm) ** 2).mean()
    if kind == "asym":
        return asymmetric_penalty_loss(pred_mm, true_mm, norm_mm, weight)
    raise ValueError(f"unknown data loss {kind!r}")


def sample_weights(crack_mm, cycles, norm_mm, kind: str = "challenge",
                   n_anchor: int = 2, anchor_weight: float = 6.0):
    # The challenge's T(i) down-weights the early measurements, the only ones the
    # network estimates under the protocol and the anchors of the extrapolation.
    x_true = crack_mm / norm_mm
    if kind == "challenge":
        return 2.0 + 10.0 * x_true
    if kind not in ("flat", "anchor"):
        raise ValueError(f"unknown weighting {kind!r}")
    w = torch.ones_like(x_true)
    if kind == "anchor":
        nz = crack_mm > 0
        if bool(nz.any()):
            anclas = sorted({float(c) for c in cycles[nz]})[:n_anchor]
            for c in anclas:
                w = torch.where(cycles == c, torch.full_like(w, anchor_weight), w)
    return w


def physics_loss(residual_mm):
    if residual_mm.numel() == 0:
        return residual_mm.new_zeros(())
    # Huber: consecutive growth-rate intervals of the optical labels jump by 3-5x.
    return F.huber_loss(residual_mm, torch.zeros_like(residual_mm), delta=0.5)


def monotonicity_loss(pred_mm, order_index, specimen_index):
    if pred_mm.numel() < 2:
        return pred_mm.new_zeros(())
    order = torch.argsort(specimen_index * 1_000_000 + order_index)
    p = pred_mm[order]
    same = specimen_index[order][1:] == specimen_index[order][:-1]
    if not bool(same.any()):
        return pred_mm.new_zeros(())
    drop = F.relu(p[:-1][same] - p[1:][same])
    return (drop ** 2).mean()
