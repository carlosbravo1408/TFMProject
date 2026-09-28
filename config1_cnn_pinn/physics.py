from __future__ import annotations

import numpy as np
import torch

MM_PER_M = 1000.0
_M_SINGULAR_TOL = 1e-3


def equivalent_stress_range(load_block, m: float) -> float:
    # (mean dsigma_i^m)^(1/m) reproduces the block-averaged rate, Kong et al. (2020) Eq. 11.
    block = np.asarray(load_block, dtype=float)
    n, s_max, s_min = block[:, 0], block[:, 1], block[:, 2]
    d_sigma = s_max - s_min
    return float((np.sum(n * d_sigma ** m) / np.sum(n)) ** (1.0 / m))


def pooled_log_c(m: float, specimens=None) -> float:
    from physics_calibration.data import CALIBRATION_SPECIMENS, load_curve
    nombres = specimens if specimens is not None else CALIBRATION_SPECIMENS
    # Re-identified at m: C and m move together along the log-log ridge.
    valores = []
    for s in nombres:
        curve = load_curve(s)
        valores.append(fit_log_c(curve.cycles, curve.crack_mm, m,
                                 equivalent_stress_range(curve.load_block, m)))
    return float(np.mean(valores))


def growth_constant(d_sigma_mpa: float, C, m: float):
    return C * (d_sigma_mpa * np.sqrt(np.pi)) ** m


def propagate_mm(a0_mm, delta_cycles, C, m: float, d_sigma_mpa: float, a_max_mm: float = 20.0):
    is_torch = torch.is_tensor(a0_mm) or torch.is_tensor(C)
    exp_fn, pow_fn, clamp = (torch.exp, torch.pow, torch.clamp) if is_torch else (
        np.exp, np.power, np.clip)

    a0_m = a0_mm / MM_PER_M
    k = growth_constant(d_sigma_mpa, C, m)
    # Closed form of da/dN = C dK^m with Y = 1; a_max_mm (20 mm, ~3x the longest
    # measured crack) keeps diverging trajectories finite.
    if abs(m - 2.0) < _M_SINGULAR_TOL:
        a_m = a0_m * exp_fn(k * delta_cycles)
    else:
        p = 1.0 - m / 2.0
        base = pow_fn(a0_m, p) + p * k * delta_cycles
        floor = 1e-9
        base = clamp(base, floor, None) if is_torch else np.maximum(base, floor)
        a_m = pow_fn(base, 1.0 / p)
    a_mm = a_m * MM_PER_M
    return clamp(a_mm, 0.0, a_max_mm) if is_torch else np.clip(a_mm, 0.0, a_max_mm)


# Measured and rejected on the training folds (worst fold 48.81 -> 86.23);
# not wired into the submission.
def reconciled_anchor_mm(cycles, estimates_mm, C, m: float, d_sigma_mpa: float,
                         n_grid: int = 20001, max_mm: float = 20.0) -> float:
    cycles = np.asarray(cycles, dtype=float)
    est = np.asarray(estimates_mm, dtype=float)
    if len(est) < 2:
        return float(est[-1])
    n_cut = cycles[-1]
    rejilla = np.linspace(1e-3, max(max_mm, float(est.max()) * 2.0), n_grid)[:, None]
    pred = propagate_mm(rejilla, (cycles - n_cut)[None, :], C, m, d_sigma_mpa,
                        a_max_mm=max_mm)
    sse = np.sum((pred - est[None, :]) ** 2, axis=1)
    return float(rejilla[int(np.argmin(sse)), 0])


def residual(a_from_mm, a_to_mm, delta_cycles, C, m: float, d_sigma_mpa: float):
    # Detached, otherwise both estimates collapse onto one value to zero the residual.
    anchor = a_from_mm.detach() if torch.is_tensor(a_from_mm) else a_from_mm
    predicted = propagate_mm(anchor, delta_cycles, C, m, d_sigma_mpa)
    return a_to_mm - predicted


def fit_log_c(cycles, cracks_mm, m: float, d_sigma_mpa: float,
              bounds=(-13.0, -6.0), grid: int = 4001) -> float:
    cycles = np.asarray(cycles, dtype=float)
    cracks = np.asarray(cracks_mm, dtype=float)
    log_c = np.linspace(*bounds, grid)
    pred = propagate_mm(cracks[0], (cycles - cycles[0])[None, :],
                        (10.0 ** log_c)[:, None], m, d_sigma_mpa)
    sse = np.sum((pred - cracks[None, :]) ** 2, axis=1)
    return float(log_c[int(np.argmin(sse))])
