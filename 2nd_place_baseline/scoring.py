"""Scoring of the 2019 PHM Conference Data Challenge (Section 2.3).

Three penalty functions are combined per cycle (Eqs. 1-4):
    T(i) = 2 + 10*x_i                                   (time penalty)
    A(i) = exp(|x̃_i - x_i| / 0.5) - 1   if x̃_i >= x_i   (asymmetric penalty)
           exp(|x̃_i - x_i| / 0.2) - 1   if x̃_i <  x_i
    M(i) = 1 + 10*|x̃_i - x̃_{i-1}|      if decreasing    (monotonicity penalty)
    S(i) = T(i) * A(i) * M(i),      S_sum = sum_i S(i)

IMPORTANT: as in the official scoring spreadsheet distributed by the PHM
Society (and as required to reproduce the paper's penalty score of 7.63),
the crack lengths entering these formulas are NORMALIZED by the final (last
cycle) true crack length of the corresponding specimen. The monotonicity
constant m = 10 matches Eq. 3 (the original spreadsheet's m = 100 was
corrected by the organizers).

RMSE (used for Table 3 of the paper) is computed on the raw crack lengths in
mm over all reported cycles of a specimen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def time_penalty(x_true_norm: float) -> float:
    """Eq. 1 on normalized crack length."""
    return 2 + 10 * x_true_norm


def asymmetric_penalty(x_hat_norm: float, x_true_norm: float) -> float:
    """Eq. 2: underestimation is penalized more harshly than overestimation."""
    diff = x_hat_norm - x_true_norm
    scale = 0.5 if diff >= 0 else 0.2
    return float(np.exp(abs(diff) / scale) - 1)


def monotonicity_penalty(x_hat_norm: float, x_hat_prev_norm: float | None) -> float:
    """Eq. 3 (m = 10): penalizes decreasing estimate sequences."""
    if x_hat_prev_norm is None:
        return 1.0
    delta = x_hat_norm - x_hat_prev_norm
    return 1 + 10 * abs(delta) if delta < 0 else 1.0


def score_table(
    cycles, x_hat_mm, x_true_mm, final_crack_mm: float | None = None
) -> pd.DataFrame:
    """Per-cycle penalty scores for one specimen's chronological sequence of
    estimates/predictions. ``final_crack_mm`` (the normalizing true crack
    length) defaults to the last true value."""
    x_hat_mm = np.asarray(x_hat_mm, dtype=float)
    x_true_mm = np.asarray(x_true_mm, dtype=float)
    norm = float(final_crack_mm if final_crack_mm is not None else x_true_mm[-1])
    rows, prev = [], None
    for cycle, hat, true in zip(cycles, x_hat_mm / norm, x_true_mm / norm):
        t, a, m = time_penalty(true), asymmetric_penalty(hat, true), monotonicity_penalty(hat, prev)
        rows.append(
            {"cycle": cycle, "estimated_norm": hat, "true_norm": true,
             "T": t, "A": a, "M": m, "S": t * a * m}
        )
        prev = hat
    return pd.DataFrame(rows)


def penalty_score(cycles, x_hat_mm, x_true_mm, final_crack_mm: float | None = None) -> float:
    """S_sum (Eq. 4) for one specimen."""
    return float(score_table(cycles, x_hat_mm, x_true_mm, final_crack_mm)["S"].sum())


def rmse_mm(x_hat_mm, x_true_mm) -> float:
    x_hat_mm, x_true_mm = np.asarray(x_hat_mm, float), np.asarray(x_true_mm, float)
    return float(np.sqrt(np.mean((x_hat_mm - x_true_mm) ** 2)))
