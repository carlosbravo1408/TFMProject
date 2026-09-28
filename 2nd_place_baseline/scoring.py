from __future__ import annotations

import numpy as np
import pandas as pd


def time_penalty(x_true_norm: float) -> float:
    return 2 + 10 * x_true_norm


def asymmetric_penalty(x_hat_norm: float, x_true_norm: float) -> float:
    diff = x_hat_norm - x_true_norm
    scale = 0.5 if diff >= 0 else 0.2
    return float(np.exp(abs(diff) / scale) - 1)


def monotonicity_penalty(x_hat_norm: float, x_hat_prev_norm: float | None) -> float:
    if x_hat_prev_norm is None:
        return 1.0
    delta = x_hat_norm - x_hat_prev_norm
    return 1 + 10 * abs(delta) if delta < 0 else 1.0


def score_table(
    cycles, x_hat_mm, x_true_mm, final_crack_mm: float | None = None
) -> pd.DataFrame:
    x_hat_mm = np.asarray(x_hat_mm, dtype=float)
    x_true_mm = np.asarray(x_true_mm, dtype=float)
    # Normalised by the final true crack, as in the official scoring spreadsheet.
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
    return float(score_table(cycles, x_hat_mm, x_true_mm, final_crack_mm)["S"].sum())


def rmse_mm(x_hat_mm, x_true_mm) -> float:
    x_hat_mm, x_true_mm = np.asarray(x_hat_mm, float), np.asarray(x_true_mm, float)
    return float(np.sqrt(np.mean((x_hat_mm - x_true_mm) ** 2)))
