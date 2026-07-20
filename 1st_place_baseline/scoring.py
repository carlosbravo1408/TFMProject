"""Scoring functions from the 2019 PHM Conference Data Challenge (Section 2.3).

Verified by hand against the paper's own worked examples (Table 5): for
specimen T7 at cycle 44054 (estimated 1.92 mm, true 2.07 mm, previous
estimate 0 mm) this module reproduces S(i) = 25.36; at cycle 47022
(estimated 3.08 mm, true 3.14 mm) it reproduces S(i) = 11.69.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def time_penalty(x_true: float) -> float:
    """Eq. 1: T(i) = 2 + 10*x_i."""
    return 2 + 10 * x_true


def asymmetric_penalty(x_hat: float, x_true: float) -> float:
    """Eq. 2: harsher penalty for underestimation than overestimation."""
    diff = x_hat - x_true
    scale = 0.5 if diff >= 0 else 0.2
    return float(np.exp(abs(diff) / scale) - 1)


def monotonicity_penalty(x_hat: float, x_hat_prev: float | None) -> float:
    """Eq. 3: penalizes a decreasing sequence of estimates/predictions."""
    if x_hat_prev is None:
        return 1.0
    delta = x_hat - x_hat_prev
    return 1 + 10 * abs(delta) if delta < 0 else 1.0


def penalty_score(x_hat: float, x_true: float, x_hat_prev: float | None) -> float:
    """Eq. 4: S(i) = T(i) * A(i) * M(i)."""
    return (
        time_penalty(x_true)
        * asymmetric_penalty(x_hat, x_true)
        * monotonicity_penalty(x_hat, x_hat_prev)
    )


def score_table(cycles: list[float], x_hat: list[float], x_true: list[float]) -> pd.DataFrame:
    """Per-cycle penalty scores for a sequence of estimates/predictions,
    given in chronological order (monotonicity compares consecutive x_hat)."""
    rows = []
    prev = None
    for cycle, hat, true in zip(cycles, x_hat, x_true):
        s = penalty_score(hat, true, prev)
        rows.append(
            {
                "cycle": cycle,
                "estimated": hat,
                "true": true,
                "T": time_penalty(true),
                "A": asymmetric_penalty(hat, true),
                "M": monotonicity_penalty(hat, prev),
                "S": s,
            }
        )
        prev = hat
    return pd.DataFrame(rows)
