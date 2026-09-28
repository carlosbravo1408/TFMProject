from __future__ import annotations

import numpy as np
import pandas as pd


def time_penalty(x_true: float) -> float:
    return 2 + 10 * x_true


def asymmetric_penalty(x_hat: float, x_true: float) -> float:
    diff = x_hat - x_true
    scale = 0.5 if diff >= 0 else 0.2
    return float(np.exp(abs(diff) / scale) - 1)


def monotonicity_penalty(x_hat: float, x_hat_prev: float | None) -> float:
    if x_hat_prev is None:
        return 1.0
    delta = x_hat - x_hat_prev
    return 1 + 10 * abs(delta) if delta < 0 else 1.0


def penalty_score(x_hat: float, x_true: float, x_hat_prev: float | None) -> float:
    return (
        time_penalty(x_true)
        * asymmetric_penalty(x_hat, x_true)
        * monotonicity_penalty(x_hat, x_hat_prev)
    )


def score_table(cycles: list[float], x_hat: list[float], x_true: list[float]) -> pd.DataFrame:
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
