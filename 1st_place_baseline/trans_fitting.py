"""Crack length prediction via the "trans-fitting" method (Section 3.2).

Pipeline:
  1. Candidate functions are fit to a training specimen's post-initiation
     crack-growth curve a(N) (Section 3.2.1/3.2.2); SSE and DFE rank them.
  2. The fitted curve's parameters become a Bayesian *prior*. Given the
     target specimen's known/estimated crack length(s), a regularized
     (MAP) refit "translocates" the curve to match the target while staying
     close to the trained shape (Section 3.2.3, Eq. 17).
  3. Multiple translocated curves (one per reference training specimen) are
     averaged, and the anchor set grows one predicted cycle at a time
     ("sequential updating", Section 3.2.4); the final prediction for each
     target cycle is the average of every value predicted for it before it
     became an anchor itself.

Implementation notes (not specified verbatim in the paper, chosen for
numerical stability and documented here):
  - Cycle counts (O(1e4-1e5)) are internally rescaled to ``t = N / 1e4``
    before evaluating candidate functions; this is required for the
    Exp1+Gaussian1 form (Eq. 14) whose Gaussian term has *unit* variance
    in ``N`` and would otherwise underflow to exactly zero.
  - Regularization is implemented as extra pseudo-residuals
    ``sqrt(lambda) * (theta - prior)`` appended to the least-squares
    residual vector (standard Tikhonov-via-least_squares trick), solved
    with ``scipy.optimize.least_squares`` (trust-region-reflective, 'trf'),
    matching the paper's stated trust-region algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

CYCLE_SCALE = 1e4


def _t(N: np.ndarray) -> np.ndarray:
    return np.asarray(N, dtype=float) / CYCLE_SCALE


# ---------------------------------------------------------------------------
# Candidate functions (MATLAB Curve Fitting Toolbox naming, as used in the
# paper's Table 4: Poly1, Poly2, Exp1, Exp2, Gaussian1, Gaussian2, Power1,
# Power2, and the custom Exp1+Gaussian1 of Eq. 14). Parameter counts were
# cross-checked against the paper's DFE = L - P values in Table 4 (L=7 for
# both T3 and T4's post-initiation data).
# ---------------------------------------------------------------------------

def _poly1(N, a1, a2):
    return a1 * _t(N) + a2


def _poly2(N, a1, a2, a3):
    t = _t(N)
    return a1 * t**2 + a2 * t + a3


def _exp1(N, a1, a2):
    return a1 * np.exp(a2 * _t(N))


def _exp2(N, a1, a2, a3, a4):
    t = _t(N)
    return a1 * np.exp(a2 * t) + a3 * np.exp(a4 * t)


def _gaussian1(N, a1, a2, a3):
    t = _t(N)
    return a1 * np.exp(-(((t - a2) / a3) ** 2))


def _gaussian2(N, a1, a2, a3, a4, a5, a6):
    return _gaussian1(N, a1, a2, a3) + _gaussian1(N, a4, a5, a6)


def _power1(N, a1, a2):
    return a1 * _t(N) ** a2


def _power2(N, a1, a2, a3):
    return a1 * _t(N) ** a2 + a3


def _exp_gaussian1(N, a1, a2, a3, a4, a5):
    """Eq. 14: theta1*exp(-(N-theta2)^2) + theta3*exp(theta4*N) + theta5."""
    t = _t(N)
    return a1 * np.exp(-((t - a2) ** 2)) + a3 * np.exp(a4 * t) + a5


def _p0_poly1(t, a):
    slope = (a[-1] - a[0]) / (t[-1] - t[0] + 1e-9)
    return np.array([slope, a[0]])


def _p0_poly2(t, a):
    return np.array([0.0, *_p0_poly1(t, a)])


def _p0_exp1(t, a):
    a0 = a[0] if a[0] != 0 else 1.0
    return np.array([a0, 0.1])


def _p0_exp2(t, a):
    a0 = (a[0] if a[0] != 0 else 1.0) / 2
    return np.array([a0, 0.05, a0, 0.5])


def _p0_gaussian1(t, a):
    peak = t[np.argmax(a)]
    width = max((t.max() - t.min()) / 2, 1e-3)
    return np.array([a.max(), peak, width])


def _p0_gaussian2(t, a):
    p1 = _p0_gaussian1(t, a)
    p2 = p1.copy()
    p2[1] = p2[1] + 0.5 * (t.max() - t.min() + 1e-3)
    p2[0] = p2[0] / 2
    p1[0] = p1[0] / 2
    return np.concatenate([p1, p2])


def _p0_power1(t, a):
    return np.array([1.0, 1.0])


def _p0_power2(t, a):
    return np.array([1.0, 1.0, 0.0])


def _p0_exp_gaussian1(t, a):
    mid = t[len(t) // 2]
    a0 = a[0] if a[0] != 0 else 0.1
    return np.array([a.max() - a.min(), mid, a0, 0.1, a0])


@dataclass
class Candidate:
    name: str
    func: Callable
    n_params: int
    p0: Callable


CANDIDATES: dict[str, Candidate] = {
    "Poly1": Candidate("Poly1", _poly1, 2, _p0_poly1),
    "Poly2": Candidate("Poly2", _poly2, 3, _p0_poly2),
    "Exp1": Candidate("Exp1", _exp1, 2, _p0_exp1),
    "Exp2": Candidate("Exp2", _exp2, 4, _p0_exp2),
    "Gaussian1": Candidate("Gaussian1", _gaussian1, 3, _p0_gaussian1),
    "Gaussian2": Candidate("Gaussian2", _gaussian2, 6, _p0_gaussian2),
    "Power1": Candidate("Power1", _power1, 2, _p0_power1),
    "Power2": Candidate("Power2", _power2, 3, _p0_power2),
    "Exp1+Gaussian1": Candidate("Exp1+Gaussian1", _exp_gaussian1, 5, _p0_exp_gaussian1),
}


def fit_regularized(
    func: Callable,
    N: np.ndarray,
    a: np.ndarray,
    n_params: int,
    p0: np.ndarray,
    lam: float = 0.0,
    prior: np.ndarray | None = None,
    max_nfev: int = 20000,
) -> np.ndarray:
    """Regularized nonlinear least squares (Eq. 16 / Eq. 17):
    minimize sum((a_fit(N) - a)^2) + lam * sum((theta - prior)^2).
    """
    prior_vec = np.zeros(n_params) if prior is None else np.asarray(prior, dtype=float)
    sqrt_lam = np.sqrt(lam)

    def residuals(theta):
        data_resid = func(N, *theta) - a
        reg_resid = sqrt_lam * (theta - prior_vec)
        return np.concatenate([data_resid, reg_resid])

    result = least_squares(residuals, p0, method="trf", max_nfev=max_nfev)
    return result.x


def goodness_of_fit_table(N: np.ndarray, a: np.ndarray, lam: float = 1e-4) -> pd.DataFrame:
    """Reproduces the SSE/DFE goodness-of-fit comparison of Table 4."""
    N = np.asarray(N, dtype=float)
    a = np.asarray(a, dtype=float)
    rows = []
    for cand in CANDIDATES.values():
        t = _t(N)
        p0 = cand.p0(t, a)
        try:
            theta = fit_regularized(cand.func, N, a, cand.n_params, p0, lam=lam)
            pred = cand.func(N, *theta)
            sse = float(np.sum((pred - a) ** 2))
        except Exception:
            theta, sse = None, np.inf
        dfe = len(N) - cand.n_params
        rows.append({"candidate": cand.name, "sse": sse, "dfe": dfe, "theta": theta})
    return pd.DataFrame(rows)


def select_candidate(table: pd.DataFrame, min_dfe: int = 2) -> pd.Series:
    """Lowest-SSE candidate among those with DFE >= min_dfe (excludes
    over-fit models such as Gaussian2, matching Section 3.2.1's reasoning)."""
    valid = table[table["dfe"] >= min_dfe]
    return valid.loc[valid["sse"].idxmin()]


class TransFitCurve:
    """One reference training curve (a candidate function fit to one
    training specimen), which can be "translocated" to a target specimen's
    known anchor points via MAP refitting (Section 3.2.3)."""

    def __init__(self, candidate: Candidate, prior_theta: np.ndarray):
        self.candidate = candidate
        self.prior_theta = np.asarray(prior_theta, dtype=float)

    @classmethod
    def fit_from_training(
        cls, candidate: Candidate, N: np.ndarray, a: np.ndarray, lam: float = 1e-4
    ) -> "TransFitCurve":
        t = _t(np.asarray(N, dtype=float))
        p0 = candidate.p0(t, np.asarray(a, dtype=float))
        theta = fit_regularized(candidate.func, N, a, candidate.n_params, p0, lam=lam)
        return cls(candidate, theta)

    def translocate(self, anchors: list[tuple[float, float]], lam: float) -> np.ndarray:
        N = np.array([p[0] for p in anchors], dtype=float)
        a = np.array([p[1] for p in anchors], dtype=float)
        return fit_regularized(
            self.candidate.func,
            N,
            a,
            self.candidate.n_params,
            p0=self.prior_theta,
            lam=lam,
            prior=self.prior_theta,
        )

    def predict(self, theta: np.ndarray, N: np.ndarray) -> np.ndarray:
        return self.candidate.func(np.asarray(N, dtype=float), *theta)


def sequential_trans_fit(
    curves: list[TransFitCurve],
    initial_anchors: list[tuple[float, float]],
    target_cycles: list[float],
    lam: float,
) -> tuple[dict[float, float], list[dict[float, float]]]:
    """Sequential-updating trans-fitting prediction (Section 3.2.4).

    At each iteration, every reference curve is translocated using the
    current anchor set, predictions for all still-unanchored target cycles
    are averaged across curves, the *first* remaining target cycle is then
    "locked in" as a new anchor using that averaged prediction, and the
    process repeats. The final prediction for each target cycle is the
    average of every value predicted for it while it was still unanchored
    -- this reproduces the paper's Tables 7 and 11 exactly.

    Returns ``(final_predictions, iteration_tables)`` where
    ``iteration_tables[k]`` is the k-th average trans-fitting curve's
    predictions (one entry per cycle still remaining at that iteration).
    """
    anchors = list(initial_anchors)
    remaining = list(target_cycles)
    contributions: dict[float, list[float]] = {c: [] for c in target_cycles}
    iteration_tables = []

    while remaining:
        per_curve_preds = np.array(
            [curve.predict(curve.translocate(anchors, lam), remaining) for curve in curves]
        )
        avg_pred = per_curve_preds.mean(axis=0)
        iteration_tables.append(dict(zip(remaining, avg_pred)))
        for cycle, value in zip(remaining, avg_pred):
            contributions[cycle].append(float(value))

        next_cycle, next_value = remaining[0], avg_pred[0]
        anchors.append((next_cycle, next_value))
        remaining = remaining[1:]

    final_predictions = {c: float(np.mean(v)) for c, v in contributions.items()}
    return final_predictions, iteration_tables
