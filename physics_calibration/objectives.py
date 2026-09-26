"""Objective functions for the fracture-parameter identification.

Which loss the identification is run under is itself a design decision with
consequences for the TFM, so three are provided and reported side by side:

``rmse``    plain RMSE on a(N) in mm. Physically neutral; this is the loss
            that answers "which (C, m, gamma) best satisfies the growth law
            for this material", so it is the primary one.
``kong``    the i^2-weighted squared error of Kong et al. (2020), Eq. 12,
            which deliberately over-weights the late, large cracks.
``phm``     the official challenge penalty S_sum = sum_i T(i)*A(i)*M(i) on
            crack lengths normalised by the specimen's final true crack
            length. Asymmetric (underestimation is punished ~2.5x harder) and
            monotonicity-aware, i.e. the metric the TFM must actually beat.

Fitting under ``phm`` and reporting under ``rmse`` (and vice versa) quantifies
how much the challenge's risk asymmetry pulls the identified physics away from
the maximum-likelihood fit — a result worth reporting in its own right.
"""
from __future__ import annotations

import numpy as np

from .data import CrackCurve
from .models import GrowthLaw, predict_mm

_LARGE = 1e12


def _predict(law: GrowthLaw, x: np.ndarray, curve: CrackCurve, max_step: float) -> np.ndarray:
    """(n_candidates, n_points) crack lengths in mm, anchored at the curve's
    first non-zero measurement."""
    pred = predict_mm(law, x, curve.a0_mm, curve.delta_cycles, curve.load_block, max_step)
    return pred.T


def _sanitize(err: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(err), err, _LARGE)


def rmse_objective(law: GrowthLaw, curve: CrackCurve, max_step: float = 200.0):
    """RMSE in mm over the specimen's non-zero measurements."""
    truth = curve.crack_mm[None, :]

    def f(x):
        pred = _predict(law, x, curve, max_step)
        return _sanitize(np.sqrt(np.mean((pred - truth) ** 2, axis=1)))

    return f


def kong_objective(law: GrowthLaw, curve: CrackCurve, max_step: float = 200.0):
    """Kong et al. (2020) Eq. 12: mean of i^2-weighted squared errors."""
    truth = curve.crack_mm[None, :]
    w = (np.arange(1, len(curve.crack_mm) + 1) ** 2)[None, :]

    def f(x):
        pred = _predict(law, x, curve, max_step)
        return _sanitize(np.mean(w * (pred - truth) ** 2, axis=1))

    return f


def phm_penalty(pred_mm: np.ndarray, true_mm: np.ndarray, norm_mm: float) -> np.ndarray:
    """Vectorised official penalty S_sum over a (n_candidates, n_points) batch.

    T(i) = 2 + 10*x_i;  A(i) = exp(|dx|/0.5)-1 if over-estimating else
    exp(|dx|/0.2)-1;  M(i) = 1 + 10*|dx_hat| when the estimate decreases.
    """
    x_true = np.asarray(true_mm, dtype=float)[None, :] / norm_mm
    x_hat = np.asarray(pred_mm, dtype=float) / norm_mm
    diff = x_hat - x_true
    scale = np.where(diff >= 0, 0.5, 0.2)
    t = 2.0 + 10.0 * x_true
    a = np.exp(np.clip(np.abs(diff) / scale, 0.0, 50.0)) - 1.0
    step = np.diff(x_hat, axis=1)
    m = np.ones_like(x_hat)
    m[:, 1:] = np.where(step < 0, 1.0 + 10.0 * np.abs(step), 1.0)
    return np.sum(t * a * m, axis=1)


def phm_objective(law: GrowthLaw, curve: CrackCurve, max_step: float = 200.0):
    """Official challenge penalty, normalised by the specimen's final crack."""

    def f(x):
        pred = _predict(law, x, curve, max_step)
        return _sanitize(phm_penalty(pred, curve.crack_mm, curve.final_crack_mm))

    return f


OBJECTIVES = {"rmse": rmse_objective, "kong": kong_objective, "phm": phm_objective}


def evaluate_all(law: GrowthLaw, x: np.ndarray, curve: CrackCurve, max_step: float = 200.0) -> dict:
    """Report every metric for a single identified parameter vector."""
    pred = _predict(law, np.atleast_2d(x), curve, max_step)
    truth = curve.crack_mm[None, :]
    return {
        "rmse_mm": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mae_mm": float(np.mean(np.abs(pred - truth))),
        "max_abs_err_mm": float(np.max(np.abs(pred - truth))),
        "kong": float(np.mean((np.arange(1, truth.shape[1] + 1) ** 2) * (pred - truth) ** 2)),
        "phm_penalty": float(phm_penalty(pred, curve.crack_mm, curve.final_crack_mm)[0]),
        "pred_mm": pred[0].tolist(),
        "true_mm": curve.crack_mm.tolist(),
    }
