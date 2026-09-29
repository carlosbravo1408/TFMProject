from __future__ import annotations

import numpy as np

from .data import CrackCurve
from .models import GrowthLaw, predict_mm

_LARGE = 1e12


def _predict(law: GrowthLaw, x: np.ndarray, curve: CrackCurve, max_step: float) -> np.ndarray:
    pred = predict_mm(law, x, curve.a0_mm, curve.delta_cycles, curve.load_block, max_step)
    return pred.T


def _sanitize(err: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(err), err, _LARGE)


def rmse_objective(law: GrowthLaw, curve: CrackCurve, max_step: float = 200.0):
    truth = curve.crack_mm[None, :]

    def f(x):
        pred = _predict(law, x, curve, max_step)
        return _sanitize(np.sqrt(np.mean((pred - truth) ** 2, axis=1)))

    return f


def kong_objective(law: GrowthLaw, curve: CrackCurve, max_step: float = 200.0):
    truth = curve.crack_mm[None, :]
    w = (np.arange(1, len(curve.crack_mm) + 1) ** 2)[None, :]

    def f(x):
        pred = _predict(law, x, curve, max_step)
        return _sanitize(np.mean(w * (pred - truth) ** 2, axis=1))

    return f


def phm_penalty(pred_mm: np.ndarray, true_mm: np.ndarray, norm_mm: float) -> np.ndarray:
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
    def f(x):
        pred = _predict(law, x, curve, max_step)
        return _sanitize(phm_penalty(pred, curve.crack_mm, curve.final_crack_mm))

    return f


OBJECTIVES = {"rmse": rmse_objective, "kong": kong_objective, "phm": phm_objective}


def evaluate_all(law: GrowthLaw, x: np.ndarray, curve: CrackCurve, max_step: float = 200.0) -> dict:
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
