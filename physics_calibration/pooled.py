"""Hierarchical (pooled) identification across specimens.

A single specimen cannot identify a growth law. ``C`` and ``m`` are tied by the
log-log ridge ``log10 C ~ log10(da/dN) - m*log10(dK)``, and each PHM 2019
specimen only spans dK ~ 5-15 MPa*sqrt(m) in 5-7 measurements: fitting them
freely gives an excellent RMSE at an ``m`` that is essentially arbitrary, which
is useless as a physics prior for the PINN.

The pooled fit resolves this the way fracture mechanics does: the *exponents*
(``m``, ``gamma``, ``Kc``) are material constants shared by every specimen of
the same alloy and joint geometry, while the *coefficient* (``C`` / ``C0``)
absorbs specimen-to-specimen scatter (rivet fit-up, local geometry, crack
initiation site). One search vector therefore carries

    [ shared exponents ... , log10 C_1, log10 C_2, ..., log10 C_S ]

and the objective is the mean per-specimen loss. This is exactly the structure
the TFM's PINN needs: the shared exponents become fixed physics, and the
per-specimen ``C`` becomes the single scalar the 1D-CNN + attention encoder has
to regress from the Lamb-wave history.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import CrackCurve
from .models import GrowthLaw, predict_mm
from .objectives import phm_penalty, _LARGE


@dataclass(frozen=True)
class PooledProblem:
    law: GrowthLaw
    curves: tuple[CrackCurve, ...]
    loss: str = "rmse"
    max_step: float = 200.0

    @property
    def shared_names(self) -> tuple[str, ...]:
        """Law parameters other than the leading coefficient."""
        return self.law.param_names[1:]

    @property
    def coefficient_name(self) -> str:
        return self.law.param_names[0]

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        coeff_bounds, *shared_bounds = self.law.bounds
        return tuple(shared_bounds) + tuple(coeff_bounds for _ in self.curves)

    def split(self, x: np.ndarray):
        """(n, d) search matrix -> (shared (n, k), per-specimen log10 C (n, S))."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        k = len(self.shared_names)
        return x[:, :k], x[:, k:]

    def specimen_vectors(self, x: np.ndarray, i: int) -> np.ndarray:
        """Rebuild the law's own search vector for specimen ``i``."""
        shared, coeffs = self.split(x)
        return np.column_stack([coeffs[:, i], shared])

    def per_specimen_losses(self, x: np.ndarray) -> np.ndarray:
        """(n_candidates, n_specimens) losses under ``self.loss``."""
        out = np.empty((len(np.atleast_2d(x)), len(self.curves)))
        for i, curve in enumerate(self.curves):
            xi = self.specimen_vectors(x, i)
            pred = predict_mm(
                self.law, xi, curve.a0_mm, curve.delta_cycles, curve.load_block, self.max_step
            ).T
            truth = curve.crack_mm[None, :]
            if self.loss == "rmse":
                val = np.sqrt(np.mean((pred - truth) ** 2, axis=1))
            elif self.loss == "kong":
                w = (np.arange(1, truth.shape[1] + 1) ** 2)[None, :]
                val = np.mean(w * (pred - truth) ** 2, axis=1)
            elif self.loss == "phm":
                val = phm_penalty(pred, curve.crack_mm, curve.final_crack_mm)
            else:
                raise ValueError(f"unknown loss {self.loss!r}")
            out[:, i] = np.where(np.isfinite(val), val, _LARGE)
        return out

    def objective(self):
        def f(x):
            return self.per_specimen_losses(x).mean(axis=1)

        return f

    def describe(self, x: np.ndarray) -> dict:
        """Named physical parameters for one pooled solution."""
        shared, coeffs = self.split(np.atleast_2d(x))
        result = {"shared": {}, "coefficient": {}}
        for j, name in enumerate(self.shared_names):
            v = float(shared[0, j])
            result["shared"][name] = 10.0 ** v if name in self.law.log_params else v
        for i, curve in enumerate(self.curves):
            result["coefficient"][curve.name] = float(10.0 ** coeffs[0, i])
        result["coefficient_name"] = self.coefficient_name
        return result


@dataclass(frozen=True)
class ProfiledPooledProblem:
    """The pooled problem with the per-specimen coefficients projected out.

    :class:`PooledProblem` searches ``1 + S`` variables at once, and its
    landscape turns out to be multimodal enough that independent runs of the
    same metaheuristic land on visibly different optima. But the problem is
    *separable*: for any fixed set of exponents, each specimen's coefficient
    can be found on its own, and the loss is unimodal in ``log10 C`` (crack
    length is monotone in the coefficient). Projecting the coefficients out by
    a vectorised grid search — the classical variable-projection trick —
    leaves the metaheuristics with only the 1-3 shared exponents to explore,
    which is where they converge reliably and where comparing them is
    meaningful.
    """

    law: GrowthLaw
    curves: tuple[CrackCurve, ...]
    loss: str = "rmse"
    max_step: float = 200.0
    grid: int = 48
    refinements: int = 3

    @property
    def shared_names(self) -> tuple[str, ...]:
        return self.law.param_names[1:]

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple(self.law.bounds[1:])

    def _loss(self, pred: np.ndarray, curve: CrackCurve) -> np.ndarray:
        truth = curve.crack_mm[None, :]
        if self.loss == "rmse":
            val = np.sqrt(np.mean((pred - truth) ** 2, axis=1))
        elif self.loss == "kong":
            w = (np.arange(1, truth.shape[1] + 1) ** 2)[None, :]
            val = np.mean(w * (pred - truth) ** 2, axis=1)
        elif self.loss == "phm":
            val = phm_penalty(pred, curve.crack_mm, curve.final_crack_mm)
        else:
            raise ValueError(f"unknown loss {self.loss!r}")
        return np.where(np.isfinite(val), val, _LARGE)

    def _best_coefficient(self, shared: np.ndarray, curve: CrackCurve):
        """(n,) best log10 C and (n,) loss for each row of ``shared``."""
        n = len(shared)
        lo, hi = self.law.bounds[0]
        centre = np.full(n, 0.5 * (lo + hi))
        half = 0.5 * (hi - lo)
        best_c = centre
        best_v = np.full(n, np.inf)
        for _ in range(self.refinements):
            offsets = np.linspace(-half, half, self.grid)
            cand = np.clip(centre[:, None] + offsets[None, :], lo, hi)  # (n, G)
            x = np.column_stack(
                [cand.reshape(-1), np.repeat(shared, self.grid, axis=0)]
            )
            pred = predict_mm(
                self.law, x, curve.a0_mm, curve.delta_cycles, curve.load_block, self.max_step
            ).T
            val = self._loss(pred, curve).reshape(n, self.grid)
            j = np.argmin(val, axis=1)
            rows = np.arange(n)
            improved = val[rows, j] < best_v
            best_v = np.where(improved, val[rows, j], best_v)
            best_c = np.where(improved, cand[rows, j], best_c)
            centre = best_c
            half = 2.0 * (2 * half / (self.grid - 1))  # zoom to +/- 2 grid steps
        return best_c, best_v

    def per_specimen(self, shared: np.ndarray):
        """(n, S) losses and (n, S) log10 coefficients."""
        shared = np.atleast_2d(np.asarray(shared, dtype=float))
        n, s = len(shared), len(self.curves)
        losses = np.empty((n, s))
        coeffs = np.empty((n, s))
        for i, curve in enumerate(self.curves):
            coeffs[:, i], losses[:, i] = self._best_coefficient(shared, curve)
        return losses, coeffs

    def objective(self):
        def f(shared):
            return self.per_specimen(shared)[0].mean(axis=1)

        return f

    def describe(self, shared: np.ndarray) -> dict:
        shared = np.atleast_2d(np.asarray(shared, dtype=float))
        losses, coeffs = self.per_specimen(shared)
        out = {"shared": {}, "coefficient": {}, "coefficient_name": self.law.param_names[0]}
        for j, name in enumerate(self.shared_names):
            v = float(shared[0, j])
            out["shared"][name] = 10.0 ** v if name in self.law.log_params else v
        for i, curve in enumerate(self.curves):
            out["coefficient"][curve.name] = float(10.0 ** coeffs[0, i])
        out["per_specimen_loss"] = {
            c.name: float(losses[0, i]) for i, c in enumerate(self.curves)
        }
        out["log10_coefficients"] = {
            c.name: float(coeffs[0, i]) for i, c in enumerate(self.curves)
        }
        return out


def fit_coefficient_only(
    law: GrowthLaw,
    shared: np.ndarray,
    curve: CrackCurve,
    n_points: int | None = None,
    loss: str = "rmse",
    max_step: float = 200.0,
    grid: int = 3001,
) -> tuple[float, np.ndarray]:
    """Identify only the leading coefficient of ``law`` for one specimen,
    holding the shared exponents fixed.

    This is the transfer step: exponents identified on the training specimens
    are frozen as material constants, and the single remaining degree of
    freedom is calibrated on the first ``n_points`` measurements of a new
    specimen — the information a prognosis model actually has before the Lamb
    wave signals stop. A dense 1-D grid search over log10 C is used because
    with one variable it is both exhaustive and cheaper than any heuristic.

    Returns ``(log10_C, predictions_mm_at_all_cycles)``.
    """
    lo, hi = law.bounds[0]
    log_c = np.linspace(lo, hi, grid)
    x = np.column_stack([log_c, np.tile(np.asarray(shared, float), (grid, 1))])
    pred = predict_mm(law, x, curve.a0_mm, curve.delta_cycles, curve.load_block, max_step).T
    k = len(curve.crack_mm) if n_points is None else min(n_points, len(curve.crack_mm))
    truth = curve.crack_mm[None, :k]
    fit = pred[:, :k]
    if loss == "rmse":
        val = np.sqrt(np.mean((fit - truth) ** 2, axis=1))
    elif loss == "phm":
        val = phm_penalty(fit, curve.crack_mm[:k], curve.final_crack_mm)
    else:
        raise ValueError(f"unknown loss {loss!r}")
    best = int(np.argmin(np.where(np.isfinite(val), val, _LARGE)))
    return float(log_c[best]), pred[best]


def profile_exponent(
    problem: PooledProblem,
    optimizer,
    index: int = 0,
    values: np.ndarray | None = None,
    **opt_kwargs,
) -> list[dict]:
    """Profile likelihood over one shared exponent.

    For each fixed value of the exponent (``index`` into ``shared_names``) the
    remaining parameters are re-optimised. The resulting curve shows how flat
    the identification really is — the honest way to report ``m`` when the
    available dK span is only half a decade.
    """
    if values is None:
        lo, hi = problem.law.bounds[index + 1]
        values = np.linspace(lo, hi, 15)
    from .metaheuristics import polish

    base = problem.objective()
    full_bounds = list(problem.bounds)
    out = []
    for v in values:
        bounds = list(full_bounds)
        bounds[index] = (float(v), float(v) + 1e-9)
        res = optimizer(base, bounds, **opt_kwargs)
        x, fun = polish(base, res.x, bounds)
        out.append({"value": float(v), "objective": float(fun), "x": np.asarray(x).tolist()})
    return out
