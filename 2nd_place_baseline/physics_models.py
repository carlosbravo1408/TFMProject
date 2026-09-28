from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit, differential_evolution


def double_exponential(n: np.ndarray, a0: float, a1: float, b0: float, b1: float) -> np.ndarray:
    return a0 * np.exp(a1 * n) + b0 * np.exp(b1 * n)


@dataclass
class ExponentialModel:
    specimen: str
    params: np.ndarray

    def __call__(self, n) -> np.ndarray:
        return double_exponential(np.asarray(n, dtype=float), *self.params)


def observed_rate_bound(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    # A looser bound lets T1's fit park a tiny amplitude on a rate pinned at the
    # bound, climbing from 4.28 to 8.44 mm when extrapolated.
    rates = []
    for cycles, cracks in curves.values():
        order = np.argsort(cycles)
        n, a = np.asarray(cycles, float)[order], np.asarray(cracks, float)[order]
        keep = a > 0
        if keep.sum() < 2:
            continue
        rates.extend(np.diff(np.log(a[keep])) / np.diff(n[keep]))
    return float(np.max(rates)) if rates else 1.5e-3


def fit_double_exponential(
    specimen: str, cycles: np.ndarray, cracks: np.ndarray, rate_ub: float = 1.5e-3
) -> ExponentialModel:
    cycles = cycles.astype(float)
    cracks = cracks.astype(float)
    bounds = ([0.0, 0.0, 0.0, 0.0], [20.0, rate_ub, 20.0, rate_ub])
    rng = np.random.default_rng(0)
    best_params, best_sse = None, np.inf
    # Multi-start: single-start fits stall in poor local minima.
    for _ in range(60):
        p0 = [rng.uniform(0, 3), rng.uniform(0, rate_ub / 5), rng.uniform(0, 0.05),
              rng.uniform(rate_ub / 8, rate_ub)]
        try:
            params, _ = curve_fit(
                double_exponential, cycles, cracks, p0=p0, bounds=bounds, maxfev=20000
            )
        except RuntimeError:
            continue
        sse = float(np.sum((double_exponential(cycles, *params) - cracks) ** 2))
        if sse < best_sse - 1e-12:
            best_params, best_sse = params, sse
    return ExponentialModel(specimen, best_params)


class EnsemblePrognostics:
    # sigma is unpublished; Figure 7 plots a PDF peak of 0.4, i.e. sigma = 1 mm.
    def __init__(self, models: list[ExponentialModel], sigma_mm: float = 1.0) -> None:
        self.models = models
        self.sigma_mm = sigma_mm

    def weights(self, current_cycle: float, current_crack: float) -> np.ndarray:
        means = np.array([m(current_cycle) for m in self.models])
        pdf = np.exp(-0.5 * ((current_crack - means) / self.sigma_mm) ** 2) / (
            self.sigma_mm * np.sqrt(2 * np.pi)
        )
        return pdf / pdf.sum()

    def predict(
        self, start_cycle: float, start_crack: float, target_cycles: np.ndarray
    ) -> np.ndarray:
        preds = []
        current_cycle, current_crack = float(start_cycle), float(start_crack)
        for next_cycle in np.asarray(target_cycles, dtype=float):
            w = self.weights(current_cycle, current_crack)
            crack_next = float(w @ np.array([m(next_cycle) for m in self.models]))
            preds.append(crack_next)
            current_cycle, current_crack = next_cycle, crack_next
        return np.asarray(preds)


# (cycles, S_max MPa, S_min MPa), Table 2.
VARIABLE_LOADING_BLOCK = (
    (500, 90.0, 4.77),
    (500, 100.21, 4.77),
)


class WalkerModel:
    def __init__(self, block=VARIABLE_LOADING_BLOCK) -> None:
        self.block = block
        self.n_block = sum(n for n, _, _ in block)

    def block_increment_mm(self, a_mm, c0, gamma, m):
        a_m = np.asarray(a_mm, dtype=float) * 1e-3
        da_m = 0.0
        for n_i, s_max, s_min in self.block:
            r = s_min / s_max
            k_max = s_max * np.sqrt(np.pi * a_m)
            da_m = da_m + n_i * c0 * (k_max * (1.0 - r) ** gamma) ** m
        return da_m * 1e3

    def integrate(self, a0_mm, target_cycles, c0, gamma, m):
        a = np.array(a0_mm, dtype=float, ndmin=1).copy()
        c0 = np.asarray(c0, dtype=float)
        gamma = np.asarray(gamma, dtype=float)
        m = np.asarray(m, dtype=float)
        targets = np.asarray(target_cycles, dtype=float)
        out = np.zeros((len(targets),) + a.shape)
        cycle = 0.0
        for it, target in enumerate(targets):
            while cycle < target:
                step = min(self.n_block, target - cycle)
                rate = self.block_increment_mm(a, c0, gamma, m) / self.n_block
                a = a + rate * step
                cycle += step
            out[it] = a
        return out


@dataclass
class WalkerFitResult:
    c0: float
    gamma: float
    m: float
    sigma_fc: float
    objective: float


class WalkerMonteCarlo:
    # C0, gamma, m, sigma_FC
    BOUNDS = [(1e-12, 1e-8), (0.0, 1.0), (2.0, 4.0), (-0.5, 0.5)]

    def __init__(self, fit_cycles, fit_cracks, n_models: int = 100, seed: int = 0) -> None:
        self.model = WalkerModel()
        self.fit_cycles = np.asarray(fit_cycles, dtype=float)
        self.fit_cracks = np.asarray(fit_cracks, dtype=float)
        self.n_models = n_models
        self.seed = seed
        self.fits: list[WalkerFitResult] = []

    def objective(self, params: np.ndarray) -> np.ndarray:
        params = np.atleast_2d(params.T).T
        c0, gamma, m, sigma_fc = params
        a0 = self.fit_cracks[0] + sigma_fc
        y = self.model.integrate(a0, self.fit_cycles, c0, gamma, m)
        # The paper only asks for more weight on later cracks; i^2 is this replica's reading.
        weights = (np.arange(1, len(self.fit_cycles) + 1) ** 2)[:, None]
        err = (y - self.fit_cracks[:, None]) ** 2
        return np.squeeze(np.sqrt(np.sum(weights * err, axis=0) / weights.sum()))

    def fit(self) -> list[WalkerFitResult]:
        self.fits = []
        for k in range(self.n_models):
            res = differential_evolution(
                self.objective,
                bounds=self.BOUNDS,
                seed=self.seed + k,
                init="random",
                maxiter=200,
                tol=1e-10,
                vectorized=True,
                polish=False,
                updating="deferred",
            )
            self.fits.append(WalkerFitResult(*res.x, objective=float(res.fun)))
        return self.fits

    def predict(self, target_cycles) -> np.ndarray:
        if not self.fits:
            self.fit()
        c0 = np.array([f.c0 for f in self.fits])
        gamma = np.array([f.gamma for f in self.fits])
        m = np.array([f.m for f in self.fits])
        sigma_fc = np.array([f.sigma_fc for f in self.fits])
        a0 = self.fit_cracks[0] + sigma_fc
        curves = self.model.integrate(a0, target_cycles, c0, gamma, m)
        return curves.mean(axis=1)


def linear_regression_extrapolation(cycles, cracks, target_cycles) -> np.ndarray:
    coeffs = np.polyfit(np.asarray(cycles, float), np.asarray(cracks, float), 1)
    return np.polyval(coeffs, np.asarray(target_cycles, float))
