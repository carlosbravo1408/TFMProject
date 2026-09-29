from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The state is clamped at 20 mm, about three times the longest measured crack, so
# diverging trajectories stay finite and their penalty stays interpretable.
_A_MIN_M = 1e-6
_A_MAX_M = 0.02
_RATE_MAX_M_PER_CYCLE = 1e-2


# Schijve, Fatigue of Structures and Materials, Eq. 9.10; valid for -1 <= R <= 1.
def elber_schijve_closure(r: np.ndarray | float) -> np.ndarray | float:
    return 0.55 + 0.35 * np.asarray(r) + 0.1 * np.asarray(r) ** 2


@dataclass(frozen=True)
class GrowthLaw:
    name: str
    param_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    log_params: tuple[str, ...] = ()
    geometry: str = "unit"

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        out = {}
        for i, key in enumerate(self.param_names):
            v = x[:, i]
            out[key] = 10.0 ** v if key in self.log_params else v
        return out

    def geometry_factor(self, a_m, params):
        if self.geometry == "unit":
            return 1.0
        if self.geometry == "hole":
            lam_m = params["lambda_h"] * 1e-3
            return 1.0 + (params["Kt"] - 1.0) * np.exp(-a_m / lam_m)
        raise ValueError(f"unknown geometry model {self.geometry!r}")

    def driving_forces(self, a_m, params, s_max, s_min):
        a_c = np.clip(a_m, _A_MIN_M, _A_MAX_M)
        root = self.geometry_factor(a_c, params) * np.sqrt(np.pi * a_c)
        return s_max * root, (s_max - s_min) * root

    def rate(self, a_m, params, s_max, s_min):  # pragma: no cover - overridden
        raise NotImplementedError


class ParisLaw(GrowthLaw):
    def rate(self, a_m, params, s_max, s_min):
        _, d_k = self.driving_forces(a_m, params, s_max, s_min)
        return params["C"] * d_k ** params["m"]


class WalkerLaw(GrowthLaw):
    def rate(self, a_m, params, s_max, s_min):
        k_max, _ = self.driving_forces(a_m, params, s_max, s_min)
        r = np.clip(s_min / s_max, 0.0, 0.999)
        return params["C0"] * (k_max * (1.0 - r) ** params["gamma"]) ** params["m"]


class WalkerClosureLaw(GrowthLaw):
    def rate(self, a_m, params, s_max, s_min):
        _, d_k = self.driving_forces(a_m, params, s_max, s_min)
        r = np.clip(s_min / s_max, 0.0, 0.999)
        d_k_eff = elber_schijve_closure(r) * d_k
        return params["C0"] * (d_k_eff * (1.0 - r) ** (params["gamma"] - 1.0)) ** params["m"]


class FormanLaw(GrowthLaw):
    def rate(self, a_m, params, s_max, s_min):
        _, d_k = self.driving_forces(a_m, params, s_max, s_min)
        r = np.clip(s_min / s_max, 0.0, 0.999)
        denom = (1.0 - r) * params["Kc"] - d_k
        # The rate diverges near K_c; clamped so the optimiser sees a finite objective.
        denom = np.maximum(denom, 1e-3)
        return params["C"] * d_k ** params["m"] / denom


# Far wider than the generic metallic window: Y = 1 pushes the joint compliance
# into C, and the C-m ridge alone spans ~8 decades of C over m in [1.5, 8].
_LOG_C = (-20.0, -5.0)
_M = (1.0, 8.0)
_GAMMA = (0.0, 1.0)
_KC = (20.0, 200.0)
_KT = (1.0, 8.0)
_LAMBDA_H = (0.05, 15.0)

LAWS: dict[str, GrowthLaw] = {
    "paris": ParisLaw("paris", ("C", "m"), (_LOG_C, _M), log_params=("C",)),
    "walker": WalkerLaw("walker", ("C0", "m", "gamma"), (_LOG_C, _M, _GAMMA), log_params=("C0",)),
    "walker_closure": WalkerClosureLaw(
        "walker_closure", ("C0", "m", "gamma"), (_LOG_C, _M, _GAMMA), log_params=("C0",)
    ),
    "forman": FormanLaw("forman", ("C", "m", "Kc"), (_LOG_C, _M, _KC), log_params=("C",)),
    "paris_hole": ParisLaw(
        "paris_hole", ("C", "m", "Kt", "lambda_h"), (_LOG_C, _M, _KT, _LAMBDA_H),
        log_params=("C",), geometry="hole",
    ),
    "walker_hole": WalkerLaw(
        "walker_hole", ("C0", "m", "gamma", "Kt", "lambda_h"),
        (_LOG_C, _M, _GAMMA, _KT, _LAMBDA_H), log_params=("C0",), geometry="hole",
    ),
    "forman_hole": FormanLaw(
        "forman_hole", ("C", "m", "Kc", "Kt", "lambda_h"),
        (_LOG_C, _M, _KC, _KT, _LAMBDA_H), log_params=("C",), geometry="hole",
    ),
}


def block_average_rate(law: GrowthLaw, a_m, params, load_block: np.ndarray):
    total = 0.0
    n_block = load_block[:, 0].sum()
    for n_i, s_max, s_min in load_block:
        total = total + n_i * law.rate(a_m, params, s_max, s_min)
    return np.clip(total / n_block, 0.0, _RATE_MAX_M_PER_CYCLE)


def integrate_rk4(
    law: GrowthLaw,
    params: dict[str, np.ndarray],
    a0_m,
    delta_cycles: np.ndarray,
    load_block: np.ndarray,
    max_step: float = 200.0,
):
    a = np.atleast_1d(np.asarray(a0_m, dtype=float)).astype(float).copy()
    targets = np.asarray(delta_cycles, dtype=float)
    out = np.empty((len(targets), a.shape[0]))
    cycle = 0.0

    def f(a_state):
        return block_average_rate(law, a_state, params, load_block)

    for i, target in enumerate(targets):
        span = target - cycle
        if span > 0:
            n_steps = max(1, int(np.ceil(span / max_step)))
            h = span / n_steps
            for _ in range(n_steps):
                k1 = f(a)
                k2 = f(a + 0.5 * h * k1)
                k3 = f(a + 0.5 * h * k2)
                k4 = f(a + h * k3)
                a = a + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                a = np.clip(a, _A_MIN_M, _A_MAX_M)
            cycle = target
        out[i] = a
    return out


def predict_mm(
    law: GrowthLaw,
    x: np.ndarray,
    a0_mm: float,
    delta_cycles: np.ndarray,
    load_block: np.ndarray,
    max_step: float = 200.0,
) -> np.ndarray:
    params = law.unpack(x)
    n = len(next(iter(params.values())))
    a0 = np.full(n, a0_mm * 1e-3)
    return integrate_rk4(law, params, a0, delta_cycles, load_block, max_step) * 1e3
