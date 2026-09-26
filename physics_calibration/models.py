"""Fatigue-crack-growth laws and the explicit RK4 integrator used to fit them.

All laws are written in SI-consistent engineering units:

    a          crack length                       [m]
    S          far-field stress                   [MPa]
    K, dK      stress intensity (range)           [MPa*sqrt(m)]
    da/dN      crack growth rate                  [m/cycle]

so a Paris coefficient ``C`` quoted here is directly comparable with the
values tabulated in the fracture-mechanics literature for da/dN in m/cycle
and dK in MPa*sqrt(m).

The stress intensity factor follows the centre-cracked-plate idealisation used
by every PHM 2019 entry that reasoned physically about the problem
(Kong et al., 2020, Eq. 9; Rao et al., 2021, Eq. 3):

    K_max = Y * S_max * sqrt(pi * a),    dK = Y * dS * sqrt(pi * a),   Y = 1.

Y = 1 is *not* physically right here. The dataset ReadMe states that the crack
"initiates at the countersunk hole in the first row [and] will connect these
holes"; the labels are optical surface measurements of that crack. A crack
growing out of a fastener hole sits in the hole's stress-concentration field,
so its Y starts near the hole's K_t and decays towards the plain-plate value
as the crack tip moves away — the Bowie/Newman problem. That decay is not a
detail: it is the difference between da/dN accelerating like a^(m/2) and the
much flatter acceleration actually measured on T3, T6 and T8, and forcing
Y = 1 pushes the identified exponent below the physically admissible m >= 2
for those specimens.

Two geometry models are therefore provided (``geometry`` field):

``unit``     Y = 1. The centre-cracked-plate idealisation used by every PHM
             2019 entry that reasoned physically (Kong et al., 2020, Eq. 9;
             Rao et al., 2021, Eq. 3). Kept as the reference case.
``hole``     Y(a) = 1 + (K_t - 1) * exp(-a / lambda_h), with K_t and the decay
             length lambda_h fitted. This is a *phenomenological surrogate*
             for the Bowie solution, not the Bowie solution itself: it
             reproduces its two asymptotes (Y -> K_t as a -> 0 at the hole
             edge, Y -> 1 for a >> hole radius) with two interpretable
             parameters, which is all 5-7 measurements per specimen can
             support. A fitted K_t near 3 and lambda_h of the order of the
             fastener radius is the consistency check that it is capturing
             the hole and not absorbing noise.

Laws implemented
----------------
``paris``    da/dN = C * dK^m                                (Paris & Erdogan, 1963)
``walker``   da/dN = C0 * (K_max * (1-R)^gamma)^m            (Walker, 1970)
``forman``   da/dN = C * dK^m / ((1-R)*Kc - dK)              (Forman et al., 1967)
``walker_closure``
             Walker driven by the Elber/Schijve effective range for 2024-T3,
             dK_eff = U * dK with U = 0.55 + 0.35R + 0.1R^2 (Schijve, 2009).

Every ``rate`` is vectorised over a population of candidate parameter vectors,
which is what makes a 20 000-evaluation metaheuristic budget tractable in pure
NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Numerical guards. An unstable parameter set must not produce inf/NaN inside
# the integrator, so the state is clamped. The upper clamp is 20 mm: nearly
# three times the longest crack ever measured in this dataset (7.46 mm on T1),
# so it never interferes with a physically meaningful trajectory, while
# keeping a diverging one finite and its penalty score interpretable instead
# of overflowing to 1e12.
_A_MIN_M = 1e-6
_A_MAX_M = 0.02
_RATE_MAX_M_PER_CYCLE = 1e-2


def elber_schijve_closure(r: np.ndarray | float) -> np.ndarray | float:
    """Crack-opening function U = dK_eff/dK for 2024-T3 sheet.

    Schijve's quadratic refinement of Elber's measurement, U = 0.55 + 0.35R +
    0.1R^2, valid for -1 <= R <= +1 and reported specifically for aluminium
    alloy 2024-T3 (Schijve, *Fatigue of Structures and Materials*, Eq. 9.10).
    It is the only 2024-T3-specific fracture-mechanics constant that the
    reviewed literature supplies for this material, so it enters the model
    fixed rather than fitted.
    """
    return 0.55 + 0.35 * np.asarray(r) + 0.1 * np.asarray(r) ** 2


@dataclass(frozen=True)
class GrowthLaw:
    """A crack-growth law: parameter names, search bounds and a rate function.

    ``bounds`` are given for the *search* variables, which use log10 for the
    coefficient (the only way a population-based search can explore six
    decades of ``C``) and linear scale for the exponents.
    """

    name: str
    param_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    log_params: tuple[str, ...] = ()
    geometry: str = "unit"

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Search vector -> named parameters in physical units."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        out = {}
        for i, key in enumerate(self.param_names):
            v = x[:, i]
            out[key] = 10.0 ** v if key in self.log_params else v
        return out

    def geometry_factor(self, a_m, params):
        """Y(a). See the module docstring for the two geometry models."""
        if self.geometry == "unit":
            return 1.0
        if self.geometry == "hole":
            lam_m = params["lambda_h"] * 1e-3  # fitted in mm
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
        # Approaching K_c the rate diverges: clamp instead of letting the
        # integrator emit inf, so the optimiser sees a finite (bad) objective.
        denom = np.maximum(denom, 1e-3)
        return params["C"] * d_k ** params["m"] / denom


# --- Search-space bounds --------------------------------------------------
#
# log10(C) in [-20, -5]: far wider than the generic-metal window quoted by
# Rao et al. (2021) (C in [1e-13, 1e-11], m in [2, 4], after Li, Wang & Gong
# 2012). Two reasons. (i) Y = 1 forces the riveted-joint compliance into C
# (see module docstring), so the published coupon range cannot reproduce the
# observed rates. (ii) C and m are near-perfectly anti-correlated along the
# log-log Paris ridge, log10 C ~ log10(da/dN) - m*log10(dK); over m in [1.5, 8]
# and dK ~ 5-15 MPa*sqrt(m) that ridge alone spans ~8 decades of C, so a
# tighter box would pin the optimum against a wall and make the identified m
# an artefact of the bounds instead of of the data. m in [1.5, 8] likewise
# widens the classical [2, 4] metal window.
# gamma in [0, 1] is Walker's admissible range. Kc in [20, 200] MPa*sqrt(m)
# brackets the fracture toughness of thin 2024-T3 sheet.
_LOG_C = (-20.0, -5.0)
_M = (1.0, 8.0)
_GAMMA = (0.0, 1.0)
_KC = (20.0, 200.0)
# Hole-effect surrogate: K_t in [1, 8] brackets the open-hole value (~3) and
# the higher concentration of a filled countersunk fastener; lambda_h in
# [0.05, 15] mm brackets the fastener radius over which Bowie's factor decays.
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
    """Cycle-averaged da/dN over one repetition of the load block.

    Equivalent to Kong et al. (2020) Eq. 11: the variable-amplitude spectrum
    is treated as a repeating block and the growth it produces is spread
    uniformly over its cycles. For a constant-amplitude specimen the block has
    a single segment and this is just the law's own rate.
    """
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
    """Classical explicit RK4 integration of da/dN over the cycle domain.

    This is the same integrator the TFM's multi-step predictor (*componente
    c*) will run at inference time, so the hyper-parameters identified here
    are calibrated against exactly the numerical scheme that will consume
    them — not against a closed-form solution that only exists for constant
    amplitude.

    Returns an array of shape ``(len(delta_cycles), n_candidates)`` with the
    crack length in metres at each requested cycle offset. Vectorised over the
    candidate population held in ``params``.
    """
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
    """Crack length in mm at ``delta_cycles`` for search vector(s) ``x``."""
    params = law.unpack(x)
    n = len(next(iter(params.values())))
    a0 = np.full(n, a0_mm * 1e-3)
    return integrate_rk4(law, params, a0, delta_cycles, load_block, max_step) * 1e3
