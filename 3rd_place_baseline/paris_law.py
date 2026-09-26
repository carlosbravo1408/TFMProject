"""Crack length prediction via Paris' Law (Sec. 4).

Paris' Law (Eq. 2): da/dN = C*(dK)^m, with the stress intensity factor range
(Eq. 3) dK = Y*ds*sqrt(pi*a), Y = 1. Integrating from an initial crack
length a0 (at cycle N0) to an arbitrary af gives the closed-form cycle
increment (Eq. 5) and its inverse, the crack length reached after a given
cycle increment (Eq. 6). All internal Paris'-Law computation is in SI units
(crack length in metres, stress in MPa, consistent with the paper's stated
material-parameter ranges C in [1e-13, 1e-11], m in [2, 4]); the public
functions accept/return crack length in mm.

Material parameters C and m (Sec. 4.2.2) and, for the variable-amplitude
T8 case, the equivalent stress range ds_ve (Sec. 4.2.3/4.3.2) are fitted by
minimizing Eq. (7)/(8): the squared error between the *observed* cycle
increments of an "initial dataset" of (N, a) anchor points and Eq. (5)'s
prediction. ``scipy.optimize.differential_evolution`` (a population-based
global stochastic optimizer, in the same family as the paper's Genetic
Algorithm) is used in place of the paper's unspecified GA implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution

MM_PER_M = 1000.0

# Eq. (4): constant-amplitude stress range for T1-T7.
DELTA_SIGMA_CONSTANT_MPA = 100.21 - 4.77

# Sec. 4.2.2: metallic material parameter ranges ("for metallic materials, m
# varies between 2 and 4, and C varies from 1e-13 to 1e-11", citing Li, Wang
# & Gong 2012) -- a general reference range, not a hard bound for this
# specific specimen. Fitting the paper's own Table 5/6/7/8 numbers (rapid
# crack growth reaching into fatigue Stage 3, Fig. 8, over a handful of
# widely-spaced anchor points) needs C well above 1e-11 at m near the top of
# its range, so C's upper bound is relaxed here to let the optimizer reach a
# good fit instead of pinning against an artificially tight boundary. m is
# kept away from 2 (m/2-1 = 0 would make Eq. 5/6 singular).
C_BOUNDS = (1e-13, 1e-8)
M_BOUNDS = (2.05, 4.0)
DELTA_SIGMA_VE_BOUNDS = (10.0, 500.0)  # MPa; not given explicitly for T8 in the paper.


def _delta_n0f(a0_m: np.ndarray, af_m: np.ndarray, C: float, m: float, delta_sigma_mpa: float) -> np.ndarray:
    """Eq. (5): cycle increment from a0 to af."""
    k = C * (delta_sigma_mpa * np.sqrt(np.pi)) ** m
    p = m / 2 - 1
    return (1.0 / k) * (1.0 / p) * (a0_m ** (-p) - af_m ** (-p))


def crack_length_after(a0_mm: float, delta_n: float, C: float, m: float, delta_sigma_mpa: float) -> float:
    """Eq. (6): crack length (mm) reached after ``delta_n`` cycles from an
    initial crack length ``a0_mm`` (mm)."""
    a0_m = a0_mm / MM_PER_M
    k = C * (delta_sigma_mpa * np.sqrt(np.pi)) ** m
    p = m / 2 - 1
    x = a0_m ** (-p) - delta_n * p * k
    if x <= 0:
        return float("inf")
    af_m = x ** (-1 / p)
    return af_m * MM_PER_M


@dataclass
class ParisFit:
    C: float
    m: float
    delta_sigma_mpa: float
    a0_mm: float
    n0: float
    sse: float

    def predict(self, cycles: np.ndarray) -> np.ndarray:
        cycles = np.atleast_1d(np.asarray(cycles, dtype=float))
        return np.array(
            [crack_length_after(self.a0_mm, n - self.n0, self.C, self.m, self.delta_sigma_mpa) for n in cycles]
        )


def fit_paris_law(
    cycles: np.ndarray,
    cracks_mm: np.ndarray,
    delta_sigma_mpa: float | None = DELTA_SIGMA_CONSTANT_MPA,
    seed: int = 0,
    maxiter: int = 300,
) -> ParisFit:
    """Fit (C, m) -- and, if ``delta_sigma_mpa`` is ``None``, the equivalent
    stress range too (Eq. 8, the T8/variable-load case) -- to an "initial
    dataset" of (cycle, crack length) anchor points via Eq. (7)/(8)."""
    cycles = np.asarray(cycles, dtype=float)
    cracks_mm = np.asarray(cracks_mm, dtype=float)
    order = np.argsort(cycles)
    cycles, cracks_mm = cycles[order], cracks_mm[order]
    n0, a0_mm = cycles[0], cracks_mm[0]
    a0_m = a0_mm / MM_PER_M
    delta_n_obs = cycles[1:] - n0
    af_m_obs = cracks_mm[1:] / MM_PER_M

    fit_stress = delta_sigma_mpa is None
    bounds = [C_BOUNDS, M_BOUNDS] + ([DELTA_SIGMA_VE_BOUNDS] if fit_stress else [])

    def objective(p):
        C, m = p[0], p[1]
        ds = p[2] if fit_stress else delta_sigma_mpa
        pred = _delta_n0f(np.full_like(af_m_obs, a0_m), af_m_obs, C, m, ds)
        if not np.all(np.isfinite(pred)):
            return 1e18
        return float(np.sum((delta_n_obs - pred) ** 2))

    result = differential_evolution(
        objective, bounds, seed=seed, maxiter=maxiter, tol=1e-10, polish=True
    )
    C, m = result.x[0], result.x[1]
    ds = result.x[2] if fit_stress else delta_sigma_mpa
    return ParisFit(C=C, m=m, delta_sigma_mpa=ds, a0_mm=a0_mm, n0=n0, sse=float(result.fun))


# ---------------------------------------------------------------------------
# "Initial dataset" construction (Sec. 4.3.1 / 4.3.2): a handful of (N, a)
# anchor points -- the specimen's own crack-length estimates plus one
# extrapolated point at the largest crack length observed across T1-T6 (the
# paper's 7.46 mm) -- used to fit the Paris'-Law parameters above.
# ---------------------------------------------------------------------------


def donor_curve_cycle_at_crack(
    donor_cycles: np.ndarray, donor_cracks_mm: np.ndarray, target_crack_mm: float, degree: int = 2
) -> float:
    """Sec. 4.3.1: fit a polynomial crack-growth curve a(N) to a donor
    specimen's own (cycle, crack) history (the paper picks T4, "the most
    thorough information about crack length growth in terms of load cycle
    number") and invert it to find the cycle at which it reaches
    ``target_crack_mm``. ``donor_cycles`` should already be relative to the
    donor's own crack-initiation cycle."""
    coeffs = np.polyfit(donor_cracks_mm, donor_cycles, degree)  # invert: N as a function of a
    return float(np.polyval(coeffs, target_crack_mm))


ANCHOR_CRACK_MM = 2.0


def build_t7_initial_dataset(
    own_cycles: np.ndarray,
    own_cracks_mm: np.ndarray,
    donor_cycles: np.ndarray,
    donor_cracks_mm: np.ndarray,
    target_crack_mm: float,
    n_discard_early: int = 2,
    anchor_crack_mm: float = ANCHOR_CRACK_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Sec. 4.3.1's initial dataset for T7 (Table 5): the specimen's own
    nonzero crack-length estimates (the first ``n_discard_early`` are
    dropped -- the paper discards its first two as they "greatly deviate
    from the relationship curve") plus a point at ``target_crack_mm``
    obtained by the equidistant-curve construction of the paper's Figure 12.

    That construction fits a polynomial to the donor's own (crack, cycle)
    history and reads off two points on it: A at ``anchor_crack_mm`` and B at
    ``target_crack_mm``. Assuming T7's crack-growth curve is the equidistant
    curve of the donor's, the cycle of B' on T7 satisfies
    ``N_B' - N_B = N_A' - N_A``, where A' is T7's own cycle at
    ``anchor_crack_mm``. The paper reports 54,795 cycles for the 7.46 mm point
    of T7 obtained this way.

    Anchoring the translation on the crack length -- and not on each
    specimen's crack-initiation cycle -- is what the paper prescribes, and the
    difference is large: initiation-anchored translation puts the 7.46 mm
    point some 13,000 cycles later, which flattens the subsequent Paris fit.
    """
    own_cycles = np.asarray(own_cycles, dtype=float)
    own_cracks_mm = np.asarray(own_cracks_mm, dtype=float)
    order = np.argsort(own_cycles)
    own_cycles, own_cracks_mm = own_cycles[order], own_cracks_mm[order]
    kept_cycles = own_cycles[n_discard_early:]
    kept_cracks = own_cracks_mm[n_discard_early:]

    donor_cycles = np.asarray(donor_cycles, dtype=float)
    donor_cracks_mm = np.asarray(donor_cracks_mm, dtype=float)
    cycle_a = donor_curve_cycle_at_crack(donor_cycles, donor_cracks_mm, anchor_crack_mm)
    cycle_b = donor_curve_cycle_at_crack(donor_cycles, donor_cracks_mm, target_crack_mm)
    cycle_a_prime = float(np.interp(anchor_crack_mm, own_cracks_mm, own_cycles))
    extra_cycle = cycle_b + (cycle_a_prime - cycle_a)

    cycles = np.concatenate([kept_cycles, [extra_cycle]])
    cracks = np.concatenate([kept_cracks, [target_crack_mm]])
    return cycles, cracks


def build_t8_initial_dataset(
    own_cycles: np.ndarray, own_cracks_mm: np.ndarray, target_crack_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sec. 4.3.2's initial dataset for T8 (Table 7): the specimen's own
    nonzero crack-length estimates plus a point at ``target_crack_mm``
    obtained by linear extrapolation of those estimates (the paper: "linear
    interpolation from any two of these three data points")."""
    own_cycles = np.asarray(own_cycles, dtype=float)
    own_cracks_mm = np.asarray(own_cracks_mm, dtype=float)
    order = np.argsort(own_cycles)
    own_cycles, own_cracks_mm = own_cycles[order], own_cracks_mm[order]
    slope, intercept = np.polyfit(own_cracks_mm, own_cycles, 1)
    extra_cycle = slope * target_crack_mm + intercept
    cycles = np.concatenate([own_cycles, [extra_cycle]])
    cracks = np.concatenate([own_cracks_mm, [target_crack_mm]])
    return cycles, cracks
