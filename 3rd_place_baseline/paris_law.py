from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution

MM_PER_M = 1000.0

DELTA_SIGMA_CONSTANT_MPA = 100.21 - 4.77

# The paper's C range tops out at 1e-11, too tight to fit its own Tables 5-8.
# m stays above 2 because Eq. 5/6 are singular there.
C_BOUNDS = (1e-13, 1e-8)
M_BOUNDS = (2.05, 4.0)
# MPa; not given by the paper.
DELTA_SIGMA_VE_BOUNDS = (10.0, 500.0)


def _delta_n0f(a0_m: np.ndarray, af_m: np.ndarray, C: float, m: float, delta_sigma_mpa: float) -> np.ndarray:
    k = C * (delta_sigma_mpa * np.sqrt(np.pi)) ** m
    p = m / 2 - 1
    return (1.0 / k) * (1.0 / p) * (a0_m ** (-p) - af_m ** (-p))


def crack_length_after(a0_mm: float, delta_n: float, C: float, m: float, delta_sigma_mpa: float) -> float:
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

    # Stand-in for the paper's unspecified genetic algorithm.
    result = differential_evolution(
        objective, bounds, seed=seed, maxiter=maxiter, tol=1e-10, polish=True
    )
    C, m = result.x[0], result.x[1]
    ds = result.x[2] if fit_stress else delta_sigma_mpa
    return ParisFit(C=C, m=m, delta_sigma_mpa=ds, a0_mm=a0_mm, n0=n0, sse=float(result.fun))


def donor_curve_cycle_at_crack(
    donor_cycles: np.ndarray, donor_cracks_mm: np.ndarray, target_crack_mm: float, degree: int = 2
) -> float:
    coeffs = np.polyfit(donor_cracks_mm, donor_cycles, degree)
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
    own_cycles = np.asarray(own_cycles, dtype=float)
    own_cracks_mm = np.asarray(own_cracks_mm, dtype=float)
    order = np.argsort(own_cycles)
    own_cycles, own_cracks_mm = own_cycles[order], own_cracks_mm[order]
    # The paper discards the first two estimates as off the growth curve.
    kept_cycles = own_cycles[n_discard_early:]
    kept_cracks = own_cracks_mm[n_discard_early:]

    donor_cycles = np.asarray(donor_cycles, dtype=float)
    donor_cracks_mm = np.asarray(donor_cracks_mm, dtype=float)
    cycle_a = donor_curve_cycle_at_crack(donor_cycles, donor_cracks_mm, anchor_crack_mm)
    cycle_b = donor_curve_cycle_at_crack(donor_cycles, donor_cracks_mm, target_crack_mm)
    cycle_a_prime = float(np.interp(anchor_crack_mm, own_cracks_mm, own_cycles))
    # Figure 12's equidistant curve, anchored on crack length: anchoring on each
    # specimen's initiation puts the 7.46 mm point ~13000 cycles later.
    extra_cycle = cycle_b + (cycle_a_prime - cycle_a)

    cycles = np.concatenate([kept_cycles, [extra_cycle]])
    cracks = np.concatenate([kept_cracks, [target_crack_mm]])
    return cycles, cracks


def build_t8_initial_dataset(
    own_cycles: np.ndarray, own_cracks_mm: np.ndarray, target_crack_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    own_cycles = np.asarray(own_cycles, dtype=float)
    own_cracks_mm = np.asarray(own_cracks_mm, dtype=float)
    order = np.argsort(own_cycles)
    own_cycles, own_cracks_mm = own_cycles[order], own_cracks_mm[order]
    slope, intercept = np.polyfit(own_cracks_mm, own_cycles, 1)
    extra_cycle = slope * target_crack_mm + intercept
    cycles = np.concatenate([own_cycles, [extra_cycle]])
    cracks = np.concatenate([own_cracks_mm, [target_crack_mm]])
    return cycles, cracks
