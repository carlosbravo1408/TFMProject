"""Variable-loading correction for specimen T8 (Section 4.2).

Two ingredients are needed to translate the constant-amplitude training
curves (T3, T4) into an equivalent curve usable for the variable-loading
target specimen T8:

1. The Paris' law exponent ``m`` (Eq. 20-21), estimated here from the
   pooled training data via a log-log linear regression of the crack growth
   rate da/dN against the crack length a. The paper instead used a
   Delayed-Rejection-Adaptive-Metropolis MCMC fit (m = 2.0597); we use a
   simpler deterministic regression as a documented approximation -- since
   the stress range is the same (constant loading) across all training
   specimens, it cancels out of the exponent estimate and only shifts the
   fitted intercept (Eq. 21: da/dN ~ (delta_sigma)^m independent of a).
2. The equivalent constant-loading stress ratio (Eq. 18-19), computed
   directly from the "Constant/Variable Loading Profile" CSVs shipped with
   the dataset rather than hard-coded, giving ~0.947 (paper reports ~0.95).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import SpecimenData


def paris_law_growth_rate_samples(
    specimens: dict[str, SpecimenData]
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled (a, da/dN) samples from constant-loading specimens' measured
    post-initiation crack-length-vs-cycle data, via forward differences."""
    a_samples, dadn_samples = [], []
    for sd in specimens.values():
        cycles = sd.post_initiation_cycles()
        if len(cycles) < 2:
            continue
        N = np.array(cycles, dtype=float)
        a = np.array([sd.crack_length(c) for c in cycles], dtype=float)
        dN = np.diff(N)
        da = np.diff(a)
        valid = (dN > 0) & (da > 0)
        a_mid = (a[:-1] + a[1:]) / 2
        dadn = da / dN
        a_samples.append(a_mid[valid])
        dadn_samples.append(dadn[valid])
    return np.concatenate(a_samples), np.concatenate(dadn_samples)


def fit_paris_law_exponent(specimens: dict[str, SpecimenData]) -> tuple[float, float]:
    """Return ``(m, C)`` from da/dN = C * a^m (log-log linear regression).

    Since all specimens share the same constant applied stress range,
    Delta K = Y * dsigma * sqrt(pi*a) ∝ sqrt(a), so log(da/dN) = log(C) +
    m*log(a) with C absorbing the (specimen-independent) stress-range term.
    """
    a, dadn = paris_law_growth_rate_samples(specimens)
    log_a = np.log(a)
    log_dadn = np.log(dadn)
    m, log_c = np.polyfit(log_a, log_dadn, 1)
    return float(m), float(np.exp(log_c))


def equivalent_stress_ratio(
    constant_profile: pd.DataFrame, variable_profile: pd.DataFrame
) -> float:
    """Eq. 18-19: ratio of the variable loading's equivalent constant stress
    range to the true constant loading's stress range, computed from the
    "Loading Profile" CSVs. The variable profile alternates between a lower-
    and an upper-amplitude block; their average is the equivalent range.
    """
    delta_sigma_constant = constant_profile["loading"].max() - constant_profile["loading"].min()

    period = _estimate_period(constant_profile)
    cycle_idx = (variable_profile["time"] // period).astype(int)
    per_cycle_max = variable_profile.groupby(cycle_idx)["loading"].max()
    per_cycle_min = variable_profile.groupby(cycle_idx)["loading"].min()
    delta_sigma_per_cycle = per_cycle_max - per_cycle_min

    levels = np.sort(delta_sigma_per_cycle.unique())
    delta_sigma_lower, delta_sigma_upper = levels.min(), levels.max()
    delta_sigma_variable = (delta_sigma_lower + delta_sigma_upper) / 2
    return float(delta_sigma_variable / delta_sigma_constant)


def _estimate_period(profile: pd.DataFrame) -> float:
    """Loading period from a profile spanning an integer number of cycles
    (the "Constant Loading Profile-5 cycles.csv" files: 5 cycles over the
    file's full time span)."""
    n_cycles = 5
    return (profile["time"].max() - profile["time"].min()) / n_cycles


def variable_loading_exponent(m: float, stress_ratio: float) -> float:
    """Eq. 22: dN' = (stress_ratio)^(-m) dN -> the exponent used in Eq. 24."""
    return float(stress_ratio ** (-m))


def transform_cycles(N: np.ndarray, N0: float, exponent: float) -> np.ndarray:
    """Eq. 24: N*(k) = N(k) * (N(k)/N0)^exponent."""
    N = np.asarray(N, dtype=float)
    return N * (N / N0) ** exponent
