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
    """Return Paris' ``(m, C)`` by log-log regression of da/dN against ``a``.

    All specimens share the same constant applied stress range, so
    ``Delta K = Y * dsigma * sqrt(pi * a)`` is proportional to ``sqrt(a)`` and
    Paris' law reads ``da/dN = C * (Delta K)^m ∝ a^(m/2)``. The slope of
    ``log(da/dN)`` against ``log(a)`` is therefore ``m / 2``, and the exponent
    has to be recovered as twice that slope.

    Reading the slope directly as ``m`` halves it: on this dataset that is
    1.3291 instead of 2.6581, against the 2.0597 the paper obtains by MCMC.
    """
    a, dadn = paris_law_growth_rate_samples(specimens)
    slope, log_c = np.polyfit(np.log(a), np.log(dadn), 1)
    return float(2.0 * slope), float(np.exp(log_c))


SAMPLES_PER_LOADING_CYCLE = 20


def equivalent_stress_ratio(
    constant_profile: pd.DataFrame, variable_profile: pd.DataFrame
) -> float:
    """Eq. 18-19: ratio of the variable loading's equivalent constant stress
    range to the true constant loading's stress range, computed from the
    "Loading Profile" CSVs. The variable profile alternates between a lower-
    and an upper-amplitude block; their average is the equivalent range.
    """
    delta_sigma_constant = constant_profile["loading"].max() - constant_profile["loading"].min()

    delta_sigma_per_cycle = _per_cycle_stress_ranges(variable_profile)

    # The variable profile holds exactly two load levels, 85.23 and 95.44 MPa,
    # each over 500 of the block's 1000 cycles. Two safeguards are needed to
    # recover them. First, the cycles are segmented by samples per cycle: an
    # estimated period does not divide the sampling grid exactly, so roughly
    # one bin in thirteen falls a sample short of its cycle's extremum and
    # yields a spurious range (83.14 MPa in 78 bins, 93.10 in 75). Second, the
    # two modal levels are taken rather than the minimum and the maximum,
    # since a single mis-segmented bin would otherwise set the lower level and
    # drag the ratio from the paper's ~0.95 down to 0.9356.
    levels, counts = np.unique(np.round(delta_sigma_per_cycle, 2), return_counts=True)
    dominant = np.sort(levels[np.argsort(counts)[-2:]])
    delta_sigma_variable = float(dominant.mean())
    return float(delta_sigma_variable / delta_sigma_constant)


def _per_cycle_stress_ranges(profile: pd.DataFrame) -> np.ndarray:
    """Peak-to-trough stress range of every load cycle in a profile.

    The profile files are sampled at a fixed number of points per cycle, so the
    cycles are segmented by that count rather than by an estimated period,
    which avoids straddling two cycles in one bin.
    """
    values = profile["loading"].to_numpy(dtype=float)
    per_cycle = SAMPLES_PER_LOADING_CYCLE
    n_cycles = (len(values) - 1) // per_cycle
    return np.array([
        np.ptp(values[i * per_cycle:(i + 1) * per_cycle + 1]) for i in range(n_cycles)
    ])


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
