from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import SpecimenData


def paris_law_growth_rate_samples(
    specimens: dict[str, SpecimenData]
) -> tuple[np.ndarray, np.ndarray]:
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
    a, dadn = paris_law_growth_rate_samples(specimens)
    slope, log_c = np.polyfit(np.log(a), np.log(dadn), 1)
    # At constant stress range dK is proportional to sqrt(a), so the slope is m/2.
    return float(2.0 * slope), float(np.exp(log_c))


SAMPLES_PER_LOADING_CYCLE = 20


def equivalent_stress_ratio(
    constant_profile: pd.DataFrame, variable_profile: pd.DataFrame
) -> float:
    delta_sigma_constant = constant_profile["loading"].max() - constant_profile["loading"].min()

    delta_sigma_per_cycle = _per_cycle_stress_ranges(variable_profile)

    # The two modal levels, not min/max: a few mis-segmented cycles yield spurious
    # ranges that would drag the ratio from ~0.95 to 0.9356.
    levels, counts = np.unique(np.round(delta_sigma_per_cycle, 2), return_counts=True)
    dominant = np.sort(levels[np.argsort(counts)[-2:]])
    delta_sigma_variable = float(dominant.mean())
    return float(delta_sigma_variable / delta_sigma_constant)


def _per_cycle_stress_ranges(profile: pd.DataFrame) -> np.ndarray:
    values = profile["loading"].to_numpy(dtype=float)
    per_cycle = SAMPLES_PER_LOADING_CYCLE
    n_cycles = (len(values) - 1) // per_cycle
    return np.array([
        np.ptp(values[i * per_cycle:(i + 1) * per_cycle + 1]) for i in range(n_cycles)
    ])


def _estimate_period(profile: pd.DataFrame) -> float:
    n_cycles = 5
    return (profile["time"].max() - profile["time"].min()) / n_cycles


def variable_loading_exponent(m: float, stress_ratio: float) -> float:
    return float(stress_ratio ** (-m))


def transform_cycles(N: np.ndarray, N0: float, exponent: float) -> np.ndarray:
    N = np.asarray(N, dtype=float)
    return N * (N / N0) ** exponent
