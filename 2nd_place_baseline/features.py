from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from signal_processing import DT_S, SignalPreprocessor, consensus_shift

FEATURE_NAMES = [
    "max_amplitude",
    "max_energy",
    "dtw_residual_energy",
    "xcorr_lag",
    "phase_delay",
    "dtw_distance",
    "corr_coef",
    "prev_crack",
]

OPTIMAL_FEATURES = ["max_amplitude", "max_energy", "phase_delay", "corr_coef", "prev_crack"]

# One period of the 200 kHz tone at 20 MHz.
SHORT_TIME_ENERGY_SAMPLES = 100


def max_amplitude(window: np.ndarray) -> float:
    return float(np.max(np.abs(window)))


def max_energy(window: np.ndarray, n: int = SHORT_TIME_ENERGY_SAMPLES) -> float:
    energy = np.convolve(window**2, np.ones(n), mode="valid")
    return float(np.max(energy))


def cross_correlation_lag_us(window: np.ndarray, reference: np.ndarray) -> float:
    corr = np.correlate(window, reference, mode="full")
    lag = int(np.argmax(corr)) - (len(reference) - 1)
    return lag * DT_S * 1e6


def point_time_delay_us(window: np.ndarray, reference: np.ndarray) -> float:
    # Envelope peak, not raw peak: the raw peak jumps by a carrier period when two
    # oscillation peaks swap order.
    env = np.abs(hilbert(window))
    env_ref = np.abs(hilbert(reference))
    return (int(np.argmax(env)) - int(np.argmax(env_ref))) * DT_S * 1e6


def correlation_coefficient(window: np.ndarray, reference: np.ndarray) -> float:
    return float(np.corrcoef(window, reference)[0, 1])


def _dtw_path(x: np.ndarray, y: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    n, m = len(x), len(y)
    cost = np.abs(x[:, None] - y[None, :])
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row = acc[i]
        arow = acc[i - 1]
        crow = cost[i - 1]
        for j in range(1, m + 1):
            row[j] = crow[j - 1] + min(arow[j], arow[j - 1], row[j - 1])
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(acc[n, m]), path


# Keeps the O(n^2) alignment cheap without changing the feature trends.
DTW_DECIMATION = 4


def dtw_features(window: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    x = window[::DTW_DECIMATION]
    y = reference[::DTW_DECIMATION]
    distance, path = _dtw_path(x, y)
    warped_sum = np.zeros_like(y)
    warped_count = np.zeros_like(y)
    for i, j in path:
        warped_sum[j] += x[i]
        warped_count[j] += 1
    warped = warped_sum / np.maximum(warped_count, 1)
    residual = y - warped
    return float(np.sum(residual**2)), distance


def signal_features(window: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    residual_energy, distance = dtw_features(window, reference)
    return {
        "max_amplitude": max_amplitude(window),
        "max_energy": max_energy(window),
        "dtw_residual_energy": residual_energy,
        "xcorr_lag": cross_correlation_lag_us(window, reference),
        "phase_delay": point_time_delay_us(window, reference),
        "dtw_distance": distance,
        "corr_coef": correlation_coefficient(window, reference),
    }


class FeatureExtractor:
    # T8's cycle-40000 packet arrives ~10 us apart from every later T8 record.
    DEFAULT_REFERENCE_OVERRIDE = {"T8": 50000}

    # Every specimen reaches 0.99 except T1 (0.20 at cycle 50000).
    REFERENCE_COHERENCE_MIN = 0.9

    # Lowest-PM training specimen. Fixed explicitly: the anchor moves the final
    # penalty between 111 and 440 with no training-side signal to choose it.
    ALIGNMENT_ANCHOR = ("T6", 55000)

    def __init__(
        self,
        preprocessor: SignalPreprocessor | None = None,
        reference_override: dict[str, int] | None = None,
    ) -> None:
        self.preprocessor = preprocessor or SignalPreprocessor()
        self.alignment_anchor: tuple[str, int] | None = None
        self.degenerate_reference: set[str] = set()
        self.reference_override = (
            dict(self.DEFAULT_REFERENCE_OVERRIDE)
            if reference_override is None
            else reference_override
        )

    def specimen_windows(self, specimen, channel: str = "signal_1") -> dict[int, np.ndarray]:
        if self.preprocessor.reference_ch1 is None:
            raise RuntimeError(
                "No alignment anchor set; call set_alignment_anchor first so every "
                "specimen is aligned against the same actuation burst."
            )
        cycles = list(specimen.signal_cycles)
        records = {c: specimen.signal(c, channel) for c in cycles}
        shifts = [self.preprocessor.record_shift(records[c]["ch1"].to_numpy()) for c in cycles]
        _, shifts = consensus_shift(shifts)
        return {
            c: self.preprocessor.s0_window(
                records[c]["ch1"].to_numpy(), records[c]["ch2"].to_numpy(), shift
            )
            for c, shift in zip(cycles, shifts)
        }

    def specimen_consensus_shift(self, specimen) -> int:
        shifts = [
            self.preprocessor.record_shift(specimen.signal(c)["ch1"].to_numpy())
            for c in specimen.signal_cycles
        ]
        return consensus_shift(shifts)[0]

    def set_alignment_anchor(self, specimens: dict) -> tuple[str, int]:
        name, cycle = self.ALIGNMENT_ANCHOR
        self.preprocessor.set_reference(specimens[name].signal(cycle)["ch1"].to_numpy())
        self.alignment_anchor = (name, cycle)
        return self.alignment_anchor

    def reference_window(self, specimen) -> np.ndarray:
        cycle = self.reference_override.get(specimen.name, specimen.reference_cycle())
        from data_loader import available_signal_channels

        available = available_signal_channels(specimen.root, specimen.name, cycle)
        own = self.specimen_windows(specimen)[cycle]
        if "signal_2" not in available:
            self.degenerate_reference.add(specimen.name)
            return own
        # The dataset ships no cycle-0 baseline; signal_2 stands in as an independent
        # record, since comparing signal_1 with itself gives corr = 1 and delay = 0.
        df = specimen.signal(cycle, "signal_2")
        repeat = self.preprocessor.s0_window(
            df["ch1"].to_numpy(), df["ch2"].to_numpy(), self.specimen_consensus_shift(specimen)
        )
        if np.corrcoef(own, repeat)[0, 1] < self.REFERENCE_COHERENCE_MIN:
            self.degenerate_reference.add(specimen.name)
            return own
        return repeat

    def specimen_feature_table(self, specimen, labeled_only: bool = True) -> pd.DataFrame:
        windows = self.specimen_windows(specimen)
        reference = self.reference_window(specimen)
        cycles = specimen.labeled_cycles() if labeled_only else specimen.signal_cycles
        rows = []
        prev_crack = 0.0
        for cycle in cycles:
            feats = signal_features(windows[cycle], reference)
            crack = specimen.crack_length(cycle)
            feats.update(
                {
                    "specimen": specimen.name,
                    "cycle": cycle,
                    "prev_crack": prev_crack,
                    "crack_length_mm": np.nan if crack is None else crack,
                }
            )
            rows.append(feats)
            if crack is not None:
                prev_crack = crack
        cols = ["specimen", "cycle"] + FEATURE_NAMES + ["crack_length_mm"]
        return pd.DataFrame(rows)[cols]
