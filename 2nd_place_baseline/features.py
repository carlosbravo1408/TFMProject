"""Feature extraction from the pre-processed S0 windows (Section 3.2.2).

Based on the physical interpretation of crack-length effects on the received
Lamb waves, eight candidate features are computed for each measurement cycle,
always comparing the current (possibly cracked) window against the *reference*
window of the same specimen — its first measured cycle, where no crack is
present:

Energy-loss features (the received energy decreases as the crack grows):
  1. ``max_amplitude``     — maximum absolute amplitude of the S0 window.
  2. ``max_energy``        — maximum of the short-time energy of the window.
  3. ``dtw_residual_energy`` — energy of the residual after aligning the
     current window onto the reference window with dynamic time warping.

Phase-change features (scattering at the crack delays the transmitted wave):
  4. ``xcorr_lag``         — cross-correlation time lag (us) w.r.t. the
     reference window.
  5. ``phase_delay``       — point time delay (us): time of the window's
     absolute peak minus the time of the reference window's absolute peak.
  6. ``dtw_distance``      — dynamic-time-warping distance to the reference.

Similarity feature (crack discontinuities distort the transmitted shape):
  7. ``corr_coef``         — Pearson correlation coefficient between the
     current and reference windows.

Sequential-process feature:
  8. ``prev_crack``        — previously estimated/measured crack length (mm).

The optimal subset found by the paper's k-fold search (Section 3.2.4) is
``max_amplitude, max_energy, phase_delay, corr_coef, prev_crack``.

Considering the different ranges of the features, they are standardized with
a standard normal distribution (z-score fitted on the training rows).
"""
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

# Optimal feature subset reported in Section 3.2.4 of the paper.
OPTIMAL_FEATURES = ["max_amplitude", "max_energy", "phase_delay", "corr_coef", "prev_crack"]

# Short-time energy window: one period of the ~200 kHz tone at 20 MHz sampling.
SHORT_TIME_ENERGY_SAMPLES = 100


def max_amplitude(window: np.ndarray) -> float:
    return float(np.max(np.abs(window)))


def max_energy(window: np.ndarray, n: int = SHORT_TIME_ENERGY_SAMPLES) -> float:
    """Maximum of the moving (short-time) energy of the window."""
    energy = np.convolve(window**2, np.ones(n), mode="valid")
    return float(np.max(energy))


def cross_correlation_lag_us(window: np.ndarray, reference: np.ndarray) -> float:
    """Lag (us) that maximizes the cross-correlation with the reference
    window; positive = current signal delayed w.r.t. the reference."""
    corr = np.correlate(window, reference, mode="full")
    lag = int(np.argmax(corr)) - (len(reference) - 1)
    return lag * DT_S * 1e6


def point_time_delay_us(window: np.ndarray, reference: np.ndarray) -> float:
    """Point time delay ('phase delay', us): arrival-time difference of the
    S0 packet peak w.r.t. the reference window's packet peak. The peak is
    located on the signal envelope (Hilbert transform), which tracks the
    packet arrival robustly; tracking the raw absolute peak instead jumps by
    a carrier period whenever two neighbouring oscillation peaks swap order
    (visible in Figure 8(c) of the paper, where T4's delay jumps to ~4.5 us
    at the last cycle)."""
    env = np.abs(hilbert(window))
    env_ref = np.abs(hilbert(reference))
    return (int(np.argmax(env)) - int(np.argmax(env_ref))) * DT_S * 1e6


def correlation_coefficient(window: np.ndarray, reference: np.ndarray) -> float:
    return float(np.corrcoef(window, reference)[0, 1])


def _dtw_path(x: np.ndarray, y: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """Classic O(n*m) dynamic time warping with absolute-difference cost.
    Returns (accumulated distance, warping path)."""
    n, m = len(x), len(y)
    cost = np.abs(x[:, None] - y[None, :])
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        # acc[i, j] depends on acc[i, j-1]; fill each row sequentially.
        row = acc[i]
        arow = acc[i - 1]
        crow = cost[i - 1]
        for j in range(1, m + 1):
            row[j] = crow[j - 1] + min(arow[j], arow[j - 1], row[j - 1])
    # Backtrack
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


# The DTW features are computed on decimated windows to keep the O(n^2)
# alignment cheap; the decimation (factor 4 -> 100 points per window) does not
# change their monotonic trends.
DTW_DECIMATION = 4


def dtw_features(window: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """(dtw_residual_energy, dtw_distance) of the current window w.r.t. the
    reference window. The residual is the reference minus the current window
    warped onto the reference's time base along the DTW path."""
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
    """The seven signal-based features (all but ``prev_crack``)."""
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
    """Builds the per-cycle feature table of a specimen.

    ``reference_override`` maps specimen name -> cycle to use as the
    undamaged reference instead of the first measured cycle. It is used for
    T8, whose cycle-40000 record is anomalous — its received wave packet
    differs completely from every later T8 record (arriving ~10 us apart)
    even though both its repetitions agree, so it cannot serve as the
    undamaged reference. Cycle 50000 (also crack-free) is used instead; the
    resulting correlation-coefficient trend (~0.45 at 40000, ~0.9 at the
    last cycles) closely matches the T8 curve of Figure 8(d) in the paper,
    which suggests the authors made the same choice."""

    DEFAULT_REFERENCE_OVERRIDE = {"T8": 50000}

    # Minimum correlation between the two repetitions of a reference cycle for
    # the second one to stand in as the undamaged baseline. Seven of the eight
    # specimens clear it with 0.99 or better; only T1's cycle 50000 fails, at
    # 0.20, and T1 has no other crack-free cycle to fall back to.
    REFERENCE_COHERENCE_MIN = 0.9

    # Record whose actuation burst every other record is cross-correlated
    # against (see ``set_alignment_anchor``).
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
        """Pre-processed S0 windows for every measured cycle of a specimen.

        The alignment lag of each record is taken from the specimen's
        consensus rather than from its own cross-correlation peak, because
        the sensing geometry does not change within a specimen and a lag
        that departs from the rest is a lobe-lock failure, not a real delay
        (see ``consensus_shift``)."""
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
        """Alignment lag shared by every record of a specimen."""
        shifts = [
            self.preprocessor.record_shift(specimen.signal(c)["ch1"].to_numpy())
            for c in specimen.signal_cycles
        ]
        return consensus_shift(shifts)[0]

    def set_alignment_anchor(self, specimens: dict) -> tuple[str, int]:
        """Fix the common actuation burst every specimen is aligned against.

        It is ``ALIGNMENT_ANCHOR``, the undamaged reference cycle of the
        training specimen with the smallest PM over the training folds, which
        is the paper's own selection metric and involves no validation
        specimen. Fixing it explicitly keeps the anchor out of the hands of
        dictionary order.

        The choice matters far more than that margin suggests, and the
        notebook reports the sweep: across the six candidate anchors PM spans
        only 1.10 to 1.28 while the final penalty spans 111 to 440, with no
        ordering between the two. The anchor is therefore a nuisance
        parameter with a large effect and no training-side signal to pin it
        down, and that has to be declared rather than hidden behind whichever
        record happened to be read first.
        """
        name, cycle = self.ALIGNMENT_ANCHOR
        self.preprocessor.set_reference(specimens[name].signal(cycle)["ch1"].to_numpy())
        self.alignment_anchor = (name, cycle)
        return self.alignment_anchor

    def reference_window(self, specimen) -> np.ndarray:
        """Undamaged reference window of a specimen.

        The released copy of the dataset lacks the 'Baseline' (cycle 0)
        folders described in the ReadMe, so the reference is taken at the
        specimen's first measured cycle (crack length 0). To keep it an
        *independent* record — as a true baseline measurement would be — the
        second repetition (``signal_2``) is used: comparing every
        ``signal_1`` window (including the reference cycle's own) against it
        yields realistic undamaged feature values instead of the degenerate
        corr = 1 / delay = 0 that self-comparison would produce.

        If the two repetitions of the reference cycle disagree (their windows
        correlate below 0.9 — the case for T1, whose cycle-50000 records are
        anomalously dissimilar), ``signal_2`` is not a trustworthy stand-in
        and the specimen falls back to the ``signal_1`` self-reference."""
        cycle = self.reference_override.get(specimen.name, specimen.reference_cycle())
        from data_loader import available_signal_channels

        available = available_signal_channels(specimen.root, specimen.name, cycle)
        own = self.specimen_windows(specimen)[cycle]
        if "signal_2" not in available:
            self.degenerate_reference.add(specimen.name)
            return own
        df = specimen.signal(cycle, "signal_2")
        repeat = self.preprocessor.s0_window(
            df["ch1"].to_numpy(), df["ch2"].to_numpy(), self.specimen_consensus_shift(specimen)
        )
        if np.corrcoef(own, repeat)[0, 1] < self.REFERENCE_COHERENCE_MIN:
            self.degenerate_reference.add(specimen.name)
            return own
        return repeat

    def specimen_feature_table(self, specimen, labeled_only: bool = True) -> pd.DataFrame:
        """Signal features for each cycle (chronological). ``prev_crack`` and
        the target ``crack_length_mm`` are filled from the description labels;
        for validation specimens the target may be missing (NaN) and
        ``prev_crack`` is overwritten recursively at estimation time."""
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
