"""Preprocessing and feature extraction for the PZT wave signals.

Implements Sections 3.1.1 (Preprocessing) and 3.1.2 (Feature extraction) of
the paper:

1. A 5th-order Butterworth band-pass filter (100-500 kHz) removes noise from
   the raw received signal (``ch2``), since the actuator's center frequency
   is 200 kHz.
2. A 400-sample window (four cycles of a 200 kHz tone at the 20 MHz sampling
   rate: 4 / 200e3 * 20e6 = 400) covering the leading edge of the received
   wave packet is extracted.
3. Four features are computed on that window: RMS, standard deviation, a
   metric of orthogonality against an "undamaged" reference window, and the
   FFT magnitude at 300 kHz.
4. Each feature is normalized by the same feature computed on the
   undamaged/baseline signal, to remove specimen-to-specimen actuation
   amplitude variation (Section 3.1.2, last paragraph).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

SAMPLING_FREQUENCY_HZ = 20e6
BANDPASS_LOW_HZ = 100e3
BANDPASS_HIGH_HZ = 500e3
BANDPASS_ORDER = 5

ACTUATOR_CENTER_FREQ_HZ = 200e3
FEATURE_WINDOW_N_CYCLES = 4
FEATURE_WINDOW_SAMPLES = int(
    round(FEATURE_WINDOW_N_CYCLES / ACTUATOR_CENTER_FREQ_HZ * SAMPLING_FREQUENCY_HZ)
)  # 400 samples

TARGET_FREQUENCY_HZ = 300e3

# The onset detected on the actuator channel marks when the burst is emitted,
# not when it arrives at the receiver. Anchoring the feature window there puts
# it on the electrical crosstalk between channels: measured on T7 at cycle
# 40167 it spans 75.8-95.8 us, holds 3.2 % of the filtered receiver energy and
# straddles the silent 90-100 us gap, while the propagated S0 packet arrives
# from 100 us onwards. The propagation delay is therefore measured once per
# specimen, on its undamaged record, as the first envelope arrival after the
# actuation burst has died out.
ARRIVAL_SEARCH_SKIP_SAMPLES = 400
ARRIVAL_SEARCH_END_SAMPLES = 2800
ARRIVAL_THRESHOLD_RATIO = 0.25

FEATURE_NAMES = ["rms", "std", "orthogonality", "mag_300khz"]


def bandpass_filter(
    x: np.ndarray,
    fs: float = SAMPLING_FREQUENCY_HZ,
    low: float = BANDPASS_LOW_HZ,
    high: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """5th-order Butterworth band-pass filter, zero-phase (filtfilt)."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def detect_onset(x: np.ndarray, threshold_ratio: float = 0.2) -> int:
    """Index of the first sample where a burst's envelope rises above
    ``threshold_ratio`` times its peak absolute amplitude.

    Applied to the excitation channel (``ch1``), not the received signal:
    the actuator burst is time-locked to the acquisition trigger and its
    onset index is stable across cycles/specimens (verified empirically,
    +/-1 sample), unlike the received signal ``ch2`` whose shape changes
    with crack damage -- thresholding ``ch2`` directly would shift the
    feature window and misalign comparisons across cycles.
    """
    envelope = np.abs(x)
    peak = envelope.max()
    if peak == 0:
        return 0
    above = np.where(envelope >= threshold_ratio * peak)[0]
    return int(above[0]) if len(above) else 0


def extract_window(
    x: np.ndarray, onset: int, n_samples: int = FEATURE_WINDOW_SAMPLES
) -> np.ndarray:
    """Fixed-length window of ``n_samples`` starting at ``onset`` (zero-padded
    if the record is too short to hold the full window)."""
    end = onset + n_samples
    window = x[onset:end]
    if len(window) < n_samples:
        window = np.pad(window, (0, n_samples - len(window)))
    return window


def propagated_arrival_index(
    ch2: np.ndarray,
    actuation_onset: int,
    threshold_ratio: float = ARRIVAL_THRESHOLD_RATIO,
) -> int:
    """Sample at which the propagated wave packet arrives at the receiver.

    Searched on the filtered receiver channel, from far enough past the
    actuation burst that the inter-channel crosstalk has decayed, and taken as
    the first sample whose envelope clears ``threshold_ratio`` of the largest
    envelope value inside the search window. The window stops before the late
    reflections that dominate the record's energy.
    """
    filtered = bandpass_filter(ch2)
    envelope = np.abs(filtered)
    start = actuation_onset + ARRIVAL_SEARCH_SKIP_SAMPLES
    end = min(actuation_onset + ARRIVAL_SEARCH_END_SAMPLES, len(envelope))
    if start >= end:
        return actuation_onset
    segment = envelope[start:end]
    above = np.flatnonzero(segment >= threshold_ratio * segment.max())
    return start + int(above[0]) if above.size else start


def preprocess_received_signal(ch2: np.ndarray, onset: int) -> np.ndarray:
    """Filter the raw received signal and extract its four-cycle feature window
    at a pre-computed ``onset`` sample index (see :func:`detect_onset`)."""
    filtered = bandpass_filter(ch2)
    return extract_window(filtered, onset)


def compute_raw_features(
    window: np.ndarray,
    reference_window: np.ndarray,
    fs: float = SAMPLING_FREQUENCY_HZ,
    target_freq: float = TARGET_FREQUENCY_HZ,
) -> np.ndarray:
    """The four un-normalized features (Eqs. 5-8 of the paper) for one window.

    ``reference_window`` is the equivalent window from the undamaged/baseline
    signal (used by the orthogonality metric, Eq. 7).
    """
    n = len(window)
    # Eq. 5 is labelled "root mean square" but its printed form carries no
    # square root, while Eq. 6 does. The root is applied here: without it the
    # feature is a mean square, and since the SVR runs at a fixed kernel width
    # of 1.0 the resulting change of scale is not cosmetic.
    feature1_rms = np.sqrt(np.mean(window**2))
    feature2_std = np.std(window)
    # Eq. 7 prints its denominator without a square root, which would leave the
    # metric dimensionally inconsistent. It is read here as the cosine between
    # the two windows, which is what "orthogonality" requires.
    denom = np.sqrt(np.sum(window**2) * np.sum(reference_window**2))
    feature3_orthogonality = np.sum(window * reference_window) / denom if denom else 0.0
    spectrum = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    k300 = int(np.argmin(np.abs(freqs - target_freq)))
    feature4_mag300 = spectrum[k300]
    return np.array(
        [feature1_rms, feature2_std, feature3_orthogonality, feature4_mag300]
    )


def normalize_features(raw: np.ndarray, baseline_raw: np.ndarray) -> np.ndarray:
    """Divide each feature by the same feature computed on the baseline window."""
    baseline_raw = np.where(baseline_raw == 0, 1.0, baseline_raw)
    return raw / baseline_raw


class FeatureExtractor:
    """Computes the 4-feature vector for one specimen cycle, averaged over the
    two recorded sensor paths (``signal_1`` and ``signal_2``)."""

    def __init__(self, onset_threshold_ratio: float = 0.2) -> None:
        self.onset_threshold_ratio = onset_threshold_ratio

    def cycle_windows(self, signals: dict) -> dict[str, np.ndarray]:
        windows = {}
        for path, df in signals.items():
            ch1, ch2 = df["ch1"].to_numpy(), df["ch2"].to_numpy()
            actuation = detect_onset(ch1, self.onset_threshold_ratio)
            arrival = propagated_arrival_index(ch2, actuation)
            windows[path] = preprocess_received_signal(ch2, arrival)
        return windows

    def cycle_features(
        self, signals: dict, baseline_signals: dict
    ) -> np.ndarray:
        """Normalized 4-feature vector for a cycle, averaged across sensor paths."""
        windows = self.cycle_windows(signals)
        baseline_windows = self.cycle_windows(baseline_signals)
        per_path = []
        for path in windows:
            raw = compute_raw_features(windows[path], baseline_windows[path])
            baseline_raw = compute_raw_features(
                baseline_windows[path], baseline_windows[path]
            )
            per_path.append(normalize_features(raw, baseline_raw))
        return np.mean(per_path, axis=0)
