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
)

TARGET_FREQUENCY_HZ = 300e3

# The onset on ch1 marks the emission, not the arrival: a window anchored there
# lands on the inter-channel crosstalk (75-96 us), while the S0 packet arrives
# from 100 us onwards.
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
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def detect_onset(x: np.ndarray, threshold_ratio: float = 0.2) -> int:
    envelope = np.abs(x)
    peak = envelope.max()
    if peak == 0:
        return 0
    above = np.where(envelope >= threshold_ratio * peak)[0]
    return int(above[0]) if len(above) else 0


def extract_window(
    x: np.ndarray, onset: int, n_samples: int = FEATURE_WINDOW_SAMPLES
) -> np.ndarray:
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
    filtered = bandpass_filter(ch2)
    return extract_window(filtered, onset)


def compute_raw_features(
    window: np.ndarray,
    reference_window: np.ndarray,
    fs: float = SAMPLING_FREQUENCY_HZ,
    target_freq: float = TARGET_FREQUENCY_HZ,
) -> np.ndarray:
    n = len(window)
    # Eq. 5 is printed without the square root its name implies.
    feature1_rms = np.sqrt(np.mean(window**2))
    feature2_std = np.std(window)
    # Eq. 7 is printed without the root in the denominator; read as a cosine.
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
    baseline_raw = np.where(baseline_raw == 0, 1.0, baseline_raw)
    return raw / baseline_raw


class FeatureExtractor:
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
