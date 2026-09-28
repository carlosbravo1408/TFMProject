from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

SAMPLING_FREQUENCY_HZ = 20e6
DT_S = 1 / SAMPLING_FREQUENCY_HZ

BANDPASS_LOW_HZ = 150e3
BANDPASS_HIGH_HZ = 350e3
BANDPASS_ORDER = 5

# t = 85 us, the typical actuator peak in the raw records.
REFERENCE_ACTUATOR_PEAK_INDEX = 1700

# Figure 4 prints 10.6-12.6 "us"; read in units of 1e-5 s, which is where the
# first propagated packet sits.
S0_WINDOW_START_S = 106e-6
S0_WINDOW_END_S = 126e-6


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


def phase_alignment_shift(
    ch1: np.ndarray,
    reference_ch1: np.ndarray | None = None,
    reference_index: int = REFERENCE_ACTUATOR_PEAK_INDEX,
) -> int:
    if reference_ch1 is None:
        return reference_index - int(np.argmax(ch1))
    # Cross-correlation instead of argmax, which jumps by a carrier period when the
    # burst has two near-equal peaks.
    corr = np.correlate(ch1, reference_ch1, mode="full")
    lag = int(np.argmax(corr)) - (len(reference_ch1) - 1)
    return (reference_index - int(np.argmax(reference_ch1))) - lag


# The geometry is fixed within a specimen; only T3's cycle 50000 locks onto a
# wrong lobe, 196 samples away.
SHIFT_CONSENSUS_TOLERANCE = 5


def consensus_shift(shifts: list[int], tolerance: int = SHIFT_CONSENSUS_TOLERANCE) -> tuple[int, list[int]]:
    if not shifts:
        return 0, []
    consensus = int(np.median(shifts))
    return consensus, [consensus if abs(s - consensus) > tolerance else s for s in shifts]


def apply_shift(x: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(x)
    if shift >= 0:
        out[shift:] = x[: len(x) - shift] if shift else x
    else:
        out[:shift] = x[-shift:]
    return out


class SignalPreprocessor:
    def __init__(
        self,
        window_start_s: float = S0_WINDOW_START_S,
        window_end_s: float = S0_WINDOW_END_S,
        reference_index: int = REFERENCE_ACTUATOR_PEAK_INDEX,
    ) -> None:
        self.window_start_s = window_start_s
        self.window_end_s = window_end_s
        self.reference_index = reference_index
        self.reference_ch1: np.ndarray | None = None

    def set_reference(self, ch1: np.ndarray) -> None:
        self.reference_ch1 = np.asarray(ch1, dtype=float)

    def record_shift(self, ch1: np.ndarray) -> int:
        return phase_alignment_shift(ch1, self.reference_ch1, self.reference_index)

    def preprocess(self, ch1: np.ndarray, ch2: np.ndarray, shift: int | None = None) -> np.ndarray:
        if shift is None:
            shift = self.record_shift(ch1)
        return apply_shift(bandpass_filter(ch2), shift)

    def window_slice(self, n_samples: int) -> slice:
        i0 = int(round(self.window_start_s / DT_S))
        i1 = int(round(self.window_end_s / DT_S))
        return slice(max(i0, 0), min(i1, n_samples))

    def s0_window(self, ch1: np.ndarray, ch2: np.ndarray, shift: int | None = None) -> np.ndarray:
        aligned = self.preprocess(ch1, ch2, shift)
        return aligned[self.window_slice(len(aligned))]

    def window_times_s(self, n_samples: int) -> np.ndarray:
        sl = self.window_slice(n_samples)
        return np.arange(sl.start, sl.stop) * DT_S
