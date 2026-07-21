"""Pre-processing of the Lamb wave signals (Section 3.2.1 of Kong et al. 2020).

Two pre-processing techniques are applied to the raw signals:

1. **Band-pass filter** — a Butterworth band-pass over the 150-350 kHz band
   removes noise outside the tone-burst band (the excitation centre frequency
   is ~200 kHz). After filtering, the two repeated measurements are nearly
   identical, so only ``signal_1`` is used.
2. **Phase alignment** — the piezoelectric sensor pairs differ between
   specimens, producing phase differences. Based on the maximum value of the
   actuator signal (``ch1``), every record is time-shifted so its actuator
   peak lands on a common reference index. This alignment is applied to all
   actuator and receiver signals.

After alignment, the principal S0 wave packet of the received signal is
separated from later arrivals in a fixed time range (Figure 4 of the paper);
features are extracted from that window.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

SAMPLING_FREQUENCY_HZ = 20e6
DT_S = 1 / SAMPLING_FREQUENCY_HZ

BANDPASS_LOW_HZ = 150e3
BANDPASS_HIGH_HZ = 350e3
BANDPASS_ORDER = 5

# Common reference index for the actuator peak after phase alignment. 1700
# corresponds to t = 85 us, the typical location of the actuator maximum in
# the raw records, so aligned times remain comparable to the raw time axis.
REFERENCE_ACTUATOR_PEAK_INDEX = 1700

# Time range holding the principal S0 mode of the received wave after
# alignment. Figure 4 of the paper compares the pre-processed T4 signals over
# 106-126 us (its axis reads 10.6-12.6 in 1e-5 s), where the separation
# between cycles is noticeable; the default window is that range slightly
# widened, which keeps the same wave packet while making the envelope-peak
# phase-delay feature more robust at the packet edges.
S0_WINDOW_START_S = 100e-6
S0_WINDOW_END_S = 130e-6


def bandpass_filter(
    x: np.ndarray,
    fs: float = SAMPLING_FREQUENCY_HZ,
    low: float = BANDPASS_LOW_HZ,
    high: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, x)


def phase_alignment_shift(
    ch1: np.ndarray,
    reference_ch1: np.ndarray | None = None,
    reference_index: int = REFERENCE_ACTUATOR_PEAK_INDEX,
) -> int:
    """Number of samples to shift a record so its actuator burst is phase
    aligned with a common reference.

    When ``reference_ch1`` is given, the shift is found by cross-correlating
    the two actuator signals (and the reference burst's own maximum is placed
    at ``reference_index``). This realizes the paper's alignment 'based on
    the maximum value of the actuator signals' while being robust to the
    actuator burst having two near-equal peaks, in which case a plain argmax
    jumps by a carrier period between records. Without a reference, the plain
    argmax rule is used."""
    if reference_ch1 is None:
        return reference_index - int(np.argmax(ch1))
    corr = np.correlate(ch1, reference_ch1, mode="full")
    lag = int(np.argmax(corr)) - (len(reference_ch1) - 1)  # ch1 ~ ref delayed by lag
    return (reference_index - int(np.argmax(reference_ch1))) - lag


def apply_shift(x: np.ndarray, shift: int) -> np.ndarray:
    """Shift a signal by ``shift`` samples (positive = delay), zero-padding."""
    out = np.zeros_like(x)
    if shift >= 0:
        out[shift:] = x[: len(x) - shift] if shift else x
    else:
        out[:shift] = x[-shift:]
    return out


class SignalPreprocessor:
    """Band-pass filter + phase alignment + S0-window extraction."""

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
        """Fix the common actuator reference used for phase alignment across
        all specimens (any record works; the excitation burst is shared)."""
        self.reference_ch1 = np.asarray(ch1, dtype=float)

    def preprocess(self, ch1: np.ndarray, ch2: np.ndarray) -> np.ndarray:
        """Filtered, phase-aligned received signal (full record)."""
        shift = phase_alignment_shift(ch1, self.reference_ch1, self.reference_index)
        return apply_shift(bandpass_filter(ch2), shift)

    def window_slice(self, n_samples: int) -> slice:
        i0 = int(round(self.window_start_s / DT_S))
        i1 = int(round(self.window_end_s / DT_S))
        return slice(max(i0, 0), min(i1, n_samples))

    def s0_window(self, ch1: np.ndarray, ch2: np.ndarray) -> np.ndarray:
        """Pre-processed received signal restricted to the S0 time range."""
        aligned = self.preprocess(ch1, ch2)
        return aligned[self.window_slice(len(aligned))]

    def window_times_s(self, n_samples: int) -> np.ndarray:
        sl = self.window_slice(n_samples)
        return np.arange(sl.start, sl.stop) * DT_S
