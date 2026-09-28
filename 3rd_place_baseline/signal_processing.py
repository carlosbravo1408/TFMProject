from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import filtfilt, find_peaks, firwin, hilbert, kaiser_beta

SAMPLING_FREQUENCY_HZ = 20e6

STOP1_HZ = 50e3
PASS1_HZ = 100e3
PASS2_HZ = 500e3
STOP2_HZ = 1000e3
STOPBAND_ATTENUATION_DB = 60.0

# filtfilt needs 3 * numtaps below the 4000-sample record; 999 taps reach ~-29 dB,
# while the paper's 60 dB would need more than 1400.
FIR_NUMTAPS = 999

FWP_BEFORE = 150
FWP_AFTER = 200

# Skips the synchronization feed-through and stops before later reflections.
REFERENCE_SKIP_SAMPLES = 300
REFERENCE_SEARCH_SAMPLES = 1600

# The first peak above this fraction, not the largest one, which locks onto a
# later reflection on every specimen.
REFERENCE_HEIGHT_RATIO = 0.20

# Residual jitter between cycles recorded far apart in time.
LFP_REFINE_WINDOW = 25


def design_bandpass_fir(
    fs: float = SAMPLING_FREQUENCY_HZ,
    stop1: float = STOP1_HZ,
    pass1: float = PASS1_HZ,
    pass2: float = PASS2_HZ,
    stop2: float = STOP2_HZ,
    numtaps: int = FIR_NUMTAPS,
    stopband_attenuation_db: float = STOPBAND_ATTENUATION_DB,
) -> np.ndarray:
    # Kaiser window instead of the paper's equiripple design: remez does not converge
    # for these band edges.
    beta = kaiser_beta(stopband_attenuation_db)
    cutoff = [(stop1 + pass1) / 2, (pass2 + stop2) / 2]
    return firwin(numtaps, cutoff, window=("kaiser", beta), pass_zero=False, fs=fs)


_FIR_TAPS = design_bandpass_fir()


def bandpass_filter(x: np.ndarray, taps: np.ndarray = _FIR_TAPS) -> np.ndarray:
    return filtfilt(taps, [1.0], np.asarray(x, dtype=float))


def actuation_index(ch1_raw: np.ndarray) -> int:
    return int(np.argmax(np.abs(ch1_raw)))


def aligned_actuation_index(ch1_raw: np.ndarray, reference_ch1: np.ndarray, reference_index: int) -> int:
    corr = np.correlate(ch1_raw, reference_ch1, mode="full")
    lag = int(np.argmax(corr)) - (len(reference_ch1) - 1)
    return reference_index + lag


def locate_reference_lfp(
    ch1_raw: np.ndarray,
    ch2_filtered: np.ndarray,
    skip: int = REFERENCE_SKIP_SAMPLES,
    search: int = REFERENCE_SEARCH_SAMPLES,
    height_ratio: float = REFERENCE_HEIGHT_RATIO,
) -> int:
    act = actuation_index(ch1_raw)
    start = min(act + skip, len(ch2_filtered) - 1)
    stop = min(start + search, len(ch2_filtered))
    envelope = np.abs(hilbert(ch2_filtered[start:stop]))
    if envelope.max() == 0:
        return start
    peaks, _ = find_peaks(envelope, height=height_ratio * envelope.max())
    idx = int(peaks[0]) if len(peaks) else int(np.argmax(envelope))
    return start + idx


def refine_peak(ch2_filtered: np.ndarray, lfp: int, window: int = LFP_REFINE_WINDOW) -> int:
    lo, hi = max(lfp - window, 0), min(lfp + window + 1, len(ch2_filtered))
    if hi <= lo:
        return lfp
    return lo + int(np.argmax(np.abs(ch2_filtered[lo:hi])))


def fwp_slice(lfp: int, n_samples: int, before: int = FWP_BEFORE, after: int = FWP_AFTER) -> slice:
    return slice(max(lfp - before, 0), min(lfp + after, n_samples))


@dataclass
class TruncatedSignal:
    lfp: int
    fwp: np.ndarray
    filtered: np.ndarray


class SignalPreprocessor:
    def __init__(self) -> None:
        self._offsets: dict[tuple[str, str], int] = {}
        self._reference_ch1: dict[tuple[str, str], np.ndarray] = {}
        self._reference_index: dict[tuple[str, str], int] = {}

    def fit_reference(self, specimen: str, channel: str, ch1_raw: np.ndarray, ch2_raw: np.ndarray) -> int:
        ch1_raw = np.asarray(ch1_raw, dtype=float)
        filtered = bandpass_filter(np.asarray(ch2_raw, dtype=float))
        # The LFP offset is fixed once per specimen: a per-cycle search locks onto
        # residual feed-through ripple.
        ref_act = actuation_index(ch1_raw)
        lfp = refine_peak(filtered, locate_reference_lfp(ch1_raw, filtered))
        key = (specimen, channel)
        self._offsets[key] = lfp - ref_act
        self._reference_ch1[key] = ch1_raw
        self._reference_index[key] = ref_act
        return lfp

    def truncate(self, specimen: str, channel: str, ch1_raw: np.ndarray, ch2_raw: np.ndarray) -> TruncatedSignal:
        key = (specimen, channel)
        if key not in self._offsets:
            raise KeyError(f"No reference offset fitted for {specimen}/{channel}; call fit_reference first")
        ch1_raw = np.asarray(ch1_raw, dtype=float)
        ch2_raw = np.asarray(ch2_raw, dtype=float)
        filtered = bandpass_filter(ch2_raw)
        act = aligned_actuation_index(ch1_raw, self._reference_ch1[key], self._reference_index[key])
        lfp = refine_peak(filtered, act + self._offsets[key])
        sl = fwp_slice(lfp, len(filtered))
        return TruncatedSignal(lfp=lfp, fwp=filtered[sl], filtered=filtered)
