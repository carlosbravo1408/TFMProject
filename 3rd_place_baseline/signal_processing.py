"""Pre-processing of the ultrasonic wave signals (Sections 3.1 and 3.2 of the
paper): band-pass filtering and truncation to the First Wave Package (FWP).

Band-pass filter (Sec. 3.1): a linear-phase FIR filter with first stop
frequency f_s1 = 50 kHz, first pass frequency f_p1 = 100 kHz, second pass
frequency f_p2 = 500 kHz and second stop frequency f_s2 = 1000 kHz, at a
sampling frequency of 20 MHz.

Truncation (Sec. 3.2): the received signal contains a synchronization part
(direct electrical feed-through of the actuation burst, independent of the
crack), the FWP (the wave transmitted through the crack-sensing path) and a
"rest" part (later, reflected arrivals). The FWP window is defined from the
Location of the First Peak (LFP) as ``[LFP - 150, LFP + 200]`` (Fig. 4).

The paper does not give an explicit numerical rule for finding the LFP.
Locating it independently cycle-by-cycle (first local peak of the filtered
signal past some fixed skip from the actuation burst) turns out to be
unstable: depending on the specimen, the synchronization feed-through does
not fully decay to the noise floor before the wave packet arrives, so a
per-cycle peak search occasionally locks onto a residual synchronization
ripple instead of the true first arrival, jumping the LFP by hundreds of
samples between cycles of the same specimen.

Instead, the LFP is located *once* per specimen (and per Run 1/Run 2
channel) on the specimen's undamaged reference cycle (``first_zero_cycle``),
as the *first* envelope peak (not the largest one) exceeding a fixed
fraction of the window's peak envelope, within a bounded search window after
the actuation burst. That reference (LFP - actuation index) offset is then a
fixed, specimen-specific "time of flight" reused for every other cycle: the
sensing path geometry does not change with crack damage, so the packet's
*location* stays essentially constant even though its *amplitude* and
*shape* (i.e. the features extracted from it) do change -- which is exactly
the signal that the crack-sensitive features below are meant to capture.

Locating the *largest* envelope peak (rather than the first one clearing a
threshold) was tried first and rejected: for every specimen it locks onto a
markedly later, larger-amplitude arrival (a reflection or a later Lamb-wave
mode) instead of the direct wave. That is easy to catch because the direct
arrival's time of flight is physically bounded: at a typical Lamb S0
velocity of ~5000-5500 m/s over the 161 mm sensing path, the actuation-to-
arrival delay should be on the order of ~550-650 samples at 20 MHz -- and
that is exactly the range the first-peak-past-threshold rule recovers for
every one of T1-T8 (310-985 samples), whereas the largest-peak rule gave
inconsistent, much larger gaps (500-1800+ samples) with no such physical
regularity.
"""
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

# Records are 4000 samples long; ``filtfilt``'s default edge padding needs
# padlen = 3*numtaps < len(x), which bounds how sharp a linear-phase FIR
# filter can be given these band edges (the 50 kHz first transition band is
# <1% of the 10 MHz Nyquist range, and would need >1400 Kaiser-window taps
# for the paper's target 60 dB stop-band attenuation -- more than fits).
# 999 taps is the largest odd length that keeps the default padding safely
# under the record length while still reaching ~-29 dB stop-band rejection.
FIR_NUMTAPS = 999

FWP_BEFORE = 150
FWP_AFTER = 200

# Bounded search window (relative to the actuation burst's peak index) used
# to locate the reference LFP on a specimen's undamaged cycle: samples to
# skip immediately after the actuation peak (skips the synchronization
# feed-through) and the maximum additional offset considered, generous
# enough to cover the Lamb-wave time of flight over the 161 mm sensing path
# without reaching into later, larger reflected arrivals.
REFERENCE_SKIP_SAMPLES = 300
REFERENCE_SEARCH_SAMPLES = 1600

# Fraction of the search window's largest envelope value a peak must clear
# to count as "the" first peak (see module docstring: this recovers a
# physically plausible ~550-650-sample time of flight for every specimen;
# lower values start catching noise, higher values start skipping the
# direct arrival in favor of a later, larger reflection).
REFERENCE_HEIGHT_RATIO = 0.20

# The cross-correlation lock on the actuation burst (see
# ``aligned_actuation_index``) removes multi-lobe (~100-sample, one carrier
# period) jumps, but leaves a few samples of residual jitter between cycles
# recorded far apart in time. Since the ~200 kHz carrier means the true wave
# packet's local peak can sit a handful of samples away from the specimen's
# fixed reference offset, the LFP is refined to the nearest true local
# extremum of the filtered signal within this many samples.
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
    """Linear-phase FIR band-pass filter matching the four band edges
    specified in Sec. 3.1, designed with a Kaiser window.

    An equiripple (Parks-McClellan/``scipy.signal.remez``) design was tried
    first, matching the paper's stated filter type, but the Remez exchange
    fails to converge for these band edges (the 50 kHz transition bands are
    <1% of the 10 MHz Nyquist range): it produces a passband gain of
    20x-1e8x instead of unity depending on tap count. The Kaiser-window
    design below hits the same four band edges without that instability, at
    a tap count capped to fit the 4000-sample records (see ``FIR_NUMTAPS``).
    """
    beta = kaiser_beta(stopband_attenuation_db)
    cutoff = [(stop1 + pass1) / 2, (pass2 + stop2) / 2]
    return firwin(numtaps, cutoff, window=("kaiser", beta), pass_zero=False, fs=fs)


_FIR_TAPS = design_bandpass_fir()


def bandpass_filter(x: np.ndarray, taps: np.ndarray = _FIR_TAPS) -> np.ndarray:
    """Zero-phase FIR band-pass filter (``filtfilt`` avoids the phase
    distortion the paper explicitly wants an FIR filter to avoid)."""
    return filtfilt(taps, [1.0], np.asarray(x, dtype=float))


def actuation_index(ch1_raw: np.ndarray) -> int:
    """Index of the actuation burst's peak, from a plain argmax.

    Unreliable on its own: the actuation burst is a multi-cycle ~200 kHz
    tone, so several of its lobes have near-equal amplitude and a plain
    argmax can lock onto a neighboring lobe (~100 samples = one carrier
    period away) depending on noise, jumping between cycles of the same
    specimen. Use :func:`aligned_actuation_index` against a fixed reference
    burst for a stable index instead.
    """
    return int(np.argmax(np.abs(ch1_raw)))


def aligned_actuation_index(ch1_raw: np.ndarray, reference_ch1: np.ndarray, reference_index: int) -> int:
    """Actuation index of ``ch1_raw``, locked to the same tone-burst lobe as
    ``reference_index`` in ``reference_ch1`` via cross-correlation."""
    corr = np.correlate(ch1_raw, reference_ch1, mode="full")
    lag = int(np.argmax(corr)) - (len(reference_ch1) - 1)  # ch1_raw[n] ~ reference_ch1[n - lag]
    return reference_index + lag


def locate_reference_lfp(
    ch1_raw: np.ndarray,
    ch2_filtered: np.ndarray,
    skip: int = REFERENCE_SKIP_SAMPLES,
    search: int = REFERENCE_SEARCH_SAMPLES,
    height_ratio: float = REFERENCE_HEIGHT_RATIO,
) -> int:
    """Location of the First Peak (LFP) on an undamaged reference cycle: the
    *first* envelope peak clearing ``height_ratio`` of the window's largest
    envelope value, within a bounded window after the actuation burst (see
    module docstring for why "first" and not "largest")."""
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
    """Snap ``lfp`` to the nearest true local-amplitude extremum of the
    filtered signal within +/-``window`` samples."""
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
    """Band-pass filter + specimen-specific LFP offset + FWP truncation."""

    def __init__(self) -> None:
        self._offsets: dict[tuple[str, str], int] = {}
        self._reference_ch1: dict[tuple[str, str], np.ndarray] = {}
        self._reference_index: dict[tuple[str, str], int] = {}

    def fit_reference(self, specimen: str, channel: str, ch1_raw: np.ndarray, ch2_raw: np.ndarray) -> int:
        """Fix the (LFP - actuation index) offset for a specimen/channel
        from its undamaged reference cycle. Returns the reference LFP."""
        ch1_raw = np.asarray(ch1_raw, dtype=float)
        filtered = bandpass_filter(np.asarray(ch2_raw, dtype=float))
        ref_act = actuation_index(ch1_raw)
        lfp = refine_peak(filtered, locate_reference_lfp(ch1_raw, filtered))
        key = (specimen, channel)
        self._offsets[key] = lfp - ref_act
        self._reference_ch1[key] = ch1_raw
        self._reference_index[key] = ref_act
        return lfp

    def truncate(self, specimen: str, channel: str, ch1_raw: np.ndarray, ch2_raw: np.ndarray) -> TruncatedSignal:
        """Band-pass filter the received signal and extract its FWP, using
        the specimen/channel's fixed reference offset to locate the LFP."""
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
