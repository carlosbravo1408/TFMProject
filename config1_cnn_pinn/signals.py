from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "PHMDC2019_Data"

SAMPLING_HZ = 20e6
N_SAMPLES = 4000
EXCITATION_HZ = 200e3
BAND_HZ = (100e3, 400e3)

# The S0 packet sits at 100-130 us; widened so the convolutions find their own edges.
WINDOW_S = (90e-6, 145e-6)
WINDOW_SLICE = slice(int(round(WINDOW_S[0] * SAMPLING_HZ)),
                     int(round(WINDOW_S[1] * SAMPLING_HZ)))
WINDOW_SAMPLES = WINDOW_SLICE.stop - WINDOW_SLICE.start

TRAINING_SPECIMENS = ("T1", "T2", "T3", "T4", "T5", "T6")
VALIDATION_SPECIMENS = ("T7", "T8")

REFERENCE_CYCLE_OVERRIDE = {"T8": 50000}

# T8 @ 40000 correlates 0.27-0.48 with every other T8 record (0.64-0.99 among
# themselves) yet is labelled with a zero crack.
EXCLUDED_RECORDS = {("T8", 40000)}


def _specimen_dir(root: Path, name: str) -> Path:
    group = "training" if name in TRAINING_SPECIMENS else "validation"
    return root / group / name


def signal_cycles(root: Path | str = DEFAULT_ROOT, name: str = "T1") -> list[int]:
    base = _specimen_dir(Path(root), name)
    return sorted(
        int(p.name) for p in base.iterdir()
        if p.is_dir() and p.name.isdigit()
        and any((p / f"signal_{i}.csv").is_file() for i in (1, 2))
    )


def available_repetitions(root: Path | str, name: str, cycle: int) -> list[int]:
    base = _specimen_dir(Path(root), name) / str(cycle)
    return [i for i in (1, 2) if (base / f"signal_{i}.csv").is_file()]


def _bandpass_sos(low=BAND_HZ[0], high=BAND_HZ[1], order=4):
    nyq = SAMPLING_HZ / 2
    return butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")


@lru_cache(maxsize=1)
def _sos():
    return _bandpass_sos()


def _apply_shift(x: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(x)
    if shift > 0:
        out[shift:] = x[: len(x) - shift]
    elif shift < 0:
        out[:shift] = x[-shift:]
    else:
        out[:] = x
    return out


@lru_cache(maxsize=256)
def _load_raw(name: str, cycle: int, rep: int, root: str | None):
    root_path = Path(root) if root is not None else DEFAULT_ROOT
    path = _specimen_dir(root_path, name) / str(cycle) / f"signal_{rep}.csv"
    frame = pd.read_csv(path, usecols=["ch1", "ch2"])
    ch1 = frame["ch1"].to_numpy(dtype=np.float64)
    ch2 = frame["ch2"].to_numpy(dtype=np.float64)
    if len(ch2) < N_SAMPLES:
        ch1 = np.pad(ch1, (0, N_SAMPLES - len(ch1)))
        ch2 = np.pad(ch2, (0, N_SAMPLES - len(ch2)))
    return ch1[:N_SAMPLES], ch2[:N_SAMPLES]


def raw_waveform(name: str, cycle: int, rep: int = 1, root: str | None = None,
                 channel: str = "ch2") -> np.ndarray:
    ch1, ch2 = _load_raw(name, cycle, rep, root)
    return ch1 if channel == "ch1" else ch2


@lru_cache(maxsize=256)
def load_waveform(name: str, cycle: int, rep: int = 1, root: str | None = None) -> np.ndarray:
    ch1, ch2 = _load_raw(name, cycle, rep, root)
    ref_cycle = reference_cycle(name, root)
    if cycle != ref_cycle or rep != 1:
        ref_ch1, _ = _load_raw(name, ref_cycle, 1, root)
        # ch1 carries only trigger jitter; cross-correlation instead of argmax because the
        # burst has two near-equal peaks.
        corr = np.correlate(ch1, ref_ch1, mode="full")
        lag = int(np.argmax(corr)) - (len(ref_ch1) - 1)
        ch2 = _apply_shift(ch2, -lag)
    filtered = sosfiltfilt(_sos(), ch2)
    return filtered[WINDOW_SLICE].astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    sd = float(np.std(x))
    return (x - float(np.mean(x))) / (sd if sd > 1e-12 else 1.0)


def reference_cycle(name: str, root: str | None = None) -> int:
    if name in REFERENCE_CYCLE_OVERRIDE:
        return REFERENCE_CYCLE_OVERRIDE[name]
    return signal_cycles(Path(root) if root else DEFAULT_ROOT, name)[0]


@lru_cache(maxsize=16)
def reference_waveform(name: str, root: str | None = None) -> np.ndarray:
    # Repetition 1 for every reference: T1 @ 50000 has incompatible repetitions.
    return zscore(load_waveform(name, reference_cycle(name, root), 1, root))


@lru_cache(maxsize=16)
def difference_scale(name: str, root: str | None = None,
                     reference_crack_mm: float = 2.0) -> float:
    from physics_calibration.data import load_curve

    root_path = Path(root) if root is not None else DEFAULT_ROOT
    ref = reference_waveform(name, root)
    curve = load_curve(name)
    usable = [c for c in signal_cycles(root_path, name)
              if (name, c) not in EXCLUDED_RECORDS and c in set(curve.cycles)]
    if not usable:
        return 1.0

    cracks = {int(c): float(a) for c, a in zip(curve.cycles, curve.crack_mm)}
    # Read at a common damage state: a per-specimen maximum is biased low for T7/T8,
    # whose signals stop early.
    at_reference = [c for c in usable if cracks[c] >= reference_crack_mm]
    anchor = min(at_reference) if at_reference else min(usable, key=lambda c: cracks[c])

    values = [
        float(np.sqrt(np.mean((zscore(load_waveform(name, anchor, rep, root)) - ref) ** 2)))
        for rep in available_repetitions(root_path, name, anchor)
    ]
    scale = float(np.mean(values)) if values else 1.0
    return scale if scale > 1e-9 else 1.0


def encode(name: str, cycle: int, rep: int = 1, root: str | None = None,
           normalise_difference: bool = True) -> np.ndarray:
    current = zscore(load_waveform(name, cycle, rep, root))
    difference = current - reference_waveform(name, root)
    if normalise_difference:
        difference = difference / difference_scale(name, root)
    return np.stack([current, difference]).astype(np.float32)


@dataclass(frozen=True)
class Sample:
    specimen: str
    cycle: int
    rep: int
    crack_mm: float
    is_reference: bool

    def features(self, root: str | None = None) -> np.ndarray:
        return encode(self.specimen, self.cycle, self.rep, root)


def build_samples(
    crack_labels: dict[str, pd.DataFrame],
    specimens=TRAINING_SPECIMENS + VALIDATION_SPECIMENS,
    root: str | None = None,
    include_zero: bool = True,
) -> list[Sample]:
    root_path = Path(root) if root is not None else DEFAULT_ROOT
    out: list[Sample] = []
    for name in specimens:
        table = crack_labels[name]
        labels = dict(zip(table["cycle"], table["crack_length_mm"]))
        ref = reference_cycle(name, root)
        for cycle in signal_cycles(root_path, name):
            if cycle not in labels or (name, cycle) in EXCLUDED_RECORDS:
                continue
            crack = float(labels[cycle])
            if crack == 0.0 and not include_zero:
                continue
            for rep in available_repetitions(root_path, name, cycle):
                out.append(Sample(name, cycle, rep, crack, cycle == ref))
    return out


FEATURE_NAMES = (
    "rms_diferencial", "max_diferencial", "correlacion",
    *[f"rms_ventana_{i}" for i in range(8)],
    "centroide_temporal", "retardo_fase_us",
)


def scalar_features(name: str, cycle: int, rep: int = 1, root: str | None = None) -> np.ndarray:
    # The reference maximum is left out: constant per specimen, it would act as an ID.
    par = encode(name, cycle, rep, root)
    actual, diferencia = par[0], par[1]
    referencia = actual - diferencia * difference_scale(name, root)

    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    ancho = max(1, len(diferencia) // 8)
    energia = np.square(diferencia)
    centroide = float((np.arange(len(energia)) * energia).sum() / max(energia.sum(), 1e-12))
    correlacion = np.correlate(actual, referencia, mode="full")
    retardo = (int(np.argmax(correlacion)) - (len(referencia) - 1)) / SAMPLING_HZ * 1e6

    return np.array([
        rms(diferencia),
        float(np.max(np.abs(diferencia))),
        float(np.corrcoef(actual, referencia)[0, 1]),
        *[rms(diferencia[i * ancho:(i + 1) * ancho]) for i in range(8)],
        centroide / len(energia),
        retardo,
    ], dtype=np.float32)
