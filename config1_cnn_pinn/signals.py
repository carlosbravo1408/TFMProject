"""Lamb-wave signal loading and preprocessing for the 1D-CNN encoder.

Design constraints, all forced by the data rather than chosen:

*Only 91 waveform files exist in the whole release* (46 measurement cycles x 2
repetitions), of which ~37 cycles carry a crack-length label. That is three
orders of magnitude less data than a 1D-CNN normally needs, and it dictates
every decision here:

1. **Differential input.** Every winning entry of the challenge worked on the
   *change* of the waveform relative to the specimen's own undamaged
   measurement, not on its absolute shape — the inter-specimen variability of
   PZT bonding and rivet fit-up dwarfs the damage signature. The encoder
   therefore receives two channels: the normalised waveform at the current
   cycle and its difference against the specimen's reference waveform. This is
   the one piece of physical prior knowledge injected by hand; everything else
   (time of flight, spectral energy, phase shift) is left for the convolutions
   to discover.
2. **Per-signal z-score.** Standardising each waveform removes the
   PZT-coupling gain, which drifts between specimens and even between cycles as
   the transducer partially debonds. This is the control for confounder 1 of
   the TFM's Research Gap 1.
3. **Band-pass around the excitation.** ``ch1`` is a 200 kHz tone burst; the
   received ``ch2`` carries its dispersed echoes. Everything outside
   ~100-400 kHz is electrical noise, and with 74 training waveforms the network
   cannot be asked to learn that by itself.

Dataset gotchas honoured here (see CLAUDE.md):
  - T8 @ 40000 is anomalous: its windowed waveform correlates only 0.27-0.48
    with every other T8 record, while those correlate 0.64-0.99 among
    themselves. (The "~10 us out of phase" previously quoted here is *not*
    reproducible: cross-correlating ch2 gives -0.65 us and an envelope-threshold
    arrival gives +-0.75 us against three of the four remaining records.) Cycle
    50000 is used as T8's undamaged reference instead.
  - T1 @ 50000 has mutually incompatible signal_1 / signal_2, so ``signal_1``
    is used for every reference, uniformly, rather than averaging repetitions.
  - training/T5/42000 is an empty folder.
"""
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

# Time range fed to the encoder, in seconds. The principal S0 packet of the
# received wave sits at 100-130 us (the window the 2nd-place entry extracts its
# features from); the range is widened to 90-145 us here so the convolutions
# can find their own boundaries instead of inheriting a hand-tuned one, while
# still discarding the ~85 % of the 200 us trace that is pre-arrival silence
# and late reverberation. Feeding the whole trace measurably hurts: with 87
# labelled waveforms, every uninformative sample is capacity spent on noise.
WINDOW_S = (90e-6, 145e-6)
WINDOW_SLICE = slice(int(round(WINDOW_S[0] * SAMPLING_HZ)),
                     int(round(WINDOW_S[1] * SAMPLING_HZ)))
WINDOW_SAMPLES = WINDOW_SLICE.stop - WINDOW_SLICE.start

TRAINING_SPECIMENS = ("T1", "T2", "T3", "T4", "T5", "T6")
VALIDATION_SPECIMENS = ("T7", "T8")

# Undamaged reference cycle per specimen. Defaults to the specimen's first
# measured cycle; T8 is overridden because its 40000 measurement is the known
# anomalous one (this also matches Kong et al.'s Fig. 8d and the 1st-place
# baseline's choice).
REFERENCE_CYCLE_OVERRIDE = {"T8": 50000}

# Records excluded as known-bad rather than as inconvenient. T8 @ 40000 is the
# anomalous acquisition documented in the dataset notes (its windowed waveform
# correlates 0.27-0.48 with every other T8 record, which correlate 0.64-0.99
# among themselves); it is labelled with a zero crack, so leaving it
# in asks the encoder to map a corrupted waveform to "no damage" and it
# reliably predicts several millimetres there. The 1st-place entry discards it
# for the same reason.
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
    """Shift by ``shift`` samples (positive = delay), zero-padding."""
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
    """Traza completa sin filtrar ni ventanear, tal como sale del fichero.

    ``load_waveform`` devuelve ya la ventana del modo S0, que es lo que consume
    la red. Esta funcion existe para poder mostrar el registro entero y situar
    esa ventana dentro de el; no la usa el modelo.
    """
    ch1, ch2 = _load_raw(name, cycle, rep, root)
    return ch1 if channel == "ch1" else ch2


@lru_cache(maxsize=256)
def load_waveform(name: str, cycle: int, rep: int = 1, root: str | None = None) -> np.ndarray:
    """Band-pass-filtered, **phase-aligned** received signal (``ch2``).

    Alignment matters more than anything else in this pipeline. The damage
    signature the encoder has to find is a sub-microsecond delay and a change
    of shape in the transmitted S0 packet; the acquisition's own trigger jitter
    is of the same order, so without alignment the difference channel is
    dominated by jitter and the network learns nothing. The shift is measured
    on the *actuator* trace ``ch1`` — which is the same tone burst in every
    record and therefore carries only jitter, no damage — by cross-correlating
    it against the specimen's reference record, and then applied to ``ch2``.

    Cross-correlating rather than taking the argmax of ``ch1`` is deliberate:
    the burst has two near-equal peaks, and a plain argmax jumps by a whole
    carrier period between records. Same reasoning as
    ``2nd_place_baseline/signal_processing.py``.
    """
    ch1, ch2 = _load_raw(name, cycle, rep, root)
    ref_cycle = reference_cycle(name, root)
    if cycle != ref_cycle or rep != 1:
        ref_ch1, _ = _load_raw(name, ref_cycle, 1, root)
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
    """Standardised undamaged waveform of a specimen (repetition 1)."""
    return zscore(load_waveform(name, reference_cycle(name, root), 1, root))


@lru_cache(maxsize=16)
def difference_scale(name: str, root: str | None = None,
                     reference_crack_mm: float = 2.0) -> float:
    """Differential-channel RMS at a **comparable damage state**, per specimen.

    The obvious choice — the specimen's maximum differential RMS — is wrong
    here, and the error is systematic rather than random. The maximum is taken
    over whatever cycles happen to carry signals, and the validation specimens
    stop having signals long before their crack grows: T7's last measurement
    with a signal sits at 3.14 mm and T8's at 2.50 mm, while every training
    specimen carries signals up to 5-7.5 mm. Their maxima therefore come out
    systematically *smaller* (0.67 and 0.57 against 0.99-1.57), the division
    inflates their differential channel, and the encoder over-estimates
    precisely the two specimens the whole study is judged on.

    The scale is instead read at a damage state **every specimen passes
    through**: the first measurement whose crack length reaches
    ``reference_crack_mm`` (2 mm, below the smallest "large" crack any
    specimen reports and inside T7's and T8's signal window). Falling back to
    the first non-zero measurement when a specimen never reports one that
    large keeps the definition total.

    This is a *structural* correction, chosen because the previous definition
    was not comparable across specimens — **not** because it scored better on
    T7/T8. Selecting the normalisation by validation performance would be the
    same leakage the audit set out to remove.

    Still **transductive**: it reads waveforms of the specimen under test. That
    is legitimate under the challenge protocol (competitors received each
    validation specimen's complete signal set at once) but would not be
    available on-line, where a running estimate would be needed.

    Por qué hace falta normalizar en absoluto: la amplitud de la firma de daño
    no es comparable entre especímenes, porque cada uno está instrumentado sobre
    un par actuador-sensor **distinto** (T1 y T6 en el par 6-1, T4 y T7 en el
    9-4, T3 en el 8-3...), de modo que la distancia de propagación, el número de
    filas de remaches atravesadas y el pegado del PZT difieren todos.
    """
    from physics_calibration.data import load_curve

    root_path = Path(root) if root is not None else DEFAULT_ROOT
    ref = reference_waveform(name, root)
    curve = load_curve(name)
    usable = [c for c in signal_cycles(root_path, name)
              if (name, c) not in EXCLUDED_RECORDS and c in set(curve.cycles)]
    if not usable:
        return 1.0

    cracks = {int(c): float(a) for c, a in zip(curve.cycles, curve.crack_mm)}
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
    """Two-channel encoder input of shape ``(2, N_SAMPLES)``.

    Channel 0: standardised waveform at ``cycle``.
    Channel 1: its difference against the specimen's undamaged reference,
    divided by the specimen's own differential scale — the damage-sensitive
    channel, made comparable across specimens (see :func:`difference_scale`).
    """
    current = zscore(load_waveform(name, cycle, rep, root))
    difference = current - reference_waveform(name, root)
    if normalise_difference:
        difference = difference / difference_scale(name, root)
    return np.stack([current, difference]).astype(np.float32)


@dataclass(frozen=True)
class Sample:
    """One (waveform, crack length) training example."""

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
    """Every labelled (waveform, crack) pair available on disk.

    ``include_zero`` keeps the cycles labelled with a zero crack length. They
    are genuine supervision — the crack was below the optical detection
    threshold, so the correct target is 0 — and with this little data the extra
    examples matter; they also anchor the network's zero point.
    """
    root_path = Path(root) if root is not None else DEFAULT_ROOT
    out: list[Sample] = []
    for name in specimens:
        table = crack_labels[name]
        labels = dict(zip(table["cycle"], table["crack_length_mm"]))
        ref = reference_cycle(name, root)
        for cycle in signal_cycles(root_path, name):
            if cycle not in labels or (name, cycle) in EXCLUDED_RECORDS:
                continue  # e.g. T3 @ 55391 has a signal but no label
            crack = float(labels[cycle])
            if crack == 0.0 and not include_zero:
                continue
            for rep in available_repetitions(root_path, name, cycle):
                out.append(Sample(name, cycle, rep, crack, cycle == ref))
    return out


# ---------------------------------------------------------------------------
# Rasgos escalares clásicos (para la cabeza híbrida)
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    "rms_diferencial", "max_diferencial", "correlacion",
    *[f"rms_ventana_{i}" for i in range(8)],
    "centroide_temporal", "retardo_fase_us",
)


def scalar_features(name: str, cycle: int, rep: int = 1, root: str | None = None) -> np.ndarray:
    """Descriptores escalares del par (onda, referencia sana).

    Existen porque con 87 ondas etiquetadas el codificador convolucional **no
    llega** a extraer de la señal tanta información como un puñado de rasgos
    físicamente motivados: un control con Random Forest sobre estos mismos
    descriptores alcanza 0,850 mm de error LOSO frente a 1,008 mm de la CNN sola
    (medido en ``control_rf.py``; antes se citaban cinco cifras distintas sin
    código que las produjera).
    En vez de elegir entre una cosa y otra, la cabeza recibe **ambas**.

    Los tres bloques replican la fenomenología que todo ganador del certamen
    explotó (véase ``2nd_place_baseline/features.py``):

    * **pérdida de energía** — la grieta dispersa y atenúa el paquete
      transmitido: RMS y máximo del canal diferencial, y su desglose en ocho
      ventanas temporales;
    * **cambio de fase** — la dispersión retrasa la llegada: retardo que
      maximiza la correlación cruzada contra la referencia sana, en µs;
    * **pérdida de similitud** — las discontinuidades distorsionan la forma:
      coeficiente de correlación de Pearson, y centroide temporal de la energía
      diferencial.

    Se excluye deliberadamente el máximo de la onda de referencia: es constante
    dentro de cada espécimen y actuaría como un identificador del espécimen, lo
    que invita a memorizar en vez de generalizar.
    """
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
