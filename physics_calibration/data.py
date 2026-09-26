"""Ground-truth crack-growth curves and load spectra of the PHM 2019 dataset.

This module deliberately depends on nothing but the released dataset: the
calibration of the fracture-mechanics hyper-parameters must be reproducible
without running any of the Lamb-wave signal-processing pipelines.

Crack lengths come from ``Description_T<k>.xlsx`` (mm) and the load spectrum
from the ``*Loading Profile*.csv`` shipped with each specimen. The constant
profile (T1-T7) is a 5-cycle sine between Smin = 4.77 MPa and
Smax = 100.21 MPa; the variable profile (T8) is one repeating block of 1000
cycles: 500 cycles at Smax = 90.00 MPa followed by 500 cycles at
Smax = 100.21 MPa, both at Smin = 4.77 MPa. Both are decoded from the CSVs by
:func:`load_load_block` rather than hard-coded, so a different release of the
dataset would be picked up automatically.

Dataset gotchas handled here (see CLAUDE.md):
  - Description_T4.xlsx has a typo'd cycle 7054 (the on-disk folder is 67054).
  - Description_T3.xlsx has no label for cycle 55391.
  - training/T5/42000 is an empty folder.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "PHMDC2019_Data"

TRAINING_SPECIMENS = ("T1", "T2", "T3", "T4", "T5", "T6")
VALIDATION_SPECIMENS = ("T7", "T8")
ALL_SPECIMENS = TRAINING_SPECIMENS + VALIDATION_SPECIMENS

# Specimens with enough non-zero crack measurements to identify a 2-3 parameter
# growth law. T2 (2 points) and T5 (2 points, and the acknowledged outlier of
# the challenge) cannot constrain C, m and gamma simultaneously.
CALIBRATION_SPECIMENS = ("T1", "T3", "T4", "T6")

VARIABLE_LOADING = ("T8",)

# Known data-entry typo in the released Description_T4.xlsx.
_CYCLE_TYPO_FIXES = {"T4": {7054: 67054}}

# Sampling of the released loading-profile CSVs: 20 samples per fatigue cycle
# (the constant profile holds 5 cycles in 101 samples of dt = 0.01 s).
_SAMPLES_PER_CYCLE = 20


def _specimen_dir(root: Path, name: str) -> Path:
    group = "training" if name in TRAINING_SPECIMENS else "validation"
    return root / group / name


def load_description(root: Path | str = DEFAULT_ROOT, name: str = "T1") -> pd.DataFrame:
    """``Description_T<k>.xlsx`` -> DataFrame[cycle, crack_length_mm]."""
    path = _specimen_dir(Path(root), name) / f"Description_{name}.xlsx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pd.read_excel(path, sheet_name=0, header=None)
    header_row = raw.index[raw[0] == "Number of cycle"][0]
    table = raw.iloc[header_row + 1:, :2].dropna(how="all")
    table.columns = ["cycle", "crack_length_mm"]
    table = table.dropna(subset=["cycle", "crack_length_mm"])
    table["cycle"] = table["cycle"].astype(int).replace(_CYCLE_TYPO_FIXES.get(name, {}))
    table["crack_length_mm"] = table["crack_length_mm"].astype(float)
    return table.sort_values("cycle").reset_index(drop=True)


def load_load_block(root: Path | str = DEFAULT_ROOT, name: str = "T1") -> np.ndarray:
    """Decode the specimen's loading profile into one repeating block.

    Returns an ``(n_segments, 3)`` array of ``(n_cycles, s_max_MPa,
    s_min_MPa)``: a single segment for the constant-amplitude specimens and
    the two-segment 1000-cycle block for T8.
    """
    base = _specimen_dir(Path(root), name)
    candidates = sorted(base.glob("*Loading Profile*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no loading profile csv for {name} in {base}")
    profile = pd.read_csv(candidates[0])
    load = profile["loading"].to_numpy(dtype=float)

    n_cycles = (len(load) - 1) // _SAMPLES_PER_CYCLE
    per_cycle = np.array(
        [
            (load[i * _SAMPLES_PER_CYCLE: (i + 1) * _SAMPLES_PER_CYCLE + 1].max(),
             load[i * _SAMPLES_PER_CYCLE: (i + 1) * _SAMPLES_PER_CYCLE + 1].min())
            for i in range(n_cycles)
        ]
    )
    # Run-length encode the (Smax, Smin) sequence into segments.
    rounded = np.round(per_cycle, 6)
    segments, start = [], 0
    for i in range(1, n_cycles + 1):
        if i == n_cycles or not np.array_equal(rounded[i], rounded[start]):
            segments.append((i - start, *rounded[start]))
            start = i
    if name not in VARIABLE_LOADING:
        # Constant amplitude: collapse to a single 1-cycle segment.
        s_max, s_min = rounded[0]
        return np.array([[1.0, s_max, s_min]])
    return np.array(segments, dtype=float)


@dataclass(frozen=True)
class CrackCurve:
    """One specimen's measured crack-growth curve, ready for law fitting.

    ``cycles`` / ``crack_mm`` keep only the measurements with a non-zero crack
    length: the fracture-mechanics laws describe *propagation*, and the zero
    labels only say the crack was below the detection threshold, not that it
    was absent. ``n0`` is the first such cycle (the propagation anchor) and
    ``a0_mm`` the crack length measured there.
    """

    name: str
    cycles: np.ndarray
    crack_mm: np.ndarray
    load_block: np.ndarray
    final_crack_mm: float
    zero_cycles: np.ndarray
    n_observed: int

    @property
    def n0(self) -> float:
        return float(self.cycles[0])

    @property
    def a0_mm(self) -> float:
        return float(self.crack_mm[0])

    @property
    def delta_cycles(self) -> np.ndarray:
        """Cycles elapsed since the propagation anchor ``n0``."""
        return self.cycles - self.n0

    @property
    def is_variable_amplitude(self) -> bool:
        return self.name in VARIABLE_LOADING

    @property
    def stress_ratio(self) -> float:
        """Load-block-averaged R = Smin/Smax (identical per segment here)."""
        return float(np.mean(self.load_block[:, 2] / self.load_block[:, 1]))


def signal_cycles(root: Path | str = DEFAULT_ROOT, name: str = "T1") -> np.ndarray:
    """Cycles for which Lamb-wave signal files actually exist on disk.

    This is what makes the challenge protocol reproducible without guessing:
    the released validation specimens simply stop having signals partway
    through (T7 after cycle 47022, T8 after 76931), and everything past that
    point had to be *predicted* blind by the competitors. ``training/T5/42000``
    is an empty folder in the release and is excluded.
    """
    base = _specimen_dir(Path(root), name)
    return np.array(
        sorted(
            int(p.name) for p in base.iterdir()
            if p.is_dir() and p.name.isdigit()
            and any((p / f"{c}.csv").is_file() for c in ("signal_1", "signal_2"))
        ),
        dtype=float,
    )


@lru_cache(maxsize=None)
def load_curve(name: str, root: str | None = None) -> CrackCurve:
    root_path = Path(root) if root is not None else DEFAULT_ROOT
    table = load_description(root_path, name)
    nonzero = table.loc[table["crack_length_mm"] > 0]
    cycles = nonzero["cycle"].to_numpy(dtype=float)
    observed = signal_cycles(root_path, name)
    return CrackCurve(
        name=name,
        cycles=cycles,
        crack_mm=nonzero["crack_length_mm"].to_numpy(dtype=float),
        load_block=load_load_block(root_path, name),
        final_crack_mm=float(nonzero["crack_length_mm"].iloc[-1]),
        zero_cycles=table.loc[table["crack_length_mm"] == 0, "cycle"].to_numpy(dtype=float),
        n_observed=int(np.sum(cycles <= observed.max())) if len(observed) else len(cycles),
    )


def load_curves(names=ALL_SPECIMENS, root: str | None = None) -> dict[str, CrackCurve]:
    return {name: load_curve(name, root) for name in names}


def empirical_growth_rates(curve: CrackCurve) -> pd.DataFrame:
    """Central/forward finite differences da/dN (m/cycle) vs. the driving
    force, for the model-free log-log sanity check of Section "diagnóstico".

    ``delta_k`` uses the block-averaged stress range and the mid-interval
    crack length, with geometry factor Y = 1.
    """
    a_m = curve.crack_mm * 1e-3
    n = curve.cycles
    a_mid = 0.5 * (a_m[1:] + a_m[:-1])
    da_dn = np.diff(a_m) / np.diff(n)
    weights = curve.load_block[:, 0]
    d_sigma = float(np.average(curve.load_block[:, 1] - curve.load_block[:, 2], weights=weights))
    s_max = float(np.average(curve.load_block[:, 1], weights=weights))
    delta_k = d_sigma * np.sqrt(np.pi * a_mid)
    return pd.DataFrame(
        {
            "specimen": curve.name,
            "cycle_mid": 0.5 * (n[1:] + n[:-1]),
            "a_mid_mm": a_mid * 1e3,
            "da_dn_m_per_cycle": da_dn,
            "delta_k_MPa_sqrt_m": delta_k,
            "k_max_MPa_sqrt_m": s_max * np.sqrt(np.pi * a_mid),
        }
    )
