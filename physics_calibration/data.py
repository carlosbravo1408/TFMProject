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

# T2 and T5 have only two non-zero measurements; T5 is also the challenge outlier.
CALIBRATION_SPECIMENS = ("T1", "T3", "T4", "T6")

VARIABLE_LOADING = ("T8",)

# Description_T4.xlsx lists cycle 7054; the signal folder is 67054.
_CYCLE_TYPO_FIXES = {"T4": {7054: 67054}}

# The constant profile holds 5 cycles in 101 samples.
_SAMPLES_PER_CYCLE = 20


def _specimen_dir(root: Path, name: str) -> Path:
    group = "training" if name in TRAINING_SPECIMENS else "validation"
    return root / group / name


def load_description(root: Path | str = DEFAULT_ROOT, name: str = "T1") -> pd.DataFrame:
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
    rounded = np.round(per_cycle, 6)
    segments, start = [], 0
    for i in range(1, n_cycles + 1):
        if i == n_cycles or not np.array_equal(rounded[i], rounded[start]):
            segments.append((i - start, *rounded[start]))
            start = i
    if name not in VARIABLE_LOADING:
        s_max, s_min = rounded[0]
        return np.array([[1.0, s_max, s_min]])
    return np.array(segments, dtype=float)


@dataclass(frozen=True)
class CrackCurve:
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
        return self.cycles - self.n0

    @property
    def is_variable_amplitude(self) -> bool:
        return self.name in VARIABLE_LOADING

    @property
    def stress_ratio(self) -> float:
        return float(np.mean(self.load_block[:, 2] / self.load_block[:, 1]))


def signal_cycles(root: Path | str = DEFAULT_ROOT, name: str = "T1") -> np.ndarray:
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
    # Zero labels only mean the crack was below the detection threshold.
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
