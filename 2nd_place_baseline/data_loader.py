"""Loading utilities for the PHM 2019 Data Challenge dataset (fatigue crack growth).

Reproduces the file layout described in ``PHMDC2019_Data/ReadMe.docx``:
    <root>/<training|validation>/T<k>/Description_T<k>.xlsx
    <root>/<training|validation>/T<k>/<cycle>/signal_1.csv
    <root>/<training|validation>/T<k>/<cycle>/signal_2.csv

Each ``signal_*.csv`` has columns ``time, ch1, ch2`` where ``ch1`` is the
excitation (actuator) signal and ``ch2`` is the received (sensor) signal,
sampled at 20 MHz (dt = 5e-8 s).

Specimen roles follow Kong et al. (2020), "A Hybrid Approach of Data-driven
and Physics-based Methods for Estimation and Prediction of Fatigue Crack
Growth" (IJPHM):
  - T1..T6: training (T5 excluded from the data-driven model as an outlier,
    Section 3.2.2; T2 and T5 excluded from the physics-based exponential
    ensemble for having too few points, Section 3.3.2).
  - T7 (constant loading) and T8 (variable loading): validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLING_FREQUENCY_HZ = 20e6

CONSTANT_LOADING = "constant"
VARIABLE_LOADING = "variable"

TRAINING_SPECIMENS = ["T1", "T2", "T3", "T4", "T5", "T6"]
VALIDATION_SPECIMENS = ["T7", "T8"]

# Specimens used by the data-driven random forest model (T5 excluded, Sec. 3.2.2).
DATA_DRIVEN_SPECIMENS = ["T1", "T2", "T3", "T4", "T6"]
# Specimens used by the ensemble-prognostics exponential models (Sec. 3.3.2).
ENSEMBLE_SPECIMENS = ["T1", "T3", "T4", "T6"]

LOADING_CONDITION = {
    "T1": CONSTANT_LOADING,
    "T2": CONSTANT_LOADING,
    "T3": CONSTANT_LOADING,
    "T4": CONSTANT_LOADING,
    "T5": CONSTANT_LOADING,
    "T6": CONSTANT_LOADING,
    "T7": CONSTANT_LOADING,
    "T8": VARIABLE_LOADING,
}


def _specimen_dir(root: Path, name: str) -> Path:
    group = "training" if name in TRAINING_SPECIMENS else "validation"
    return root / group / name


# Known data-entry typo in the released Description_T4.xlsx: cycle "7054" is the
# on-disk signal folder "67054" with a dropped leading digit (crack 2.74 mm fits
# the monotonic 2.17 -> 2.74 -> 3.13 mm progression).
_CYCLE_TYPO_FIXES = {
    "T4": {7054: 67054},
}


def load_description(root: Path, name: str) -> pd.DataFrame:
    """Read ``Description_T<k>.xlsx`` -> DataFrame[cycle, crack_length_mm].

    Note: Description_T3.xlsx has no crack-length label for cycle 55391 even
    though its signal folder exists; that cycle is absent from the returned
    table (matching Table 1 of the paper, which also omits it).
    """
    path = _specimen_dir(root, name) / f"Description_{name}.xlsx"
    raw = pd.read_excel(path, sheet_name=0, header=None)
    header_row = raw.index[raw[0] == "Number of cycle"][0]
    table = raw.iloc[header_row + 1:, :2]
    table = table.dropna(how="all")
    table.columns = ["cycle", "crack_length_mm"]
    table = table.dropna(subset=["cycle", "crack_length_mm"])
    table["cycle"] = table["cycle"].astype(int)
    table["crack_length_mm"] = table["crack_length_mm"].astype(float)
    fixes = _CYCLE_TYPO_FIXES.get(name, {})
    if fixes:
        table["cycle"] = table["cycle"].replace(fixes)
    return table.sort_values("cycle").reset_index(drop=True)


def load_signal(root: Path, name: str, cycle: int, channel: str = "signal_1") -> pd.DataFrame:
    """Read one ``signal_1.csv`` / ``signal_2.csv`` file for a given cycle.

    Section 3.2.1 of the paper: the two repeated measurements are nearly
    identical after band-pass filtering, so only the first measured signal
    (``signal_1``) is used.
    """
    path = _specimen_dir(root, name) / str(cycle) / f"{channel}.csv"
    return pd.read_csv(path)


def available_signal_channels(root: Path, name: str, cycle: int) -> list[str]:
    base = _specimen_dir(root, name) / str(cycle)
    return [c for c in ("signal_1", "signal_2") if (base / f"{c}.csv").is_file()]


def available_signal_cycles(root: Path, name: str) -> list[int]:
    """Cycle numbers with at least one signal CSV (training/T5/42000 is an
    empty folder in the released dataset)."""
    base = _specimen_dir(root, name)
    return sorted(
        int(p.name)
        for p in base.iterdir()
        if p.is_dir() and p.name.isdigit() and available_signal_channels(root, name, int(p.name))
    )


def load_loading_profile(root: Path, name: str) -> pd.DataFrame:
    base = _specimen_dir(root, name)
    candidates = list(base.glob("*Loading Profile*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No loading profile csv found for {name} in {base}")
    return pd.read_csv(candidates[0])


@dataclass
class SpecimenData:
    """All data available for one specimen (T1..T8)."""

    root: Path
    name: str
    description: pd.DataFrame = field(init=False)
    signal_cycles: list[int] = field(init=False)
    loading_condition: str = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.description = load_description(self.root, self.name)
        self.signal_cycles = available_signal_cycles(self.root, self.name)
        self.loading_condition = LOADING_CONDITION[self.name]

    def crack_length(self, cycle: int) -> float | None:
        row = self.description.loc[self.description["cycle"] == cycle]
        if row.empty:
            return None
        return float(row["crack_length_mm"].iloc[0])

    def labeled_cycles(self) -> list[int]:
        labeled = set(self.description["cycle"].tolist())
        return [c for c in self.signal_cycles if c in labeled]

    def reference_cycle(self) -> int:
        """First measured cycle of the specimen, used as the undamaged
        reference for the phase-change / correlation features (Sec. 3.2.2)."""
        return self.signal_cycles[0]

    def initiation_cycle(self) -> int | None:
        """First cycle whose *measured* crack length is nonzero (used as
        N_initial when normalizing cycles, Sec. 3.3.1)."""
        nonzero = self.description.loc[self.description["crack_length_mm"] > 0, "cycle"]
        return int(nonzero.iloc[0]) if len(nonzero) else None

    def signal(self, cycle: int, channel: str = "signal_1") -> pd.DataFrame:
        return load_signal(self.root, self.name, cycle, channel)

    def loading_profile(self) -> pd.DataFrame:
        return load_loading_profile(self.root, self.name)


def load_all(root: Path, names: list[str]) -> dict[str, SpecimenData]:
    return {name: SpecimenData(root=root, name=name) for name in names}
