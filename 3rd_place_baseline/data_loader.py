"""Loading utilities for the PHM 2019 Data Challenge dataset (fatigue crack growth).

Reproduces the file layout described in ``PHMDC2019_Data/ReadMe.docx``:
    <root>/<training|validation>/T<k>/Description_T<k>.xlsx
    <root>/<training|validation>/T<k>/<cycle>/signal_1.csv
    <root>/<training|validation>/T<k>/<cycle>/signal_2.csv

Each ``signal_*.csv`` has columns ``time, ch1, ch2`` sampled at 20 MHz
(dt = 5e-8 s). ``ch1`` is the actuation/excitation signal and ``ch2`` is the
received signal (ReadMe.docx, Section 2). Run 1 and Run 2 (``signal_1`` /
``signal_2``) are two repeated measurements of the same cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SAMPLING_FREQUENCY_HZ = 20e6

CONSTANT_LOADING = "constant"
VARIABLE_LOADING = "variable"

TRAINING_SPECIMENS = ["T1", "T2", "T3", "T4", "T5", "T6"]
VALIDATION_SPECIMENS = ["T7", "T8"]

LOADING_CONDITION = {
    "T1": CONSTANT_LOADING, "T2": CONSTANT_LOADING, "T3": CONSTANT_LOADING,
    "T4": CONSTANT_LOADING, "T5": CONSTANT_LOADING, "T6": CONSTANT_LOADING,
    "T7": CONSTANT_LOADING, "T8": VARIABLE_LOADING,
}

# Cycles for which T7/T8 have no wave signal and the crack length must be
# *predicted* with Paris' Law rather than estimated from features (matches
# the split visible on disk and Table 4/Table A of the paper).
PREDICTION_CYCLES = {
    "T7": [49026, 51030, 53019, 55031],
    "T8": [89237, 92315, 96475, 98492, 100774],
}

# Known data-entry typo in the released Description_T4.xlsx: cycle "7054" is
# the on-disk signal folder "67054" with a dropped leading digit (crack
# 2.74 mm fits the monotonic 2.17 -> 2.74 -> 3.13 mm progression).
_CYCLE_TYPO_FIXES = {"T4": {7054: 67054}}


def _specimen_dir(root: Path, name: str) -> Path:
    group = "training" if name in TRAINING_SPECIMENS else "validation"
    return root / group / name


def load_description(root: Path, name: str) -> pd.DataFrame:
    """Read ``Description_T<k>.xlsx`` -> DataFrame[cycle, crack_length_mm].

    Description_T3.xlsx has no crack-length label for cycle 55391 even
    though its signal folder exists; that cycle is absent from the
    returned table.
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


@dataclass
class SpecimenData:
    """All the data available for one specimen (T1..T8)."""

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
        """Cycles that have both a wave signal folder and a crack-length label."""
        labeled = set(self.description["cycle"].tolist())
        return [c for c in self.signal_cycles if c in labeled]

    def first_zero_cycle(self) -> int:
        """First (earliest) signal-available cycle of the specimen, used as
        the normalization reference (Sec. 3.5) and as the Table-1 crack
        detection algorithm's baseline (Step 1, ``c1 = 0``)."""
        cycles = self.labeled_cycles()
        first = cycles[0]
        if self.crack_length(first) != 0:
            raise ValueError(f"{self.name}: first labeled cycle {first} has nonzero crack length")
        return first

    def last_zero_crack_cycle(self) -> int | None:
        """Last labeled cycle with a measured crack length of exactly zero,
        i.e. the cycle immediately preceding crack initiation (v_50 in
        Eq. for feature v5, Sec. 3.5; only used for T7)."""
        zero_cycles = [c for c in self.labeled_cycles() if self.crack_length(c) == 0]
        return max(zero_cycles) if zero_cycles else None

    def signal(self, cycle: int, channel: str = "signal_1") -> pd.DataFrame:
        return load_signal(self.root, self.name, cycle, channel)

    def available_channels(self, cycle: int) -> list[str]:
        return available_signal_channels(self.root, self.name, cycle)


def load_all(root: Path, names: list[str] | None = None) -> dict[str, SpecimenData]:
    names = names or (TRAINING_SPECIMENS + VALIDATION_SPECIMENS)
    return {name: SpecimenData(root=root, name=name) for name in names}
