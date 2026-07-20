"""Loading utilities for the PHM 2019 Data Challenge dataset (fatigue crack growth).

Reproduces the file layout described in ``PHMDC2019_Data/ReadMe.docx``:
    <root>/<training|validation>/T<k>/Description_T<k>.xlsx
    <root>/<training|validation>/T<k>/<cycle>/signal_1.csv
    <root>/<training|validation>/T<k>/<cycle>/signal_2.csv

Each ``signal_*.csv`` has columns ``time, ch1, ch2`` where ``ch1`` is the
excitation (actuator) signal and ``ch2`` is the received (sensor) signal,
sampled at 20 MHz (dt = 5e-8 s).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Sampling frequency of the PZT wave acquisition system (Section 3.1.1 of the paper).
SAMPLING_FREQUENCY_HZ = 20e6

# Loading condition per specimen, taken from Table 1 / Table 2 and Section 4.2 of the paper.
CONSTANT_LOADING = "constant"
VARIABLE_LOADING = "variable"

TRAINING_SPECIMENS = ["T1", "T2", "T3", "T4", "T5", "T6"]
VALIDATION_SPECIMENS = ["T7", "T8"]

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


# Known data-entry typos in the released Description_T<k>.xlsx files, fixed here
# against the (unambiguous) wave-signal folder names on disk. T4's sheet lists a
# cycle "7054" between 65001 and 70016 with crack length 2.74 mm -- there is no
# "7054" signal folder, but a "67054" folder exists and 2.74 mm fits the monotonic
# 2.17 -> 2.74 -> 3.13 mm progression, so this is a dropped leading digit.
_CYCLE_TYPO_FIXES = {
    "T4": {7054: 67054},
}


def load_description(root: Path, name: str) -> pd.DataFrame:
    """Read ``Description_T<k>.xlsx`` -> DataFrame[cycle, crack_length_mm].

    Note: Description_T3.xlsx is missing the crack-length label for cycle
    55391 even though its wave signal folder exists (a data quality quirk of
    the released dataset, not a parsing bug); that cycle is simply absent
    from the returned table and is excluded from SVR training.
    """
    path = _specimen_dir(root, name) / f"Description_{name}.xlsx"
    raw = pd.read_excel(path, sheet_name=0, header=None)
    header_row = raw.index[raw[0] == "Number of cycle"][0]
    table = raw.iloc[header_row + 1:, :2]
    table = table.dropna(how="all")
    table.columns = ["cycle", "crack_length_mm"]
    table = table.dropna(subset=["cycle"])
    table["cycle"] = table["cycle"].astype(int)
    table["crack_length_mm"] = table["crack_length_mm"].astype(float)
    fixes = _CYCLE_TYPO_FIXES.get(name, {})
    if fixes:
        table["cycle"] = table["cycle"].replace(fixes)
    return table.sort_values("cycle").reset_index(drop=True)


def load_signal(root: Path, name: str, cycle: int, channel: str) -> pd.DataFrame:
    """Read one ``signal_1.csv`` / ``signal_2.csv`` file for a given cycle."""
    path = _specimen_dir(root, name) / str(cycle) / f"{channel}.csv"
    return pd.read_csv(path)


def available_signal_channels(root: Path, name: str, cycle: int) -> list[str]:
    """Which of signal_1.csv / signal_2.csv actually exist for this cycle."""
    base = _specimen_dir(root, name) / str(cycle)
    return [c for c in ("signal_1", "signal_2") if (base / f"{c}.csv").is_file()]


def available_signal_cycles(root: Path, name: str) -> list[int]:
    """Cycle numbers that have a wave-signal folder with *at least one*
    signal CSV for this specimen. Some released folders are incomplete:
    training/T5/42000 exists but is empty (just a stray macOS
    ``.DS_Store``), and training/T5/56000 has only ``signal_1.csv``."""
    base = _specimen_dir(root, name)
    cycles = sorted(
        int(p.name)
        for p in base.iterdir()
        if p.is_dir()
        and p.name.isdigit()
        and available_signal_channels(root, name, int(p.name))
    )
    return cycles


def load_loading_profile(root: Path, name: str) -> pd.DataFrame:
    """Read the 'Constant/Variable Loading Profile-*.csv' file for a specimen."""
    base = _specimen_dir(root, name)
    candidates = list(base.glob("*Loading Profile*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No loading profile csv found for {name} in {base}")
    return pd.read_csv(candidates[0])


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
        """Cycles that have both a wave signal folder AND a crack-length label."""
        labeled = set(self.description["cycle"].tolist())
        return [c for c in self.signal_cycles if c in labeled]

    def post_initiation_cycles(self) -> list[int]:
        """Labeled cycles with a strictly positive (post-crack-initiation) crack length."""
        return [c for c in self.labeled_cycles() if self.crack_length(c) > 0]

    def signals(self, cycle: int) -> dict[str, pd.DataFrame]:
        channels = available_signal_channels(self.root, self.name, cycle)
        return {c: load_signal(self.root, self.name, cycle, c) for c in channels}

    def undamaged_baseline_cycle(self, override: int | None = None) -> int:
        """Pick the reference ("undamaged") cycle used to normalize features.

        Rule of thumb (matches the paper for T7): the last cycle, among those
        that have a wave signal, whose measured crack length is still zero.
        For T8 the paper explicitly overrides this rule (Section 4.2): the
        wave signal already shows damage-like change at cycle 70000 even
        though the optical crack-length measurement still reads 0 mm there,
        so cycle 50000 is used as the baseline instead. Pass ``override`` to
        reproduce that documented exception.
        """
        if override is not None:
            return override
        zero_cycles = [c for c in self.labeled_cycles() if self.crack_length(c) == 0]
        if not zero_cycles:
            raise ValueError(f"No zero-crack-length labeled cycle found for {self.name}")
        return max(zero_cycles)

    def loading_profile(self) -> pd.DataFrame:
        return load_loading_profile(self.root, self.name)


def load_all(root: Path, names: list[str]) -> dict[str, SpecimenData]:
    return {name: SpecimenData(root=root, name=name) for name in names}
