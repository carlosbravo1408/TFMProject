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

PREDICTION_CYCLES = {
    "T7": [49026, 51030, 53019, 55031],
    "T8": [89237, 92315, 96475, 98492, 100774],
}

# T4's sheet lists cycle 7054; the signal folder is 67054 and 2.74 mm fits the
# progression between 65001 and 70016.
_CYCLE_TYPO_FIXES = {"T4": {7054: 67054}}


def _specimen_dir(root: Path, name: str) -> Path:
    group = "training" if name in TRAINING_SPECIMENS else "validation"
    return root / group / name


def load_description(root: Path, name: str) -> pd.DataFrame:
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
    base = _specimen_dir(root, name)
    # T5/42000 is an empty folder.
    return sorted(
        int(p.name)
        for p in base.iterdir()
        if p.is_dir() and p.name.isdigit() and available_signal_channels(root, name, int(p.name))
    )


@dataclass
class SpecimenData:
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

    def first_zero_cycle(self) -> int:
        cycles = self.labeled_cycles()
        first = cycles[0]
        if self.crack_length(first) != 0:
            raise ValueError(f"{self.name}: first labeled cycle {first} has nonzero crack length")
        return first

    def last_zero_crack_cycle(self) -> int | None:
        zero_cycles = [c for c in self.labeled_cycles() if self.crack_length(c) == 0]
        return max(zero_cycles) if zero_cycles else None

    def signal(self, cycle: int, channel: str = "signal_1") -> pd.DataFrame:
        return load_signal(self.root, self.name, cycle, channel)

    def available_channels(self, cycle: int) -> list[str]:
        return available_signal_channels(self.root, self.name, cycle)


def load_all(root: Path, names: list[str] | None = None) -> dict[str, SpecimenData]:
    names = names or (TRAINING_SPECIMENS + VALIDATION_SPECIMENS)
    return {name: SpecimenData(root=root, name=name) for name in names}
