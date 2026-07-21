"""Crack-sensitive feature extraction (Sec. 3.3), normalization and crack
detection (Sec. 3.4) of the paper.

Four features are computed on each FWP:
    v1 - first peak value: |filtered signal| at the LFP.
    v2 - root mean square value of the FWP.
    v3 - logarithm of kurtosis, Eq. (1): log(sum((s-mu)^4) / sigma^4).
    v4 - correlation coefficient of the FWP against the same specimen/run's
         FWP at its first zero-crack cycle.
A fifth feature, the (specimen-relative) cycle number, is used *in the model
built for T7* (Sec. 3.5): v5 = (cycle - v50) / 25000, where v50 is the cycle
number of the specimen's own last zero-crack cycle. Section 3.5 explains
this is only meaningful across specimens that share T7's loading (T1-T7,
excluding T8), so it is computed for every specimen here (needed as a
training feature for the T1-T6 leave-one-out folds of the "T7" model type)
but only ever included in the T7-side candidate feature set
(``crack_estimator.T7_BASE_FEATURES``).

Normalization (Sec. 3.5): v1, v2 and v3 are divided by the same feature
computed on the specimen/run's first zero-crack cycle; v4 needs no further
normalization since it is already computed relative to that same reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loader import SpecimenData
from signal_processing import SignalPreprocessor

RUN_CHANNELS = ["signal_1", "signal_2"]
V5_CYCLE_SCALE = 25000.0


def first_peak_value(fwp: np.ndarray, lfp_offset: int = 150) -> float:
    """v1: amplitude at the LFP, i.e. the FWP sample ``lfp_offset`` samples
    from its start (the FWP is built as ``[LFP-150, LFP+200]``)."""
    idx = min(lfp_offset, len(fwp) - 1)
    return float(np.abs(fwp[idx]))


def rms_value(fwp: np.ndarray) -> float:
    """v2."""
    return float(np.sqrt(np.mean(np.square(fwp))))


def log_kurtosis(fwp: np.ndarray) -> float:
    """v3, Eq. (1)."""
    mu, sigma = np.mean(fwp), np.std(fwp)
    if sigma == 0:
        return 0.0
    raw = np.sum(((fwp - mu) / sigma) ** 4)
    return float(np.log(raw)) if raw > 0 else 0.0


def correlation_coefficient(fwp: np.ndarray, reference_fwp: np.ndarray) -> float:
    """v4: Pearson correlation against the reference (first zero-crack
    cycle) FWP of the same specimen/run. Windows may differ by a couple of
    samples right at the end of a record (LFP + 200 clipped to 4000
    samples); the shorter common prefix is used in that case."""
    n = min(len(fwp), len(reference_fwp))
    fwp, reference_fwp = fwp[:n], reference_fwp[:n]
    if np.std(fwp) == 0 or np.std(reference_fwp) == 0:
        return 1.0
    return float(np.corrcoef(fwp, reference_fwp)[0, 1])


@dataclass
class RawFeatures:
    v1: float
    v2: float
    v3: float
    v4: float


def compute_raw_features(fwp: np.ndarray, reference_fwp: np.ndarray) -> RawFeatures:
    return RawFeatures(
        v1=first_peak_value(fwp),
        v2=rms_value(fwp),
        v3=log_kurtosis(fwp),
        v4=correlation_coefficient(fwp, reference_fwp),
    )


class FeatureExtractor:
    """Builds the per-cycle, per-run feature table of a specimen."""

    def __init__(self, preprocessor: SignalPreprocessor | None = None) -> None:
        self.preprocessor = preprocessor or SignalPreprocessor()
        self._reference_fwp: dict[tuple[str, str], np.ndarray] = {}

    def fit_reference(self, spec: SpecimenData) -> None:
        """Fix the LFP offset and reference FWP for every run channel of a
        specimen, from its first zero-crack cycle."""
        ref_cycle = spec.first_zero_cycle()
        for channel in spec.available_channels(ref_cycle):
            df = spec.signal(ref_cycle, channel)
            self.preprocessor.fit_reference(spec.name, channel, df["ch1"].to_numpy(), df["ch2"].to_numpy())
            ts = self.preprocessor.truncate(spec.name, channel, df["ch1"].to_numpy(), df["ch2"].to_numpy())
            self._reference_fwp[(spec.name, channel)] = ts.fwp

    def specimen_feature_table(self, spec: SpecimenData, cycles: list[int] | None = None) -> pd.DataFrame:
        """Raw + normalized v1-v4 (+v5 for T7) for every (cycle, run) of a
        specimen. ``cycles`` defaults to all labeled cycles."""
        if (spec.name, "signal_1") not in self._reference_fwp and (
            spec.name, "signal_2"
        ) not in self._reference_fwp:
            self.fit_reference(spec)
        cycles = sorted(cycles if cycles is not None else spec.labeled_cycles())
        ref_cycle = spec.first_zero_cycle()
        v50 = spec.last_zero_crack_cycle()

        # Reference (first zero-crack cycle) raw feature values, per run, for normalization.
        ref_raw: dict[str, RawFeatures] = {}
        for channel, ref_fwp in self._reference_fwp.items():
            if channel[0] != spec.name:
                continue
            ref_raw[channel[1]] = compute_raw_features(ref_fwp, ref_fwp)

        rows = []
        for cycle in cycles:
            for channel in spec.available_channels(cycle):
                if (spec.name, channel) not in self._reference_fwp:
                    continue
                df = spec.signal(cycle, channel)
                ts = self.preprocessor.truncate(spec.name, channel, df["ch1"].to_numpy(), df["ch2"].to_numpy())
                raw = compute_raw_features(ts.fwp, self._reference_fwp[(spec.name, channel)])
                ref = ref_raw[channel]
                row = {
                    "specimen": spec.name,
                    "cycle": cycle,
                    "run": channel,
                    "v1_raw": raw.v1, "v2_raw": raw.v2, "v3_raw": raw.v3, "v4_raw": raw.v4,
                    "v1": raw.v1 / ref.v1 if ref.v1 else 0.0,
                    "v2": raw.v2 / ref.v2 if ref.v2 else 0.0,
                    "v3": raw.v3 / ref.v3 if ref.v3 else 0.0,
                    "v4": raw.v4,
                    "crack_length_mm": spec.crack_length(cycle),
                }
                if v50 is not None:
                    row["v5"] = (cycle - v50) / V5_CYCLE_SCALE
                rows.append(row)
        return pd.DataFrame(rows)


def detect_crack_onset_cycle(table: pd.DataFrame) -> int | None:
    """Table 1's crack-detection algorithm applied to one specimen/run's
    chronological raw-feature sequence. Returns the first cycle at which the
    algorithm declares "crack occurs" (``None`` if never triggered within
    the given cycles). The very first cycle is Step 1's baseline
    (``c1 = 0`` by construction of ``first_zero_cycle``)."""
    table = table.sort_values("cycle").reset_index(drop=True)
    if len(table) == 0:
        return None
    baseline = table.loc[0, ["v1_raw", "v2_raw", "v3_raw"]].to_numpy(dtype=float)
    for i in range(1, len(table)):
        v = table.loc[i, ["v1_raw", "v2_raw", "v3_raw"]].to_numpy(dtype=float)
        if v[0] < baseline[0] and v[1] < baseline[1] and v[2] > baseline[2]:
            return int(table.loc[i, "cycle"])
        if v[0] > baseline[0] and v[1] > baseline[1] and v[2] < baseline[2]:
            baseline = v
        # else: "beyond scope of the algorithm" (Table 1) -- keep the
        # current baseline and continue to the next cycle.
    return None
