"""End-to-end pipeline replicating Rao et al. (2020) -- the 3rd-place entry
("Angler") of the 2019 PHM Conference Data Challenge.

Crack length ESTIMATION (Sec. 3), for cycles with wave signals:
    band-pass filter + FWP truncation -> v1-v4 (+v5 for T7) features ->
    Table 1 crack-detection algorithm -> ensemble of 3 (of 6) leave-one-
    specimen-out Best-Subset-Selected linear regression models.

Crack length PREDICTION (Sec. 4), for cycles without wave signals:
    Paris' Law (Eq. 5/6) with material parameters (C, m, and for T8 the
    equivalent stress range) fitted by a genetic-algorithm-style optimizer
    to an "initial dataset" of a few (cycle, crack) anchor points.

Scoring: official PHM 2019 penalty score (Eqs. 1-4) + RMSE, both on the full
chronological sequence of estimates+predictions for T7 and T8. The paper
reports a total score of 16.14; ``scoring.py`` reproduces that number to
within 0.3% when fed the paper's own Table A values (see its docstring),
validating the scoring implementation independently of this pipeline's own
signal-processing/modeling choices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from crack_estimator import (
    T7_BASE_FEATURES,
    T8_BASE_FEATURES,
    BSSModel,
    ensemble_models,
    ensemble_predict,
    select_ensemble,
)
from data_loader import PREDICTION_CYCLES, SpecimenData, TRAINING_SPECIMENS, load_all
from features import FeatureExtractor, detect_crack_onset_cycle
from paris_law import (
    DELTA_SIGMA_CONSTANT_MPA,
    ParisFit,
    build_t7_initial_dataset,
    build_t8_initial_dataset,
    fit_paris_law,
)
from scoring import penalty_score, rmse_mm, score_table

# Paper's Table A (Appendix) estimated/predicted and real crack lengths, for
# side-by-side comparison with this replica.
PAPER_TABLE_A = {
    "T7": {
        "cycle": [36001, 40167, 44054, 47022, 49026, 51030, 53019, 55031],
        "estimated_mm": [0, 1.09, 2.00, 2.73, 3.38, 4.27, 5.69, 8.08],
        "real_mm": [0, 0, 2.07, 3.14, 3.56, 4.13, 5.05, 7.22],
    },
    "T8": {
        "cycle": [40000, 50000, 70000, 74883, 76931, 89237, 92315, 96475, 98492, 100774],
        "estimated_mm": [0, 0, 1.76, 2.32, 2.66, 3.61, 3.76, 4.59, 5.01, 5.87],
        "real_mm": [0, 0, 0, 1.94, 2.50, 3.71, 3.88, 4.61, 4.96, 5.52],
    },
}
PAPER_TOTAL_SCORE = 16.14


@dataclass
class Angler3rdPlacePipeline:
    data_root: Path
    v5_exponent_step: float | None = 0.1
    # Sec. 4.3.1 names T4 explicitly ("the T4 dataset contains most thorough
    # information about crack length growth"). Auto-picking by number of
    # nonzero-crack points ties T3 with T4 at 7 and resolves the tie
    # arbitrarily, so the donor is fixed here instead.
    donor_specimen: str | None = "T4"
    random_state: int = 0

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.specimens: dict[str, SpecimenData] = load_all(self.data_root)
        self.extractor = FeatureExtractor()
        self.feature_tables: dict[str, pd.DataFrame] = {}
        self.ensembles: dict[str, list[BSSModel]] = {}
        self.paris_fits: dict[str, ParisFit] = {}
        self.results: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------- features
    def build_feature_tables(self) -> None:
        for name, spec in self.specimens.items():
            self.feature_tables[name] = self.extractor.specimen_feature_table(spec)

    # ------------------------------------------------------- ensemble (Sec 3)
    def fit_ensembles(self) -> None:
        if not self.feature_tables:
            self.build_feature_tables()
        train_tables = {s: self.feature_tables[s] for s in TRAINING_SPECIMENS}
        for target, base_features in (("T7", T7_BASE_FEATURES), ("T8", T8_BASE_FEATURES)):
            models = ensemble_models(train_tables, base_features, self.v5_exponent_step)
            self.ensembles[target] = select_ensemble(models)

    def _crack_onset_cycle(self, name: str) -> int:
        """First cycle from which the ensemble regression is applied,
        instead of declaring crack = 0 (Table 1). Falls back to the second
        available cycle if the strict Table 1 algorithm never triggers --
        the algorithm is admittedly fragile (the paper's own detector
        produces false alarms), and this pipeline's independently-recovered
        features do not always reproduce the paper's exact trigger timing."""
        table = self.feature_tables[name]
        onset = detect_crack_onset_cycle(table[table["run"] == "signal_1"])
        if onset is not None:
            return onset
        cycles = sorted(table["cycle"].unique())
        return cycles[1] if len(cycles) > 1 else cycles[0]

    def estimate_signal_cycles(self, name: str) -> pd.DataFrame:
        """Crack-length estimates for a validation specimen's cycles that
        have wave signals (Table 1 detection + ensemble regression)."""
        table = self.feature_tables[name]
        onset = self._crack_onset_cycle(name)
        below = sorted(c for c in table["cycle"].unique() if c < onset)
        at_or_above = table[table["cycle"] >= onset]
        rows = [{"cycle": c, "estimated_crack_mm": 0.0} for c in below]
        if len(at_or_above):
            rows.extend(ensemble_predict(self.ensembles[name], at_or_above).to_dict("records"))
        return pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)

    # --------------------------------------------------------- Paris' Law (Sec 4)
    def _pick_donor(self) -> str:
        if self.donor_specimen:
            return self.donor_specimen
        counts = {
            s: int((self.specimens[s].description["crack_length_mm"] > 0).sum())
            for s in TRAINING_SPECIMENS
        }
        return max(counts, key=counts.get)

    def _max_observed_crack_mm(self) -> float:
        return max(self.specimens[s].description["crack_length_mm"].max() for s in TRAINING_SPECIMENS)

    def predict_t7(self, estimated: pd.DataFrame) -> pd.DataFrame:
        nonzero = estimated[estimated["estimated_crack_mm"] > 0]
        donor_name = self._pick_donor()
        donor = self.specimens[donor_name]
        donor_desc = donor.description[donor.description["crack_length_mm"] > 0]
        target_crack = self._max_observed_crack_mm()
        cycles, cracks = build_t7_initial_dataset(
            nonzero["cycle"].to_numpy(float),
            nonzero["estimated_crack_mm"].to_numpy(float),
            donor_cycles=donor_desc["cycle"].to_numpy(float),
            donor_cracks_mm=donor_desc["crack_length_mm"].to_numpy(float),
            target_crack_mm=target_crack,
        )
        self.paris_fits["T7"] = fit_paris_law(
            cycles, cracks, delta_sigma_mpa=DELTA_SIGMA_CONSTANT_MPA, seed=self.random_state
        )
        preds = self.paris_fits["T7"].predict(PREDICTION_CYCLES["T7"])
        return pd.DataFrame({"cycle": PREDICTION_CYCLES["T7"], "estimated_crack_mm": preds})

    def predict_t8(self, estimated: pd.DataFrame) -> pd.DataFrame:
        nonzero = estimated[estimated["estimated_crack_mm"] > 0]
        target_crack = self._max_observed_crack_mm()
        cycles, cracks = build_t8_initial_dataset(
            nonzero["cycle"].to_numpy(float), nonzero["estimated_crack_mm"].to_numpy(float), target_crack
        )
        self.paris_fits["T8"] = fit_paris_law(cycles, cracks, delta_sigma_mpa=None, seed=self.random_state)
        preds = self.paris_fits["T8"].predict(PREDICTION_CYCLES["T8"])
        return pd.DataFrame({"cycle": PREDICTION_CYCLES["T8"], "estimated_crack_mm": preds})

    # -------------------------------------------------------------- run/all
    def run(self) -> dict[str, pd.DataFrame]:
        self.build_feature_tables()
        self.fit_ensembles()
        for name, predictor in (("T7", self.predict_t7), ("T8", self.predict_t8)):
            estimated = self.estimate_signal_cycles(name)
            predicted = predictor(estimated)
            full = pd.concat([estimated, predicted], ignore_index=True).sort_values("cycle")
            full["method"] = ["estimated"] * len(estimated) + ["predicted"] * len(predicted)
            full["true_crack_mm"] = [self.specimens[name].crack_length(c) for c in full["cycle"]]
            full["specimen"] = name
            self.results[name] = full.reset_index(drop=True)
        return self.results

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("T7", "T8"):
            r = self.results[name]
            paper = PAPER_TABLE_A[name]
            rows.append(
                {
                    "specimen": name,
                    "rmse_mm": rmse_mm(r["estimated_crack_mm"], r["true_crack_mm"]),
                    "penalty": penalty_score(r["cycle"], r["estimated_crack_mm"], r["true_crack_mm"]),
                    "paper_rmse_mm": rmse_mm(paper["estimated_mm"], paper["real_mm"]),
                    "paper_penalty": penalty_score(paper["cycle"], paper["estimated_mm"], paper["real_mm"]),
                }
            )
        df = pd.DataFrame(rows)
        total = {
            "specimen": "total",
            "rmse_mm": np.nan,
            "penalty": df["penalty"].sum(),
            "paper_rmse_mm": np.nan,
            "paper_penalty": df["paper_penalty"].sum(),
        }
        return pd.concat([df, pd.DataFrame([total])], ignore_index=True)

    def score_tables(self) -> dict[str, pd.DataFrame]:
        return {
            name: score_table(r["cycle"], r["estimated_crack_mm"], r["true_crack_mm"])
            for name, r in self.results.items()
        }
