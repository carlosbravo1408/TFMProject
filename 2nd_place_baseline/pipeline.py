from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from crack_estimator import RandomForestCrackEstimator
from data_loader import (
    DATA_DRIVEN_SPECIMENS,
    ENSEMBLE_SPECIMENS,
    SpecimenData,
    load_all,
)
from features import FeatureExtractor, OPTIMAL_FEATURES
from physics_models import (
    EnsemblePrognostics,
    WalkerMonteCarlo,
    fit_double_exponential,
    linear_regression_extrapolation,
    observed_rate_bound,
)
from scoring import penalty_score, rmse_mm, score_table

PREDICTION_CYCLES = {
    "T7": [49026, 51030, 53019, 55031],
    "T8": [89237, 92315, 96475, 98492, 100774],
}

PAPER_PREDICTIONS = {
    "T7": {36001: 0.0, 40167: 0.0, 44054: 2.175, 47022: 3.017,
           49026: 3.423, 51030: 4.310, 53019: 5.547, 55031: 7.170},
    "T8": {40000: 0.0, 50000: 0.0, 70000: 1.722, 74883: 2.291, 76931: 2.565,
           89237: 3.630, 92315: 3.930, 96475: 4.571, 98492: 4.956, 100774: 5.500},
}


@dataclass
class HybridPipeline:
    data_root: Path
    features: list[str] = field(default_factory=lambda: list(OPTIMAL_FEATURES))
    ensemble_sigma_mm: float = 1.0
    walker_models: int = 100
    random_state: int = 0

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.specimens: dict[str, SpecimenData] = load_all(
            self.data_root, ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
        )
        self.extractor = FeatureExtractor()
        self.estimator = RandomForestCrackEstimator(self.features, random_state=self.random_state)
        self.feature_tables: dict[str, pd.DataFrame] = {}

    def build_feature_tables(self) -> pd.DataFrame:
        self.extractor.set_alignment_anchor(self.specimens)
        for name, spec in self.specimens.items():
            labeled_only = name not in ("T7", "T8")
            self.feature_tables[name] = self.extractor.specimen_feature_table(
                spec, labeled_only=labeled_only
            )
        return pd.concat(self.feature_tables.values(), ignore_index=True)

    def training_table(self) -> pd.DataFrame:
        if not self.feature_tables:
            self.build_feature_tables()
        return pd.concat(
            [self.feature_tables[s] for s in DATA_DRIVEN_SPECIMENS], ignore_index=True
        )

    def run_data_driven(
        self, max_depth: int | None = None, n_trees: int | None = None
    ) -> dict[str, pd.DataFrame]:
        train = self.training_table()
        self.estimator.fit(train, max_depth=max_depth, n_trees=n_trees)
        self.estimates = {
            name: self.estimator.estimate(self.feature_tables[name]) for name in ("T7", "T8")
        }
        return self.estimates

    def _normalized_history(self, name: str) -> tuple[int, np.ndarray, np.ndarray]:
        est = self.estimates[name]
        nonzero = est[est["estimated_crack_mm"] > 0]
        if nonzero.empty:
            raise ValueError(
                f"No crack initiation detected for {name}: every data-driven "
                "estimate is below the detection threshold, so the physics-based "
                "method has no nonzero history to start from."
            )
        n_initial = int(nonzero["cycle"].iloc[0])
        return (
            n_initial,
            nonzero["cycle"].to_numpy(float) - n_initial,
            nonzero["estimated_crack_mm"].to_numpy(float),
        )

    def run_physics_t7(self) -> pd.DataFrame:
        curvas = {}
        for name in ENSEMBLE_SPECIMENS:
            spec = self.specimens[name]
            desc = spec.description[spec.description["crack_length_mm"] > 0]
            n0 = spec.initiation_cycle()
            curvas[name] = (desc["cycle"].to_numpy(float) - n0,
                            desc["crack_length_mm"].to_numpy(float))
        self.rate_bound = observed_rate_bound(curvas)
        self.exponential_models = [
            fit_double_exponential(name, *curvas[name], rate_ub=self.rate_bound)
            for name in ENSEMBLE_SPECIMENS
        ]
        self.t7_ensemble = EnsemblePrognostics(self.exponential_models, self.ensemble_sigma_mm)
        n_initial, cycles, cracks = self._normalized_history("T7")
        target = np.array(PREDICTION_CYCLES["T7"], float) - n_initial
        preds = self.t7_ensemble.predict(cycles[-1], cracks[-1], target)
        return pd.DataFrame(
            {"specimen": "T7", "cycle": PREDICTION_CYCLES["T7"], "estimated_crack_mm": preds}
        )

    def run_physics_t8(self) -> pd.DataFrame:
        n_initial, cycles, cracks = self._normalized_history("T8")
        # The paper extends the estimates by linear regression to cycles 89237 and 92315
        # so that the Walker fit has five points.
        lr_cycles = np.array(PREDICTION_CYCLES["T8"][:2], float) - n_initial
        lr_cracks = linear_regression_extrapolation(cycles, cracks, lr_cycles)
        fit_cycles = np.concatenate([cycles, lr_cycles])
        fit_cracks = np.concatenate([cracks, lr_cracks])
        self.t8_walker = WalkerMonteCarlo(
            fit_cycles, fit_cracks, n_models=self.walker_models, seed=self.random_state
        )
        self.t8_walker.fit()
        target = np.array(PREDICTION_CYCLES["T8"], float) - n_initial
        preds = self.t8_walker.predict(target)
        return pd.DataFrame(
            {"specimen": "T8", "cycle": PREDICTION_CYCLES["T8"], "estimated_crack_mm": preds}
        )

    def run(self, max_depth: int | None = None, n_trees: int | None = None) -> pd.DataFrame:
        self.run_data_driven(max_depth=max_depth, n_trees=n_trees)
        physics = {"T7": self.run_physics_t7(), "T8": self.run_physics_t8()}
        rows = []
        for name in ("T7", "T8"):
            spec = self.specimens[name]
            full = pd.concat([self.estimates[name], physics[name]], ignore_index=True)
            full["method"] = ["data-driven"] * len(self.estimates[name]) + [
                "physics-based"
            ] * len(physics[name])
            full["true_crack_mm"] = [spec.crack_length(c) for c in full["cycle"]]
            full["paper_prediction_mm"] = [PAPER_PREDICTIONS[name][c] for c in full["cycle"]]
            rows.append(full)
        self.results = pd.concat(rows, ignore_index=True)
        return self.results

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("T7", "T8"):
            sub = self.results[self.results["specimen"] == name]
            rows.append(
                {
                    "specimen": name,
                    "rmse_mm": rmse_mm(sub["estimated_crack_mm"], sub["true_crack_mm"]),
                    "penalty": penalty_score(
                        sub["cycle"], sub["estimated_crack_mm"], sub["true_crack_mm"]
                    ),
                    "paper_rmse_mm": rmse_mm(sub["paper_prediction_mm"], sub["true_crack_mm"]),
                    "paper_penalty": penalty_score(
                        sub["cycle"], sub["paper_prediction_mm"], sub["true_crack_mm"]
                    ),
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
            name: score_table(
                self.results.loc[self.results["specimen"] == name, "cycle"],
                self.results.loc[self.results["specimen"] == name, "estimated_crack_mm"],
                self.results.loc[self.results["specimen"] == name, "true_crack_mm"],
            )
            for name in ("T7", "T8")
        }
