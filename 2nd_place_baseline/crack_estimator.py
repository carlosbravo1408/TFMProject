"""Data-driven crack-length estimation with a random forest (Sections 3.2.3-3.2.4).

- Random forest regression (scikit-learn), whose two key hyper-parameters are
  the maximum depth and the number of trees.
- Grid-search hyper-parameter optimization over n_depth = {1, 2, 3} and
  n_trees = {5, 10, 15, 20, 25}, minimizing the performance metric
  PM = (1/5) * sum_i RMSE_T(i) of a specimen-wise k-fold cross validation
  (each of T1, T2, T3, T4, T6 is held out once; T5 is excluded as an outlier).
- Feature-subset selection by repeatedly running the grid search on candidate
  subsets and comparing PM.
- The final estimator is an ensemble of 20 independently seeded random forest
  models (the RF is stochastic); the estimate is their average.
- For validation specimens, the ``prev_crack`` feature is filled recursively
  with the estimate of the previous cycle (starting at 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from features import FEATURE_NAMES, OPTIMAL_FEATURES

DEPTH_GRID = (1, 2, 3)
TREES_GRID = (5, 10, 15, 20, 25)
N_ENSEMBLE_MODELS = 20
CV_SPECIMENS = ("T1", "T2", "T3", "T4", "T6")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


@dataclass
class GridSearchResult:
    features: list[str]
    max_depth: int
    n_trees: int
    pm: float
    fold_rmse: dict[str, float] = field(default_factory=dict)


class RandomForestCrackEstimator:
    """K-fold-validated random forest ensemble for crack-length estimation."""

    def __init__(
        self,
        features: list[str] | None = None,
        n_models: int = N_ENSEMBLE_MODELS,
        random_state: int = 0,
    ) -> None:
        self.features = list(features or OPTIMAL_FEATURES)
        self.n_models = n_models
        self.random_state = random_state
        self.max_depth: int | None = None
        self.n_trees: int | None = None
        self.models: list[RandomForestRegressor] = []
        self.scaler_mean: pd.Series | None = None
        self.scaler_std: pd.Series | None = None

    # ------------------------------------------------------------------ utils
    def _fit_scaler(self, table: pd.DataFrame) -> None:
        cols = [c for c in FEATURE_NAMES]
        self.scaler_mean = table[cols].mean()
        self.scaler_std = table[cols].std(ddof=0).replace(0, 1.0)

    def _standardize(self, table: pd.DataFrame) -> pd.DataFrame:
        cols = list(self.scaler_mean.index)
        out = table.copy()
        out[cols] = (table[cols] - self.scaler_mean) / self.scaler_std
        return out

    def _standardize_value(self, column: str, value: float) -> float:
        return (value - self.scaler_mean[column]) / self.scaler_std[column]

    # ----------------------------------------------------------- k-fold / PM
    def _recursive_predict_fold(
        self, model: RandomForestRegressor, test_std: pd.DataFrame, features: list[str]
    ) -> np.ndarray:
        """Predict a held-out specimen chronologically, feeding back the
        previous estimate through the (standardized) ``prev_crack`` feature —
        the same procedure later applied to T7/T8."""
        preds = []
        prev_estimate = 0.0
        X = test_std[features].to_numpy().copy()
        prev_idx = features.index("prev_crack") if "prev_crack" in features else None
        for i in range(len(X)):
            if prev_idx is not None:
                X[i, prev_idx] = self._standardize_value("prev_crack", prev_estimate)
            pred = max(float(model.predict(X[i : i + 1])[0]), 0.0)
            preds.append(pred)
            prev_estimate = pred
        return np.asarray(preds)

    def kfold_pm(
        self,
        table: pd.DataFrame,
        features: list[str],
        max_depth: int,
        n_trees: int,
        random_state: int = 0,
    ) -> tuple[float, dict[str, float]]:
        """Performance metric PM (Eq. 5): mean held-out-specimen RMSE."""
        fold_rmse: dict[str, float] = {}
        for test_specimen in CV_SPECIMENS:
            train = table[table["specimen"] != test_specimen]
            test = table[table["specimen"] == test_specimen].sort_values("cycle")
            self._fit_scaler(train)
            train_std = self._standardize(train)
            test_std = self._standardize(test)
            model = RandomForestRegressor(
                n_estimators=n_trees, max_depth=max_depth, random_state=random_state
            )
            model.fit(train_std[features], train["crack_length_mm"])
            preds = self._recursive_predict_fold(model, test_std, features)
            fold_rmse[test_specimen] = rmse(test["crack_length_mm"].to_numpy(), preds)
        pm = float(np.mean(list(fold_rmse.values())))
        return pm, fold_rmse

    def grid_search(
        self, table: pd.DataFrame, features: list[str] | None = None, random_state: int = 0
    ) -> GridSearchResult:
        """Grid-search hyper-parameter optimization (Section 3.2.4)."""
        features = list(features or self.features)
        best: GridSearchResult | None = None
        for depth in DEPTH_GRID:
            for trees in TREES_GRID:
                pm, folds = self.kfold_pm(table, features, depth, trees, random_state)
                if best is None or pm < best.pm:
                    best = GridSearchResult(features, depth, trees, pm, folds)
        return best

    def feature_selection(
        self,
        table: pd.DataFrame,
        candidate_features: list[str] = FEATURE_NAMES,
        subset_sizes: tuple[int, ...] = (3, 4, 5, 6),
        random_state: int = 0,
    ) -> pd.DataFrame:
        """Search feature subsets, running the full hyper-parameter grid search
        on each (the paper samples subsets randomly and compares PM; here the
        search is exhaustive over the given subset sizes, which contains the
        paper's random search). Returns results sorted by PM."""
        results = []
        for size in subset_sizes:
            for combo in combinations(candidate_features, size):
                res = self.grid_search(table, list(combo), random_state)
                results.append(
                    {
                        "features": ", ".join(combo),
                        "n_features": size,
                        "max_depth": res.max_depth,
                        "n_trees": res.n_trees,
                        "pm": res.pm,
                    }
                )
        return pd.DataFrame(results).sort_values("pm").reset_index(drop=True)

    # ------------------------------------------------------------------- fit
    def fit(self, table: pd.DataFrame, max_depth: int | None = None, n_trees: int | None = None):
        """Fit the 20-model ensemble on all training rows. If hyper-parameters
        are not given, they are chosen by the grid search."""
        if max_depth is None or n_trees is None:
            best = self.grid_search(table, self.features, self.random_state)
            max_depth, n_trees = best.max_depth, best.n_trees
        self.max_depth, self.n_trees = max_depth, n_trees
        self._fit_scaler(table)
        train_std = self._standardize(table)
        self.models = []
        for k in range(self.n_models):
            model = RandomForestRegressor(
                n_estimators=n_trees, max_depth=max_depth, random_state=self.random_state + k
            )
            model.fit(train_std[self.features], table["crack_length_mm"])
            self.models.append(model)
        return self

    def _ensemble_predict_row(self, x: np.ndarray) -> float:
        preds = [m.predict(x.reshape(1, -1))[0] for m in self.models]
        return max(float(np.mean(preds)), 0.0)

    def estimate(
        self, table: pd.DataFrame, detection_threshold_mm: float = 2.0
    ) -> pd.DataFrame:
        """Recursive ensemble estimation for a validation specimen's feature
        table (chronological).

        ``detection_threshold_mm`` realizes the crack-initiation decision of
        Section 3.3.1: the paper reports exactly 0 for the validation cycles
        before N_initial (36001/40167 for T7, 40000/50000/pre-70000 for T8)
        and takes N_initial as the first cycle with a nonzero *estimated*
        crack. Until the ensemble output exceeds the threshold the crack is
        considered not yet initiated and the estimate (and the recursive
        ``prev_crack`` feature) is 0. The default of 2.0 mm is the order of
        the smallest measured cracks at first damage in the training set
        (1.61-3.25 mm) and reproduces the paper's N_initial for both T7
        (44054) and T8 (70000). Since crack growth is monotonic, a single
        above-threshold output followed by sub-threshold ones is treated as a
        false alarm: N_initial is the earliest cycle from which the raw
        ensemble output *stays* at or above the threshold. Once initiated,
        the raw ensemble output is reported unthresholded.

        The detection only affects the *reported* estimate; the recursive
        ``prev_crack`` state keeps tracking the raw ensemble output, so a
        sub-threshold early estimate still informs the next cycle."""
        table = table.sort_values("cycle").reset_index(drop=True)
        table_std = self._standardize(table)
        X = table_std[self.features].to_numpy().copy()
        prev_idx = self.features.index("prev_crack") if "prev_crack" in self.features else None
        raw_estimates = []
        prev_estimate = 0.0
        for i in range(len(X)):
            if prev_idx is not None:
                X[i, prev_idx] = self._standardize_value("prev_crack", prev_estimate)
            raw = self._ensemble_predict_row(X[i])
            raw_estimates.append(raw)
            prev_estimate = raw
        raw_arr = np.asarray(raw_estimates)
        above = raw_arr >= detection_threshold_mm
        sustained = np.logical_and.accumulate(above[::-1])[::-1]
        initiation_idx = int(np.argmax(sustained)) if sustained.any() else len(raw_arr)
        estimates = [raw if i >= initiation_idx else 0.0 for i, raw in enumerate(raw_estimates)]
        out = table[["specimen", "cycle"]].copy()
        out["estimated_crack_mm"] = estimates
        out["raw_ensemble_mm"] = raw_estimates
        return out
