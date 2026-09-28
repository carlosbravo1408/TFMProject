from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.svm import SVR

from data_loader import SpecimenData
from signal_processing import FEATURE_NAMES, FeatureExtractor

PAPER_BEST_PARAMS = {"kernel": "rbf", "C": 100.0, "gamma": 1.0}


def build_feature_table(
    specimens: dict[str, SpecimenData],
    baseline_overrides: dict[str, int] | None = None,
    extractor: FeatureExtractor | None = None,
) -> pd.DataFrame:
    baseline_overrides = baseline_overrides or {}
    extractor = extractor or FeatureExtractor()
    rows = []
    for name, sd in specimens.items():
        baseline_cycle = sd.undamaged_baseline_cycle(baseline_overrides.get(name))
        baseline_signals = sd.signals(baseline_cycle)
        for cycle in sd.labeled_cycles():
            signals = sd.signals(cycle)
            feats = extractor.cycle_features(signals, baseline_signals)
            rows.append(
                {
                    "specimen": name,
                    "cycle": cycle,
                    **dict(zip(FEATURE_NAMES, feats)),
                    "crack_length_mm": sd.crack_length(cycle),
                }
            )
    return pd.DataFrame(rows)


class CrackLengthEstimator:
    def __init__(self, C: float = 100.0, gamma: float = 1.0, kernel: str = "rbf",
                 epsilon: float = 0.1):
        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CrackLengthEstimator":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y, self.predict(X))))


def grid_search(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    param_grid: dict | None = None,
) -> GridSearchCV:
    param_grid = param_grid or {
        "C": [1, 10, 50, 100, 200, 500],
        "gamma": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
        "kernel": ["rbf"],
    }
    cv = LeaveOneGroupOut()
    search = GridSearchCV(
        SVR(),
        param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X, y, groups=groups)
    return search
