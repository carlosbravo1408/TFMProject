"""Ensemble linear regression with Best Subset Selection (Sec. 3.5).

For each validation specimen chosen out of T1-T6 (leave-one-specimen-out),
Best Subset Selection (BSS) fits every combination of k of the K candidate
features (base features + pairwise interactions) for k = 1..K, keeps the
smallest-validation-error model of each size, then keeps the single overall
best of those K models. This gives six specimen-indexed models; the three
with the smallest validation error are kept as the ensemble (Fig. 7).

T7's candidate set adds the (specimen-relative) cycle-number feature v5, and
per Sec. 3.5 the optimal power applied to v5 (0 to 1, step 0.02 in the
paper) is selected jointly with BSS. The full paper step of 0.02 makes the
already-exponential 2^15 subset search ~50x more expensive; the coarser
default here (``V5_EXPONENT_STEP = 0.1``) is a deliberate, documented
approximation of that grid, exposed as a parameter for anyone who wants
paper-fidelity at the cost of runtime. ``v5`` can be negative (cycles before
the specimen's own crack-initiation reference), so the exponent is applied
as a signed power (``sign(v5) * |v5|**p``) to keep it real-valued.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

V5_EXPONENT_STEP = 0.1
T8_BASE_FEATURES = ["v1", "v2", "v3", "v4"]
T7_BASE_FEATURES = ["v1", "v2", "v3", "v4", "v5"]


def signed_power(x: np.ndarray, p: float) -> np.ndarray:
    return np.sign(x) * np.abs(x) ** p


def pairwise_names(base: list[str]) -> list[str]:
    return [a + b for a, b in itertools.combinations(base, 2)]


def build_candidate_matrix(
    df: pd.DataFrame, base_features: list[str], v5_exponent: float = 1.0
) -> tuple[np.ndarray, list[str]]:
    """``[base features | pairwise interactions]`` design matrix (Sec. 3.5:
    "features and their intersections")."""
    values = {}
    for f in base_features:
        v = df[f].to_numpy(dtype=float)
        if f == "v5" and v5_exponent != 1.0:
            v = signed_power(v, v5_exponent)
        values[f] = v
    for a, b in itertools.combinations(base_features, 2):
        values[a + b] = values[a] * values[b]
    names = list(base_features) + pairwise_names(base_features)
    X = np.column_stack([values[n] for n in names])
    return X, names


def _design(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def _fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(_design(X), y, rcond=None)
    return coef


def _predict_ols(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return _design(X) @ coef


def _ols_stats(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> tuple[float, float]:
    """R-squared and the overall F-test p-value of an OLS fit (the two
    per-model statistics reported in Table 2 / Table 3)."""
    n, p = len(y), X.shape[1]
    pred = _predict_ols(X, coef)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof_resid = max(n - p - 1, 1)
    if ss_res <= 0 or p == 0:
        return r_squared, 0.0
    f_stat = (ss_tot - ss_res) / p / (ss_res / dof_resid)
    p_value = float(stats.f.sf(f_stat, p, dof_resid))
    return r_squared, p_value


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error over nonzero-crack rows only (Sec.
    3.5's "validation error"; undefined/excluded where the true crack
    length is zero)."""
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]))) * 100


@dataclass
class BSSModel:
    validation_specimen: str
    base_features: list[str]
    feature_names: list[str]
    v5_exponent: float
    validation_error: float
    r_squared: float
    p_value: float
    coefficients: np.ndarray
    n_effective_val_cycles: int

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, all_names = build_candidate_matrix(df, self.base_features, self.v5_exponent)
        idx = [all_names.index(n) for n in self.feature_names]
        return _predict_ols(X[:, idx], self.coefficients)


def best_subset_selection(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_specimen: str,
    base_features: list[str],
    v5_exponent_step: float | None = V5_EXPONENT_STEP,
) -> BSSModel:
    """Best Subset Selection for one leave-one-specimen-out fold."""
    y_train = train_df["crack_length_mm"].to_numpy(dtype=float)
    y_val = val_df["crack_length_mm"].to_numpy(dtype=float)

    if "v5" in base_features and v5_exponent_step is not None:
        exponents = list(np.round(np.arange(0.0, 1.0 + 1e-9, v5_exponent_step), 4))
        exponents = [e if e > 0 else 1e-6 for e in exponents]  # 0 -> constant column, skip degeneracy
    else:
        exponents = [1.0]

    best = None  # (val_error, idx, exponent, names)
    for exponent in exponents:
        X_train, names = build_candidate_matrix(train_df, base_features, exponent)
        X_val, _ = build_candidate_matrix(val_df, base_features, exponent)
        K = len(names)
        for k in range(1, K + 1):
            best_for_k = None
            for combo in itertools.combinations(range(K), k):
                idx = list(combo)
                coef = _fit_ols(X_train[:, idx], y_train)
                pred_val = _predict_ols(X_val[:, idx], coef)
                err = mape(y_val, pred_val)
                if np.isnan(err):
                    continue
                if best_for_k is None or err < best_for_k[0]:
                    best_for_k = (err, idx)
            if best_for_k is None:
                continue
            err, idx = best_for_k
            if best is None or err < best[0]:
                best = (err, idx, exponent, names)

    if best is None:
        raise ValueError(f"BSS found no valid model for validation specimen {val_specimen}")
    err, idx, exponent, names = best
    X_train, _ = build_candidate_matrix(train_df, base_features, exponent)
    selected_names = [names[i] for i in idx]
    coef = _fit_ols(X_train[:, idx], y_train)
    r_squared, p_value = _ols_stats(X_train[:, idx], y_train, coef)
    n_effective = int(val_df.loc[val_df["crack_length_mm"] > 0, "cycle"].nunique())
    return BSSModel(
        validation_specimen=val_specimen,
        base_features=list(base_features),
        feature_names=selected_names,
        v5_exponent=exponent,
        validation_error=err,
        r_squared=r_squared,
        p_value=p_value,
        coefficients=coef,
        n_effective_val_cycles=n_effective,
    )


def ensemble_models(
    tables: dict[str, pd.DataFrame],
    base_features: list[str],
    v5_exponent_step: float | None = V5_EXPONENT_STEP,
    n_specimens: list[str] = ("T1", "T2", "T3", "T4", "T5", "T6"),
) -> list[BSSModel]:
    """The six leave-one-specimen-out BSS models (Table 2 / Table 3)."""
    full = pd.concat([tables[s] for s in n_specimens], ignore_index=True)
    models = []
    for val_specimen in n_specimens:
        train_df = full[full["specimen"] != val_specimen]
        val_df = full[full["specimen"] == val_specimen]
        models.append(
            best_subset_selection(train_df, val_df, val_specimen, base_features, v5_exponent_step)
        )
    return models


def select_ensemble(models: list[BSSModel], n_keep: int = 3, min_val_cycles: int = 3) -> list[BSSModel]:
    """The three smallest-validation-error models are kept for the ensemble
    (Fig. 7: "Select 3 out of 6"), skipping models whose held-out specimen
    has too few nonzero-crack cycles. The paper explicitly excludes its
    smallest-error Model 5 for T7 on this basis ("the validation set of
    Model 5 contains only two effective data samples... may perform poorly
    when generalized to T7"), even though it beats models 2 and 6 (Table 2).
    """
    candidates = [m for m in models if m.n_effective_val_cycles >= min_val_cycles]
    if len(candidates) < n_keep:
        candidates = models
    return sorted(candidates, key=lambda m: m.validation_error)[:n_keep]


def ensemble_predict(models: list[BSSModel], df: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-run predictions for a target specimen's cycles, plus
    the final ensemble estimate: for each cycle, the larger of Run 1/Run 2
    per model, averaged across the selected models (Sec. 3.5, Fig. 7)."""
    preds = df[["specimen", "cycle", "run"]].copy()
    for i, model in enumerate(models):
        preds[f"model_{i}"] = model.predict(df)
    per_model_cols = [f"model_{i}" for i in range(len(models))]

    per_cycle = []
    for cycle, group in preds.groupby("cycle"):
        larger_per_model = [group[col].max() for col in per_model_cols]
        # A linear regression can output a small negative value near
        # zero-crack inputs; crack length is physically non-negative.
        per_cycle.append({"cycle": cycle, "estimated_crack_mm": max(0.0, float(np.mean(larger_per_model)))})
    return pd.DataFrame(per_cycle).sort_values("cycle").reset_index(drop=True)
