from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crack_estimator import CrackLengthEstimator, PAPER_BEST_PARAMS, build_feature_table
from data_loader import SpecimenData, TRAINING_SPECIMENS, load_all
from scoring import score_table
from signal_processing import FEATURE_NAMES
from trans_fitting import (
    CANDIDATES,
    TransFitCurve,
    goodness_of_fit_table,
    select_candidate,
    sequential_trans_fit,
)
from variable_loading import transform_cycles


@dataclass
class TrainedEstimator:
    model: CrackLengthEstimator
    training_table: pd.DataFrame
    train_rmse: float


def train_estimator(
    root, training_names: list[str] = TRAINING_SPECIMENS, params: dict | None = None
) -> TrainedEstimator:
    params = params or PAPER_BEST_PARAMS
    specimens = load_all(root, training_names)
    table = build_feature_table(specimens)
    X = table[FEATURE_NAMES].to_numpy()
    y = table["crack_length_mm"].to_numpy()
    model = CrackLengthEstimator(**params).fit(X, y)
    return TrainedEstimator(model=model, training_table=table, train_rmse=model.rmse(X, y))


# Table 7 prints 4.48 mm as the truth at cycle 51030, its own prediction; the
# dataset gives 4.13, which reproduces the published 43.90 penalty.
PAPER_TABLES = {
    "T7": {
        "cycle": [36001, 40167, 44054, 47022, 49026, 51030, 53019, 55031],
        "predicted_mm": [0.0, 0.0, 1.92, 3.08, 3.59, 4.48, 5.67, 7.21],
        "printed_true_mm": [0.0, 0.0, 2.07, 3.14, 3.56, 4.48, 5.05, 7.22],
        "final_crack_mm": 7.22,
    },
    "T8": {
        "cycle": [40000, 50000, 70000, 74883, 76931, 89237, 92315, 96475, 98492, 100774],
        "predicted_mm": [0.0, 0.0, 1.37, 1.94, 2.51, 3.70, 4.09, 4.70, 5.02, 5.41],
        "printed_true_mm": [0.0, 0.0, 0.0, 1.94, 2.5, 3.71, 3.88, 4.61, 4.96, 5.52],
        "final_crack_mm": 5.52,
    },
}

# Not published by the paper; not tuned on T7/T8 to avoid selecting on them.
TRANSLOCATION_LAMBDA = 30000.0


def estimate_specimen(
    trained: TrainedEstimator, sd: SpecimenData, baseline_override: int | None = None
) -> pd.DataFrame:
    table = build_feature_table(
        {sd.name: sd}, baseline_overrides={sd.name: baseline_override}
    )
    X = table[FEATURE_NAMES].to_numpy()
    table["estimated"] = np.clip(trained.model.predict(X), 0.0, None)
    reference_cycle = baseline_override or sd.undamaged_baseline_cycle()
    # Tables 5 and 8 report zero up to the reference cycle inclusive.
    table.loc[table["cycle"] <= reference_cycle, "estimated"] = 0.0
    return table


def fit_reference_curve(
    sd: SpecimenData,
    lam_fit: float = 1e-4,
    warp_exponent: float | None = None,
    min_dfe: int = 2,
) -> tuple[str, TransFitCurve, np.ndarray, np.ndarray]:
    N0 = sd.undamaged_baseline_cycle()
    cycles = sd.post_initiation_cycles()
    N = np.array(cycles, dtype=float)
    a = np.array([sd.crack_length(c) for c in cycles], dtype=float)
    if warp_exponent is not None:
        N = transform_cycles(N, N0, warp_exponent)
    # Specimens initiate at very different absolute cycles; fitting in cycles since
    # initiation keeps translocated curves inside their fitted domain.
    N_rel = N - N0
    gof = goodness_of_fit_table(N_rel, a, lam=lam_fit)
    best = select_candidate(gof, min_dfe=min_dfe)
    candidate = CANDIDATES[best["candidate"]]
    curve = TransFitCurve(candidate, best["theta"])
    return best["candidate"], curve, N_rel, a


def predict_specimen(
    trained: TrainedEstimator,
    sd: SpecimenData,
    reference_names: list[str],
    all_specimens: dict[str, SpecimenData],
    lam_translocate: float = 30000.0,
    variable_loading_exponent: float | None = None,
    baseline_override: int | None = None,
) -> dict:
    est_table = estimate_specimen(trained, sd, baseline_override=baseline_override)
    known_cycles = est_table["cycle"].tolist()
    known_estimates = est_table["estimated"].tolist()
    target_N0 = sd.undamaged_baseline_cycle(baseline_override)

    curves = []
    selections = {}
    for ref_name in reference_names:
        ref_sd = all_specimens[ref_name]
        cand_name, curve, N_fit, a_fit = fit_reference_curve(
            ref_sd, warp_exponent=variable_loading_exponent
        )
        curves.append(curve)
        selections[ref_name] = cand_name

    target_cycles = [
        c for c in sd.description["cycle"].tolist() if c not in known_cycles
    ]
    # Eq. 17 anchors on every non-zero estimate, not only the initial one the prose
    # of Sec. 4 mentions.
    initial_anchors_rel = [
        (cycle - target_N0, estimate)
        for cycle, estimate in zip(known_cycles, known_estimates)
        if estimate > 0
    ]
    if not initial_anchors_rel:
        initial_anchors_rel = [(known_cycles[-1] - target_N0, known_estimates[-1])]
    target_cycles_rel = [c - target_N0 for c in target_cycles]
    final_predictions_rel, iteration_tables_rel = sequential_trans_fit(
        curves, initial_anchors_rel, target_cycles_rel, lam=lam_translocate
    )
    final_predictions = {
        c: final_predictions_rel[c - target_N0] for c in target_cycles
    }
    iteration_tables = [
        {c + target_N0: v for c, v in table.items()} for table in iteration_tables_rel
    ]

    all_cycles = known_cycles + target_cycles
    all_values = known_estimates + [final_predictions[c] for c in target_cycles]
    all_true = [sd.crack_length(c) for c in all_cycles]
    scores = score_table(all_cycles, all_values, all_true)

    return {
        "estimation_table": est_table,
        "reference_selections": selections,
        "target_cycles": target_cycles,
        "final_predictions": final_predictions,
        "iteration_tables": iteration_tables,
        "score_table": scores,
    }
