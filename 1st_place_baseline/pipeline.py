"""High-level orchestration tying together estimation (SVR) and prediction
(trans-fitting) into the full method of the paper, for a validation
specimen (T7 or T8).
"""
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


def estimate_specimen(
    trained: TrainedEstimator, sd: SpecimenData, baseline_override: int | None = None
) -> pd.DataFrame:
    """Estimated crack length for every cycle of ``sd`` that has a wave signal
    (Section 4.1/4.2 first step, reproduces Tables 5 and 8)."""
    table = build_feature_table(
        {sd.name: sd}, baseline_overrides={sd.name: baseline_override}
    )
    X = table[FEATURE_NAMES].to_numpy()
    table["estimated"] = trained.model.predict(X)
    return table


def fit_reference_curve(
    sd: SpecimenData,
    lam_fit: float = 1e-4,
    warp_exponent: float | None = None,
    min_dfe: int = 2,
) -> tuple[str, TransFitCurve, np.ndarray, np.ndarray]:
    """Select the best-fitting candidate function for one training specimen's
    post-initiation crack-growth curve and fit it (Section 3.2.1/3.2.2).

    Fitting happens in *cycles since this specimen's own crack initiation*
    (``N - N0``), not raw absolute cycle counts: different specimens start
    their fatigue test and initiate cracks at very different absolute cycle
    counts (e.g. T3 ~57000 vs T7 ~40000), so a curve translocated using
    T7's absolute cycles onto a shape fit on T3's absolute cycles would be
    evaluated far outside its fitted domain and can diverge under
    extrapolation. "Cycles since initiation" is the common, physically
    comparable timeline the trans-fitting/translocation step of the paper
    implicitly assumes. If ``warp_exponent`` is given (T8's variable-loading
    correction, Eq. 24), the absolute cycles are warped first -- the warp's
    fixed point is N0, so this composes cleanly with the N0-relative shift.
    """
    N0 = sd.undamaged_baseline_cycle()
    cycles = sd.post_initiation_cycles()
    N = np.array(cycles, dtype=float)
    a = np.array([sd.crack_length(c) for c in cycles], dtype=float)
    if warp_exponent is not None:
        N = transform_cycles(N, N0, warp_exponent)
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
    """Full estimate-then-predict pipeline for a validation specimen,
    reproducing Sections 4.1 (T7) / 4.2 (T8).

    If ``variable_loading_exponent`` is given, the reference specimens'
    cycles are warped via Eq. 24 before fitting their curves (T8 case).
    ``baseline_override`` reproduces the paper's documented exception for
    T8 (Section 4.2): cycle 50000 is used as the undamaged reference signal
    even though the optical crack-length measurement is still 0 mm at the
    later cycle 70000, because the wave signal itself already shows
    damage-like change by then.
    """
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
    # The trans-fitting/translocation step (Section 3.2.3) anchors on "the
    # initial crack length of the target specimen" (singular) -- i.e. only
    # the most recently estimated point, not every previously estimated
    # cycle. Everything downstream happens in "cycles since this target
    # specimen's own crack initiation" (see fit_reference_curve).
    last_known_cycle, last_known_estimate = known_cycles[-1], known_estimates[-1]
    initial_anchors_rel = [(last_known_cycle - target_N0, last_known_estimate)]
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
