"""Which exponent is best for *prognosis*, as opposed to best for *fitting*?

Stage C of ``calibrate.py`` shows the in-sample objective is nearly flat in m:
every value from 1 to 7 fits the training curves within 5 % of the optimum,
because dK spans barely a third of a decade. That flatness is not a licence to
pick any m, because extrapolation is not flat at all — the same exponent that
governs the fit governs how violently the multi-step integrator accelerates
past the last observation.

This module therefore re-runs the choice under the criterion that matters:
freeze m, calibrate only the coefficient on the first ``n_anchor``
measurements, predict the rest, and score with the official challenge penalty.
It is the number that decides what the TFM's PINN should carry as its physics
prior.

Run:  python -m physics_calibration.exponent_sweep
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import ALL_SPECIMENS, CALIBRATION_SPECIMENS, VALIDATION_SPECIMENS, load_curves
from .models import ParisLaw
from .objectives import phm_penalty
from .pooled import fit_coefficient_only

RESULTS = Path(__file__).resolve().parent / "results"

# Paris with the plain-plate geometry. The hole-corrected variants are
# excluded here: the anchor sweep shows they extrapolate far worse at every
# anchor budget, so their exponent is not a candidate for the physics prior.
PARIS = ParisLaw("paris", ("C", "m"), ((-20.0, -5.0), (1.0, 8.0)), log_params=("C",))


def sweep(m_values=np.arange(1.0, 6.01, 0.25), anchors=(3, 4)) -> pd.DataFrame:
    curves = load_curves(ALL_SPECIMENS)
    rows = []
    for m in m_values:
        for n_anchor in anchors:
            acc = {"train": 0.0, "validation": 0.0}
            rmses = []
            for spec in CALIBRATION_SPECIMENS + VALIDATION_SPECIMENS:
                curve = curves[spec]
                if len(curve.crack_mm) <= n_anchor:
                    continue
                _, pred = fit_coefficient_only(PARIS, np.array([m]), curve, n_points=n_anchor)
                out = slice(n_anchor, None)
                pen = float(
                    phm_penalty(pred[None, out], curve.crack_mm[out], curve.final_crack_mm)[0]
                )
                rmses.append(np.sqrt(np.mean((pred[out] - curve.crack_mm[out]) ** 2)))
                acc["validation" if spec in VALIDATION_SPECIMENS else "train"] += pen
            rows.append(
                {"m": float(m), "n_anchor": n_anchor,
                 "penalty_T7_T8": acc["validation"], "penalty_train": acc["train"],
                 "rmse_mean_mm": float(np.mean(rmses))}
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = sweep()
    df.to_csv(RESULTS / "stage_g_exponent_sweep.csv", index=False)
    for n_anchor, g in df.groupby("n_anchor"):
        best_v = g.loc[g["penalty_T7_T8"].idxmin()]
        best_t = g.loc[g["penalty_train"].idxmin()]
        print(f"\n--- {n_anchor} puntos de anclaje ---")
        print(g[["m", "penalty_T7_T8", "penalty_train", "rmse_mean_mm"]].round(3).to_string(index=False))
        print(f"  Mejor para T7+T8: m = {best_v['m']}  (penalización {best_v['penalty_T7_T8']:.2f})")
        print(f"  Mejor en entrenamiento: m = {best_t['m']}  (penalización {best_t['penalty_train']:.2f})")
    # Robust choice: the exponent whose worst penalty across anchor budgets is
    # smallest, restricted to the classical metallic window m >= 2.
    #
    # Scored on **penalty_train** — the training specimens only. Using
    # penalty_T7_T8 here would select the exponent by its performance on the
    # evaluation specimens, which is leakage; it was the bug of the first
    # version of this module and of make_priors.py. The T7+T8 column is still
    # computed and printed above, but only as an after-the-fact observation.
    pivot = df.pivot(index="m", columns="n_anchor", values="penalty_train")
    worst = pivot.max(axis=1)
    admissible = worst[worst.index >= 2.0]
    print(f"\nElección minimax sobre presupuestos de anclaje:")
    print(f"  sin restringir : m = {worst.idxmin():.2f}  (peor caso {worst.min():.2f})")
    print(f"  con m >= 2     : m = {admissible.idxmin():.2f}  (peor caso {admissible.min():.2f})")
    print(f"\n-> {RESULTS / 'stage_g_exponent_sweep.csv'}")


if __name__ == "__main__":
    main()
