from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ALL_SPECIMENS, load_curves
from .models import LAWS
from .objectives import phm_penalty
from .pooled import fit_coefficient_only

RESULTS = Path(__file__).resolve().parent / "results"


def sweep(laws=("paris", "walker", "paris_hole"), specimens=("T7", "T8", "T1", "T3", "T4", "T6")):
    best = json.loads((RESULTS / "stage_b_pooled_best.json").read_text())
    curves = load_curves(ALL_SPECIMENS)
    rows = []
    for law_name in laws:
        law = LAWS[law_name]
        shared = np.asarray(best[law_name]["shared_vector"], dtype=float)
        for spec in specimens:
            curve = curves[spec]
            n_total = len(curve.crack_mm)
            for n_anchor in range(2, n_total):
                log_c, pred = fit_coefficient_only(law, shared, curve, n_points=n_anchor)
                out = slice(n_anchor, None)
                true = curve.crack_mm
                rows.append(
                    {
                        "law": law_name, "specimen": spec, "n_anchor": n_anchor,
                        "n_predicted": n_total - n_anchor,
                        "log10_coefficient": log_c,
                        "rmse_holdout_mm": float(np.sqrt(np.mean((pred[out] - true[out]) ** 2))),
                        "phm_penalty_holdout": float(
                            phm_penalty(pred[None, out], true[out], curve.final_crack_mm)[0]
                        ),
                        "runaway": bool(pred[-1] > 2.0 * curve.final_crack_mm),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    df = sweep()
    df.to_csv(RESULTS / "stage_f_anchor_sweep.csv", index=False)
    for law_name, g in df.groupby("law"):
        print(f"\n=== {law_name} ===")
        pivot = g.pivot_table(index="n_anchor", columns="specimen",
                              values="phm_penalty_holdout")
        print("Penalización oficial sobre los puntos predichos a ciegas:")
        print(pivot.round(2).to_string())
        rm = g.pivot_table(index="n_anchor", columns="specimen", values="rmse_holdout_mm")
        print("RMSE (mm) sobre los puntos predichos a ciegas:")
        print(rm.round(3).to_string())
    print(f"\n-> {RESULTS / 'stage_f_anchor_sweep.csv'}")


if __name__ == "__main__":
    main()
