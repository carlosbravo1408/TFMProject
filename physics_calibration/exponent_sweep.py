from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import ALL_SPECIMENS, CALIBRATION_SPECIMENS, VALIDATION_SPECIMENS, load_curves
from .models import ParisLaw
from .objectives import phm_penalty
from .pooled import fit_coefficient_only

RESULTS = Path(__file__).resolve().parent / "results"

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
    # Scored on the training specimens only: selecting on T7/T8 would be leakage.
    pivot = df.pivot(index="m", columns="n_anchor", values="penalty_train")
    worst = pivot.max(axis=1)
    admissible = worst[worst.index >= 2.0]
    print(f"\nElección minimax sobre presupuestos de anclaje:")
    print(f"  sin restringir : m = {worst.idxmin():.2f}  (peor caso {worst.min():.2f})")
    print(f"  con m >= 2     : m = {admissible.idxmin():.2f}  (peor caso {admissible.min():.2f})")
    print(f"\n-> {RESULTS / 'stage_g_exponent_sweep.csv'}")


if __name__ == "__main__":
    main()
