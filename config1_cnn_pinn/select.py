from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import build_specimen_batches, load_labels
from .train import Config, RESULTS, loso_cross_validation

GRID = {
    # Fixed to the third stage's exponent, so the physics weights are selected on the
    # same closed-form branch used afterwards.
    "m_exponent": (2.00,),
    "epochs": (100, 600),
    "dropout": (0.3,),
    "weight_decay": (1e-3,),
    "lambda_physics": (0.0, 1.0, 3.0),
    "lambda_log_c_prior": (0.0, 1.0),
    "data_loss": ("asym",),
}


def search(batches, grid=GRID, n_seeds: int = 1) -> pd.DataFrame:
    from .evaluate import loso_challenge_score

    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        overrides = dict(zip(keys, values))
        cfg = Config(**overrides)
        t0 = time.perf_counter()
        rmse_rows, _ = loso_cross_validation(cfg, batches, n_seeds=n_seeds)
        protocol = loso_challenge_score(cfg, batches, n_seeds=n_seeds)
        df = pd.DataFrame(rmse_rows)
        rows.append({
            **overrides,
            "penalizacion_media": protocol["penalizacion"].mean(),
            "penalizacion_peor_fold": protocol["penalizacion"].max(),
            "penalizacion_prognosis": protocol["penalizacion_prognosis"].mean(),
            "rmse_estimacion_mm": df["rmse_mm"].mean(),
            "error_log10C": (protocol["log10_C_predicho"]
                             - protocol["log10_C_implicado"]).abs().mean(),
            "segundos": time.perf_counter() - t0,
        })
        print(f"  {overrides}\n      penalización media {rows[-1]['penalizacion_media']:8.2f} "
              f"(peor {rows[-1]['penalizacion_peor_fold']:8.2f})   "
              f"RMSE est. {rows[-1]['rmse_estimacion_mm']:.3f} mm   "
              f"|Δlog10C| {rows[-1]['error_log10C']:.3f}   [{rows[-1]['segundos']:.0f}s]")
    return pd.DataFrame(rows).sort_values("penalizacion_media").reset_index(drop=True)


def main() -> None:
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)
    labels = load_labels()
    batches = build_specimen_batches(labels, Config().m_exponent)
    batches = {k: v for k, v in batches.items() if k not in ("T7", "T8")}

    print(f"Búsqueda de hiperparámetros — LOSO sobre {', '.join(sorted(batches))}")
    table = search(batches)
    table.to_csv(RESULTS / "seleccion_capacidad.csv", index=False)
    print("\nMejores configuraciones:")
    print(table.head(10).round(3).to_string(index=False))

    best = table.iloc[0]
    chosen = {}
    for k in GRID:
        v = best[k]
        chosen[k] = int(v) if k in ("epochs", "patience") else (
            v if isinstance(v, str) else float(v))
    print(f"\nElegida: {chosen}")


if __name__ == "__main__":
    main()
