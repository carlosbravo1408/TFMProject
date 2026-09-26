"""Model selection for Configuration 1, on the training specimens only.

Every number produced here comes from leave-one-specimen-out over T1/T3/T4/T6.
**T7 and T8 are never loaded**, so the configuration finally reported on them is
selected without having seen them — unlike the challenge entries themselves,
whose authors could iterate against the leaderboard.

The grid is small and deliberately aimed at capacity control rather than
architecture search. At 87 labelled waveforms with a strong domain shift
between specimens (each uses a different actuator-sensor path), the binding
constraint is over-fitting, not expressiveness: a first run at 1500 epochs
scored *worse* on held-out specimens than the same model at 600, which is the
signature of a model that has started memorising its five training specimens.

Run:  python -m config1_cnn_pinn.select
"""
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

# The first two axes are capacity control; the last two are the ablation the
# TFM actually needs. ``lambda_physics = 0`` is the pure 1D-CNN control that
# isolates what component (b) contributes, and ``data_loss`` answers Research
# Gap 1 of the state of the art: does making the *training* loss asymmetric
# beat correcting for the asymmetry afterwards?
GRID = {
    # El exponente **no** se barre aqui: se fija al valor que la tercera etapa
    # adopta. Dejarlo al valor por defecto significaba seleccionar la capacidad
    # bajo m = 2,25, el recomendado por la calibracion, y entrenar despues bajo
    # m = 2,00, con lo que los pesos del termino fisico se elegian sobre una
    # rama de la solucion cerrada distinta de la que luego se usa.
    "m_exponent": (2.00,),
    "epochs": (100, 600),
    "dropout": (0.3,),
    "weight_decay": (1e-3,),
    "lambda_physics": (0.0, 1.0, 3.0),
    "lambda_log_c_prior": (0.0, 1.0),
    "data_loss": ("asym",),
}


def search(batches, grid=GRID, n_seeds: int = 1) -> pd.DataFrame:
    """Grid search scored by the *full challenge protocol* on held-out folds.

    Estimation RMSE is reported alongside but is not the criterion: it cannot
    see the extrapolated half of the submission, which is precisely what the
    PINN head is for (see ``evaluate.loso_challenge_score``).
    """
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
    # Guard: model selection must not see the validation specimens.
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
