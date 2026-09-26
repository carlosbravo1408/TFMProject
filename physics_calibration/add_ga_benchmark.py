"""Añade el algoritmo genético al banco de comparación ya registrado.

El GA se incorporó después de la primera ejecución de ``calibrate.py``, al
detectar que era el algoritmo que ambos papers del certamen usaron realmente
para ajustar los parámetros de fractura. Reejecutar las 900 optimizaciones de
los otros cinco algoritmos no aportaría nada: se ejecutaron con las mismas
semillas, el mismo presupuesto y la misma rejilla, de modo que basta con
correr las del GA y anexarlas.

Ejecutar:  python -m physics_calibration.add_ga_benchmark
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .calibrate import BENCHMARK_LAWS, FITTABLE, RESULTS
from .data import ALL_SPECIMENS, CALIBRATION_SPECIMENS, load_curves
from .metaheuristics import ALGORITHMS
from .models import LAWS
from .objectives import OBJECTIVES, evaluate_all
from .pooled import ProfiledPooledProblem


def stage_a_ga(curves, seeds, max_evals, laws=BENCHMARK_LAWS) -> pd.DataFrame:
    rows = []
    for law_name in laws:
        law = LAWS[law_name]
        for spec in FITTABLE:
            curve = curves[spec]
            f = OBJECTIVES["rmse"](law, curve)
            for seed in seeds:
                t0 = time.perf_counter()
                res = ALGORITHMS["ga"](f, law.bounds, seed=seed, max_evals=max_evals)
                dt = time.perf_counter() - t0
                metrics = evaluate_all(law, res.x, curve)
                rows.append({
                    "law": law_name, "specimen": spec, "algorithm": "ga",
                    "seed": seed, "objective_rmse_mm": res.fun,
                    "phm_penalty": metrics["phm_penalty"],
                    "max_abs_err_mm": metrics["max_abs_err_mm"],
                    "n_evals": res.n_evals, "seconds": dt,
                    **{f"x{i}": v for i, v in enumerate(res.x)},
                })
    return pd.DataFrame(rows)


def stage_b_ga(curves, seeds, max_evals, laws) -> pd.DataFrame:
    train = tuple(curves[s] for s in CALIBRATION_SPECIMENS)
    rows = []
    for law_name in laws:
        problem = ProfiledPooledProblem(LAWS[law_name], train, loss="rmse")
        f = problem.objective()
        for seed in seeds:
            t0 = time.perf_counter()
            res = ALGORITHMS["ga"](f, problem.bounds, seed=seed, max_evals=max_evals)
            rows.append({
                "law": law_name, "algorithm": "ga", "seed": seed,
                "objective_mean_rmse_mm": res.fun,
                "seconds": time.perf_counter() - t0,
            })
    return pd.DataFrame(rows)


def main() -> None:
    seeds = list(range(10))
    curves = load_curves(ALL_SPECIMENS)

    print("== Etapa A: algoritmo genético (mismas semillas y presupuesto) ==")
    path_a = RESULTS / "stage_a_algorithm_benchmark.csv"
    previo = pd.read_csv(path_a)
    previo = previo[previo["algorithm"] != "ga"]
    nuevo = stage_a_ga(curves, seeds, max_evals=10000)
    completo = pd.concat([previo, nuevo], ignore_index=True)
    completo.to_csv(path_a, index=False)

    completo["gap_mm"] = completo["objective_rmse_mm"] - completo.groupby(
        ["law", "specimen"])["objective_rmse_mm"].transform("min")
    tabla = (completo.groupby("algorithm")
             .agg(óptimos=("gap_mm", lambda v: int((v < 1e-6).sum())),
                  ejecuciones=("gap_mm", "size"),
                  brecha_mediana=("gap_mm", "median"),
                  brecha_p90=("gap_mm", lambda v: v.quantile(0.9)),
                  peor_brecha=("gap_mm", "max"),
                  seg=("seconds", "mean"))
             .sort_values("óptimos", ascending=False).round(5))
    print(tabla.to_string())

    print("\n== Etapa B: agrupada con proyección de variables ==")
    path_b = RESULTS / "stage_b_pooled_runs.csv"
    previo_b = pd.read_csv(path_b)
    laws = sorted(previo_b["law"].unique())
    previo_b = previo_b[previo_b["algorithm"] != "ga"]
    nuevo_b = stage_b_ga(curves, seeds, max_evals=800, laws=laws)
    completo_b = pd.concat([previo_b, nuevo_b], ignore_index=True)
    completo_b.to_csv(path_b, index=False)
    print(completo_b.groupby("algorithm")
          .agg(mediana=("objective_mean_rmse_mm", "median"),
               peor=("objective_mean_rmse_mm", "max"),
               mejor=("objective_mean_rmse_mm", "min"),
               seg=("seconds", "mean")).round(4).to_string())

    tabla.to_csv(RESULTS / "comparativa_algoritmos.csv")
    print(f"\n-> {path_a}\n-> {path_b}\n-> {RESULTS / 'comparativa_algoritmos.csv'}")


if __name__ == "__main__":
    main()
