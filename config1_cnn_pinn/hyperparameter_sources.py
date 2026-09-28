from __future__ import annotations

import json
import math
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from .data import build_specimen_batches, load_labels
from .evaluate import ABLATION, REFERENCES, build_submission
from .train import Config, RESULTS, TRAIN_POOL, train_ensemble

# Fixed, so the only variable is where m and C come from.
BASE = dict(ABLATION["Configuración 1 (1D-CNN + PINN)"])

def identify_log_c(m: float) -> float:
    from physics_calibration.data import CALIBRATION_SPECIMENS, load_curve
    from physics_calibration.models import LAWS
    from physics_calibration.pooled import fit_coefficient_only

    valores = [
        fit_coefficient_only(LAWS["paris"], np.array([m]), load_curve(s))[0]
        for s in CALIBRATION_SPECIMENS
    ]
    return float(np.mean(valores))


# T1-T7; T8 ranges from 0.0476 to 0.0530.
R_ESPECIMENES = 0.0476
MPA_SQRT_M_POR_KSI_SQRT_IN = 6.894757 * math.sqrt(0.0254)


def walker_faa_a_paris_si(c_in: float, n: float, m_plus: float, r: float) -> tuple[float, float]:
    # FAA Walker da/dN = C [dK / (1-R)^(1-m)]^n at fixed R becomes C' dK^n in SI.
    c_si = 0.0254 * c_in * MPA_SQRT_M_POR_KSI_SQRT_IN ** (-n) * (1.0 - r) ** (-n * (1.0 - m_plus))
    return n, math.log10(c_si)


M_FAA, LOG_C_FAA = walker_faa_a_paris_si(0.167e-8, 3.273, 0.618, R_ESPECIMENES)

SOURCES = {
    "Identificado, m = 2,25 (minimax en entrenamiento)": {
        "m": 2.25,
        "log_c": None,
        "procedencia": "Este trabajo, physics_calibration/",
    },
    "Identificado, m = 2,00 (forma cerrada no singular)": {
        "m": 2.00,
        "log_c": None,
        "procedencia": "Este trabajo, criterio estructural",
    },
    "Literatura — Rao et al. (ventana metálica genérica)": {
        "m": 3.0,
        "log_c": -12.0,
        "procedencia": "3.er puesto PHM 2019, tras Li, Wang & Gong (2012)",
    },
    "Literatura — Dourado & Viana (Al 2024-T3, aire)": {
        "m": 3.859,
        "log_c": math.log10(5.008e-10),
        "procedencia": "PHM Conf. 2019, cupones de Menan & Henaff (2010)",
    },
    "Literatura — FAA FCGD, Walker 2024-T3 chapa (Forman et al., 2005)": {
        "m": M_FAA,
        "log_c": LOG_C_FAA,
        "procedencia": "DOT/FAA/AR-05/15, fig. 3 (M2EA11AB1), convertido a SI con R = 0,0476",
    },
    "Literatura — Metals 2023 (AA 2024-T4, CT, R=-1)": {
        "m": 3.09,
        "log_c": math.log10(5.75e-8),
        "procedencia": "Metals 13(8):1134 (2023)",
    },
}


def run_source(name: str, spec: dict, n_seeds: int = 5, verbose: bool = True):
    log_c = spec["log_c"] if spec["log_c"] is not None else identify_log_c(spec["m"])
    cfg = Config(**{**BASE, "m_exponent": spec["m"], "log_c_prior": log_c})
    labels = load_labels()
    batches = build_specimen_batches(labels, cfg.m_exponent)
    train_names = [n for n in TRAIN_POOL if n in batches]
    if verbose:
        print(f"\n=== {name} ===")
        print(f"    m = {cfg.m_exponent:.3f}   log10 C = {cfg.log_c_prior:.3f} "
              f"(C = {10 ** cfg.log_c_prior:.3e})   [{spec['procedencia']}]")
    models = train_ensemble(cfg, batches, train_names, n_seeds=n_seeds)
    subs = [build_submission(models, batches, cfg, s) for s in ("T7", "T8")]
    return cfg, subs


def comparar_procedencias(fuentes=None, n_seeds: int = 5, verbose: bool = True):
    fuentes = SOURCES if fuentes is None else fuentes
    rows, detalle = [], {}
    for name, spec in fuentes.items():
        cfg, subs = run_source(name, spec, n_seeds=n_seeds, verbose=verbose)
        t7, t8 = subs
        total = t7["penalizacion"] + t8["penalizacion"]
        if verbose:
            print(f"    T7 = {t7['penalizacion']:8.2f} (RMSE {t7['rmse_mm']:.3f} mm)   "
                  f"T8 = {t8['penalizacion']:10.2f} (RMSE {t8['rmse_mm']:.3f} mm)   "
                  f"total = {total:10.2f}")
        rows.append({
            "procedencia de m y C": name,
            "m": round(cfg.m_exponent, 3),
            "log10 C": round(cfg.log_c_prior, 3),
            "T7": round(t7["penalizacion"], 2),
            "T8": round(t8["penalizacion"], 2),
            "total": round(total, 2),
            "RMSE T7 (mm)": round(t7["rmse_mm"], 3),
            "RMSE T8 (mm)": round(t8["rmse_mm"], 3),
        })
        detalle[name] = {"config": asdict(cfg), "submissions": subs}
    return pd.DataFrame(rows), detalle


def main() -> None:
    import sys

    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)

    filtros = [a.lower() for a in sys.argv[1:]]
    seleccion = {n: v for n, v in SOURCES.items()
                 if not filtros or any(f in n.lower() for f in filtros)}
    if filtros:
        print(f"Subconjunto: {list(seleccion)}")

    tabla, detalle = comparar_procedencias(seleccion)

    print("\n" + "=" * 110)
    print("CONFIGURACIÓN 1 CONSTRUIDA SOBRE CADA PROCEDENCIA DE LOS HIPERPARÁMETROS FÍSICOS")
    print("=" * 110)
    print(tabla.to_string(index=False))

    print("\nReferencias (penalización oficial por espécimen):")
    for label, v in REFERENCES.items():
        t7 = f"{v['T7']:9.2f}" if v["T7"] is not None else f"{'n/d':>9s}"
        t8 = f"{v['T8']:10.2f}" if v["T8"] is not None else f"{'n/d':>10s}"
        print(f"  {label:36s} T7 {t7}   T8 {t8}   total {v['total']:9.2f}")

    sufijo = "_subconjunto" if filtros else ""
    tabla.to_csv(RESULTS / f"procedencia_hiperparametros{sufijo}.csv", index=False)
    (RESULTS / f"procedencia_hiperparametros{sufijo}.json").write_text(
        json.dumps(detalle, indent=2, default=str))


if __name__ == "__main__":
    main()
