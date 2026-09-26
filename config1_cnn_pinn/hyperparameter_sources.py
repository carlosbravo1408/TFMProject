"""¿Sirven los hiperparámetros de literatura, o hacía falta identificarlos?

Este es el experimento que cierra el encargo original: construir la
Configuración 1 **sobre cada procedencia posible de los hiperparámetros de la
ecuación física** y comparar, en vez de dar por hecho que la identificación
heurística era necesaria.

Cuatro procedencias:

``identificado``
    m = 2,00 y log10 C = −8,425, identificados por metaheurísticos (evolución
    diferencial, tras compararla con recocido simulado, ACO_R, VNS y PSO) sobre
    las curvas de T1/T3/T4/T6 en ``physics_calibration/``.

``rao``
    La ventana genérica de metales que usa el 3.er puesto del certamen,
    C ∈ [1e-13, 1e-11] y m ∈ [2, 4] (tras Li, Wang & Gong, 2012). Es lo más
    cercano a un "baseline de literatura" que el propio certamen ofrece. Se toma
    el centro geométrico: m = 3, C = 1e-12.

``dourado``
    C = 5,008e-10, m = 3,859 para Al **2024-T3** en aire (Dourado & Viana, PHM
    Conf. 2019, adaptado de cupones de Menan & Henaff, 2010). Es el único par
    publicado para *esta* aleación en todo el corpus revisado. Las unidades no
    son recuperables del artículo, lo que ya es motivo de reserva.

``metals``
    C = 5,75e-8, m = 3,09 para AA 2024-**T4** en probeta CT con R = −1
    (*Metals* 13(8):1134, 2023). Temple y geometría distintos.

``faa``
    Ajuste de Walker para chapa de 2024-T3 desnuda y chapada, orientación L-T
    (Forman et al., 2005, DOT/FAA/AR-05/15, fig. 3, material M2EA11AB1):
    da/dN = C·[ΔK/(1−R)^(1−m)]^n con C = 0,167e-8 in/ciclo, n = 3,273 y
    m+ = 0,618 (ΔK en ksi·√in), ajustado con datos a R = 0; 0,5 y 0,7.
    **Unidades explícitas**: es la prueba limpia que el par de Dourado & Viana
    no permite. Se convierte a la forma de Paris del módulo (SI, Y = 1) a la
    razón de tensiones de los especímenes, R = 0,0476.

Todo lo demás — arquitectura, pérdida, protocolo, semillas — es idéntico entre
variantes, de modo que la única diferencia es de dónde salen m y C.

Ejecutar:  python -m config1_cnn_pinn.hyperparameter_sources
"""
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

# Capacidad y pesos de pérdida seleccionados sobre los folds de entrenamiento;
# se mantienen fijos para que la única variable sea la procedencia de m y C.
BASE = dict(ABLATION["Configuración 1 (1D-CNN + PINN)"])

def identify_log_c(m: float) -> float:
    """log10 C medio sobre los especímenes de ENTRENAMIENTO, al exponente ``m``.

    Se reidentifica para cada exponente porque C y m se desplazan juntos sobre
    la cresta log-log: un coeficiente citado a un exponente distinto del suyo es
    sencillamente incorrecto. Sólo entran T1/T3/T4/T6 — nunca T7 ni T8.
    """
    from physics_calibration.data import CALIBRATION_SPECIMENS, load_curve
    from physics_calibration.models import LAWS
    from physics_calibration.pooled import fit_coefficient_only

    valores = [
        fit_coefficient_only(LAWS["paris"], np.array([m]), load_curve(s))[0]
        for s in CALIBRATION_SPECIMENS
    ]
    return float(np.mean(valores))


R_ESPECIMENES = 0.0476          # T1–T7; T8 oscila entre 0,0476 y 0,0530
MPA_SQRT_M_POR_KSI_SQRT_IN = 6.894757 * math.sqrt(0.0254)


def walker_faa_a_paris_si(c_in: float, n: float, m_plus: float, r: float) -> tuple[float, float]:
    """Walker de la FAA (in/ciclo, ksi·√in) → Paris del módulo (m/ciclo, MPa·√m) a R fijo.

    da/dN = C·[ΔK/(1−R)^(1−m)]^n  ⇒  da/dN = C'·ΔK^n  con
    C' = 0,0254 · C · (6,894757·√0,0254)^(−n) · (1−R)^(−n·(1−m)).
    """
    c_si = 0.0254 * c_in * MPA_SQRT_M_POR_KSI_SQRT_IN ** (-n) * (1.0 - r) ** (-n * (1.0 - m_plus))
    return n, math.log10(c_si)


M_FAA, LOG_C_FAA = walker_faa_a_paris_si(0.167e-8, 3.273, 0.618, R_ESPECIMENES)

SOURCES = {
    # Dos exponentes, ambos elegidos SIN mirar T7/T8:
    #   2,25 = minimax de prognosis sobre los folds de entrenamiento;
    #   2,00 = queda a un 7 % de ese óptimo y es el único valor del entorno con
    #          solución cerrada exponencial: sin singularidad y monótona
    #          creciente de forma incondicional. Se comparan los dos en vez de
    #          escoger uno por conveniencia numérica a posteriori.
    "Identificado, m = 2,25 (minimax en entrenamiento)": {
        "m": 2.25,
        "log_c": None,      # se identifica sobre entrenamiento al ejecutar
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
    """Construye la Configuracion 1 sobre cada procedencia de *m* y *C*.

    Devuelve ``(tabla, detalle)``. No escribe ni lee ficheros: la version
    anterior mezclaba la tabla con el CSV de una ejecucion previa cuando se
    filtraba por subconjunto, lo que hacia que el resultado dependiera de que
    hubiera corrido antes. Ahora lo que se ejecuta es lo que se ve.
    """
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
    """Atajo de linea de comandos. El notebook llama a ``comparar_procedencias``."""
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
