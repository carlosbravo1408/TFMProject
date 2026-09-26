"""¿Hasta dónde puede crecer una grieta extrapolada? Calibración de la cota.

Qué pregunta responde
---------------------
La extrapolación de Paris es monótona creciente y sin cota superior, de modo
que un coeficiente algo alto no produce un error algo mayor, sino una grieta
arbitrariamente larga. El módulo de física ya recorta esa trayectoria en 20 mm,
pero ese valor **no se eligió para la entrega**: se eligió para que la pérdida
no desborde durante el entrenamiento, con el criterio explícito de ser tan
holgado que nunca tocase una curva razonable. Para lo que sí es un tope de
entrega, el criterio correcto es el contrario: el mayor valor que el fenómeno
produce de verdad.

Este módulo separa los dos usos y calibra el segundo sobre los especímenes de
entrenamiento, con el mismo criterio libre de fuga que el resto del trabajo.

La corrección de circularidad
-----------------------------
La regla natural, «la cota es la grieta más larga observada en entrenamiento»,
vale 7.46 mm y procede de T1. Evaluarla sobre el *fold* T1 sería circular: la
cota conocería la respuesta del espécimen que se está puntuando. Por eso, en
cada *fold*, la cota se **recalcula excluyendo ese fold**, igual que se excluyen
sus ondas del entrenamiento. Para T1 la cota honesta es 7.24 mm, la de T4.

Al aplicarla a T7 y T8 la exclusión no hace falta: ninguno de los dos es un
espécimen de entrenamiento, así que la cota se calcula con T1 a T6 completos.

Dos barridos complementarios
----------------------------
``absoluto``   recorre valores fijos de 4 a 20 mm. Describe la forma de la
               curva y dónde está el óptimo, pero **no es adoptable**: elegir
               el mínimo de esa curva sería fijar un número mirando el
               resultado.
``por regla``  evalúa reglas que derivan la cota de los datos disponibles en
               cada fold. Es la única forma adoptable, porque la misma regla
               puede aplicarse a T7 y T8 sin haberlos mirado.

Todo se evalúa sobre la banda de incertidumbre del coeficiente, ±0.3 dex, que
es el criterio con el que se eligieron el exponente y el peso de la pérdida.

Ejecutar:  python -m config1_cnn_pinn.cota_extrapolacion
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from physics_calibration.data import load_curve
from .data import build_specimen_batches, load_labels
from .evaluate import ABLATION
from .select_prognosis import CRITERIO, robustez
from .train import (Config, LOSO_FOLDS, RESULTS, TRAIN_POOL, evaluate_ensemble,
                    train_ensemble)

COTA_NUMERICA = 20.0

ABSOLUTAS = (4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0)

REGLAS = {
    "sin cota (20 mm, actual)": lambda maximos: COTA_NUMERICA,
    "máxima observada": lambda maximos: max(maximos),
    "máxima observada +10 %": lambda maximos: max(maximos) * 1.10,
    "máxima observada +25 %": lambda maximos: max(maximos) * 1.25,
    "media de las máximas": lambda maximos: float(np.mean(maximos)),
    "segunda mayor": lambda maximos: sorted(maximos)[-2],
}


def maximos_por_especimen(nombres) -> dict:
    """Grieta final medida en cada espécimen, en mm."""
    return {n: float(load_curve(n).crack_mm.max()) for n in nombres}


def cota_de_regla(regla, maximos: dict, excluir=None) -> float:
    """Aplica ``regla`` a los máximos disponibles, excluyendo un espécimen.

    ``excluir`` es lo que corrige la circularidad: al puntuar el fold T1, la
    cota no puede haber visto la grieta final de T1.
    """
    valores = [v for n, v in maximos.items() if n != excluir]
    return float(REGLAS[regla](valores))


def entrenar_folds(cfg: Config, n_seeds: int = 3, etiquetas=None, verbose: bool = True):
    """Estimaciones por fold. Se entrena una sola vez y luego sólo cambia la cota."""
    etiquetas = load_labels() if etiquetas is None else etiquetas
    lotes = {k: v for k, v in build_specimen_batches(etiquetas, cfg.m_exponent).items()
             if k not in ("T7", "T8")}
    assert "T7" not in lotes and "T8" not in lotes, "fuga: T7/T8 en la calibración"
    cache = {}
    for fuera in LOSO_FOLDS:
        nombres = [n for n in TRAIN_POOL if n != fuera and n in lotes]
        cache[fuera] = evaluate_ensemble(
            train_ensemble(cfg, lotes, nombres, n_seeds=n_seeds), lotes[fuera], cfg)
        if verbose:
            print(f"  fold {fuera}: estimador entrenado", flush=True)
    return cache


def barrido_absoluto(cache, m: float, cotas=ABSOLUTAS) -> pd.DataFrame:
    """Cada cota fija, idéntica en todos los folds."""
    filas = []
    for cota in cotas:
        r = robustez(cache, m, "cabeza", cotas_por_fold={f: cota for f in LOSO_FOLDS})
        nominal = r[r["delta_log10C"] == 0.0].iloc[0]
        filas.append({
            "cota (mm)": cota,
            "peor_fold_banda": float(r["peor_fold"].max()),
            "media_banda": float(r["media"].mean()),
            "peor_fold_nominal": float(nominal["peor_fold"]),
            "media_nominal": float(nominal["media"]),
            "puntos_en_la_cota": int(r["puntos_en_tope"].sum()),
        })
    return pd.DataFrame(filas)


def barrido_por_regla(cache, m: float, maximos: dict) -> pd.DataFrame:
    """Cada regla, con la cota recalculada en cada fold excluyendo ese fold."""
    filas = []
    for regla in REGLAS:
        cotas = {f: cota_de_regla(regla, maximos, excluir=f) for f in LOSO_FOLDS}
        r = robustez(cache, m, "cabeza", cotas_por_fold=cotas)
        nominal = r[r["delta_log10C"] == 0.0].iloc[0]
        filas.append({
            "regla": regla,
            "cota por fold (mm)": ", ".join(f"{f}:{c:.2f}" for f, c in cotas.items()),
            "cota para T7/T8 (mm)": round(cota_de_regla(regla, maximos), 2),
            "peor_fold_banda": float(r["peor_fold"].max()),
            "media_banda": float(r["media"].mean()),
            "peor_fold_nominal": float(nominal["peor_fold"]),
            "media_nominal": float(nominal["media"]),
        })
    return pd.DataFrame(filas).sort_values(CRITERIO).reset_index(drop=True)


def main(n_seeds: int = 3) -> None:
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)

    cfg = Config(**ABLATION["Configuración 1 (1D-CNN + PINN)"])
    maximos = maximos_por_especimen([n for n in TRAIN_POOL])
    print("Grieta final medida en cada espécimen de entrenamiento (mm):")
    print("  " + ", ".join(f"{n} {v:.2f}" for n, v in maximos.items()))
    print(f"\nEntrenando los {len(LOSO_FOLDS)} folds una sola vez "
          f"({n_seeds} semillas cada uno)…")
    t0 = time.perf_counter()
    cache = entrenar_folds(cfg, n_seeds=n_seeds)
    print(f"[{time.perf_counter() - t0:.0f} s]\n")

    absoluto = barrido_absoluto(cache, cfg.m_exponent)
    print("=" * 96)
    print("BARRIDO DE VALORES ABSOLUTOS (diagnóstico: describe la curva, no es adoptable)")
    print("=" * 96)
    print(absoluto.round(2).to_string(index=False))

    por_regla = barrido_por_regla(cache, cfg.m_exponent, maximos)
    print("\n" + "=" * 96)
    print(f"REGLAS CON EXCLUSIÓN POR FOLD (adoptable; criterio: {CRITERIO})")
    print("=" * 96)
    print(por_regla.round(2).to_string(index=False))

    absoluto.to_csv(RESULTS / "cota_barrido_absoluto.csv", index=False)
    por_regla.to_csv(RESULTS / "cota_por_regla.csv", index=False)
    elegida = por_regla.iloc[0]
    print(f"\nRegla con menor {CRITERIO}: {elegida['regla']}  "
          f"-> cota para T7/T8 = {elegida['cota para T7/T8 (mm)']} mm")


if __name__ == "__main__":
    main()
