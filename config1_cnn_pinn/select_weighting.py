"""Segunda etapa de selección: el peso por ciclo de la pérdida de datos.

Qué pregunta responde
---------------------
El diagnóstico LOSO dejó localizado el fallo de la mitad de *estimación*: no es
un sesgo global —la razón mediana estimado/real es 1,036— sino un **ancla mala**.
Al presupuesto de dos medidas que impone el protocolo, el error del ancla tiene
mediana 28 % y llega a −42 %, y como la extrapolación arranca de ahí, ese error
se paga en todos los ciclos ciegos y además por el lado caro de la asimetría.

La hipótesis es que la culpa la tiene el propio peso T(i) = 2 + 10·x copiado del
certamen dentro de la pérdida (ver ``losses.sample_weights``). Este módulo la
somete a prueba con el **protocolo completo del certamen sobre folds de
entrenamiento**, que es el mismo criterio con el que se eligió la capacidad de
la red y el único libre de fuga.

**T7 y T8 no se cargan.** La guarda es explícita y se comprueba en ``main``.

Se reportan dos columnas que no son el criterio pero hacen legible el resultado:
el error del ancla (la magnitud que la hipótesis dice que debe moverse) y el
RMSE de estimación (la mitad de la tarea que la penalización apenas mira: en la
rejilla anterior el 94 % de la penalización LOSO venía de la mitad extrapolada).

Ejecutar:  python -m config1_cnn_pinn.select_weighting
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from .data import build_specimen_batches, load_labels
from .evaluate import n_observed_nonzero
from .select_prognosis import robustez
from .train import (Config, LOSO_FOLDS, RESULTS, TRAIN_POOL, evaluate_ensemble,
                    train_ensemble)

# La capacidad se hereda de la primera etapa (``select.py``) y no se re-explora:
# la pregunta aquí es una sola variable, el peso de la pérdida.
# El exponente es el que fija ``select_prognosis`` por minimax sobre la banda de
# incertidumbre del coeficiente. Las dos etapas estan acopladas —el peso decide
# el ancla y el exponente decide como se propaga su error— asi que esta se
# ejecuta **con el exponente ya elegido**, no con el nominal de ``priors``.
BASE = dict(epochs=600, dropout=0.3, weight_decay=1e-3,
            lambda_physics=1.0, lambda_log_c_prior=1.0, data_loss="asym",
            m_exponent=2.00)

VARIANTES = [
    ("challenge (T(i) del certamen)", dict(data_weight="challenge")),
    ("plano",                          dict(data_weight="flat")),
    ("ancla x3",                       dict(data_weight="anchor", anchor_weight=3.0)),
    ("ancla x6",                       dict(data_weight="anchor", anchor_weight=6.0)),
    ("ancla x12",                      dict(data_weight="anchor", anchor_weight=12.0)),
    ("ancla x24",                      dict(data_weight="anchor", anchor_weight=24.0)),
    ("ancla x48",                      dict(data_weight="anchor", anchor_weight=48.0)),
    ("ancla x100",                     dict(data_weight="anchor", anchor_weight=100.0)),
]

# Un unico criterio para todo el proceso de seleccion, el mismo que usa
# ``select_prognosis``: peor fold de entrenamiento **sobre toda la banda de
# incertidumbre del coeficiente** (+-0,3 dex, la sigma del prior de la cabeza
# fisica). Puntuar en el coeficiente nominal supone que el coeficiente es
# correcto, que es justo lo que falla en un espécimen de un régimen no visto.
CRITERIO = "peor_fold_banda"


def error_ancla(cfg: Config, batches, folds=LOSO_FOLDS, n_seeds: int = 3) -> float:
    """|error| mediano del ancla sobre los folds, en tanto por uno.

    El ancla es la ``n_observed``-ésima medida no nula, que es lo que el
    protocolo deja ver antes de cortar la señal.
    """
    n_obs = n_observed_nonzero("T7")   # el presupuesto, no datos de T7
    errores = []
    for fuera in folds:
        nombres = [n for n in TRAIN_POOL if n != fuera and n in batches]
        modelos = train_ensemble(cfg, batches, nombres, n_seeds=n_seeds)
        _, est, real, _ = evaluate_ensemble(modelos, batches[fuera], cfg)
        nz = real > 0
        errores.append(abs(est[nz][n_obs - 1] / real[nz][n_obs - 1] - 1.0))
    return float(np.median(errores))


def rmse_estimacion(cfg: Config, batches, folds=LOSO_FOLDS, n_seeds: int = 3) -> float:
    total = []
    for fuera in folds:
        nombres = [n for n in TRAIN_POOL if n != fuera and n in batches]
        modelos = train_ensemble(cfg, batches, nombres, n_seeds=n_seeds)
        _, est, real, _ = evaluate_ensemble(modelos, batches[fuera], cfg)
        nz = real > 0
        total.append(np.sqrt(np.mean((est[nz] - real[nz]) ** 2)))
    return float(np.mean(total))


def buscar_peso(n_seeds: int = 3, etiquetas=None, verbose: bool = True):
    """Compara las variantes de peso y devuelve ``(tabla, elegido)``.

    No escribe nada ni lee nada: quien la llama decide si guardar. ``elegido``
    son los ajustes de ``Config`` de la variante que gana bajo ``CRITERIO``.
    """
    etiquetas = load_labels() if etiquetas is None else etiquetas
    m = BASE["m_exponent"]
    batches = {k: v for k, v in build_specimen_batches(etiquetas, m).items()
               if k not in ("T7", "T8")}
    assert "T7" not in batches and "T8" not in batches, "fuga: T7/T8 en la seleccion"

    if verbose:
        print(f"Seleccion del peso de la perdida — LOSO sobre {', '.join(sorted(batches))}")
        print(f"(m = {m}; {n_seeds} semillas; criterio: {CRITERIO})\n")

    filas = []
    for etiqueta, extra in VARIANTES:
        cfg = Config(**{**BASE, **extra})
        t0 = time.perf_counter()
        cache = {}
        for fuera in LOSO_FOLDS:
            nombres = [n for n in TRAIN_POOL if n != fuera and n in batches]
            modelos = train_ensemble(cfg, batches, nombres, n_seeds=n_seeds)
            cache[fuera] = evaluate_ensemble(modelos, batches[fuera], cfg)

        rob = robustez(cache, m, "cabeza")
        nominal = rob[rob["delta_log10C"] == 0.0].iloc[0]
        n_obs = n_observed_nonzero("T7")
        anclas, rmses = [], []
        for fuera in LOSO_FOLDS:
            _, est, real, _ = cache[fuera]
            nz = real > 0
            anclas.append(abs(est[nz][n_obs - 1] / real[nz][n_obs - 1] - 1.0))
            rmses.append(np.sqrt(np.mean((est[nz] - real[nz]) ** 2)))

        filas.append({
            "peso": etiqueta,
            "peor_fold_banda": float(rob["peor_fold"].max()),
            "puntos_en_tope": int(rob["puntos_en_tope"].sum()),
            "peor_fold_nominal": float(nominal["peor_fold"]),
            "media_nominal": float(nominal["media"]),
            "error_ancla": float(np.median(anclas)),
            "rmse_estimacion_mm": float(np.mean(rmses)),
            "segundos": time.perf_counter() - t0,
        })
        if verbose:
            f = filas[-1]
            print(f"  {etiqueta:32s} peor-banda {f['peor_fold_banda']:8.2f} "
                  f"(nominal {f['peor_fold_nominal']:7.2f})  tope {f['puntos_en_tope']:2d}  "
                  f"ancla {100 * f['error_ancla']:5.1f} %  "
                  f"RMSE est {f['rmse_estimacion_mm']:.3f} mm  [{f['segundos']:.0f}s]", flush=True)

    tabla = pd.DataFrame(filas).sort_values(CRITERIO).reset_index(drop=True)
    elegido = dict(next(e for et, e in VARIANTES if et == tabla.iloc[0]["peso"]))
    return tabla, elegido


def main(n_seeds: int = 3) -> None:
    """Atajo de linea de comandos. El notebook llama a ``buscar_peso``."""
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)
    tabla, elegido = buscar_peso(n_seeds=n_seeds)
    print("\n" + tabla.round(3).to_string(index=False))
    tabla.to_csv(RESULTS / "seleccion_peso_perdida.csv", index=False)
    print(f"\nElegido: {tabla.iloc[0]['peso']}  ->  {elegido}")


if __name__ == "__main__":
    main()
