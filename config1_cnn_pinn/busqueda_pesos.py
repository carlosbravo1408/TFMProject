"""Búsqueda de los pesos de la pérdida por evolución diferencial.

Por qué no basta una rejilla
----------------------------
Los pesos de la pérdida se habían fijado con una rejilla gruesa de tres
valores, 0, 1 y 3, elegidos a ojo. Esa rejilla falló dos veces seguidas: el
óptimo del peso físico resultó estar en 20, casi siete veces por encima de su
extremo superior, y hubo que extenderla dos veces a mano para encontrarlo.

Hay tres razones por las que una rejilla no sirve aquí, y las tres están
medidas:

* **Los pesos están acoplados.** El término que retiene al coeficiente ayuda
  cuando el residuo físico actúa con peso moderado y estorba cuando actúa con
  peso fuerte, de modo que explorarlos por separado no puede ver el óptimo
  conjunto.
* **La superficie no es suave.** En el barrido del peso físico, 10 puntúa peor
  que 6 y 320 mejor que 160.
* **El recorrido útil abarca órdenes de magnitud.** De 0,1 a 300 en el peso
  físico, de 3 a 200 en el del ancla. Una rejilla que cubra eso con resolución
  suficiente es más cara que la búsqueda.

Se emplea la evolución diferencial del módulo ``physics_calibration``, que es
la que ganó la comparativa de seis algoritmos de aquella etapa, de modo que la
elección del optimizador no introduce un grado de libertad nuevo.

Criterio y protección contra el ruido
-------------------------------------
El objetivo es el mismo criterio libre de fuga que el resto del trabajo: el
peor *fold* de entrenamiento bajo el protocolo completo del certamen, evaluado
sobre toda la banda de incertidumbre del coeficiente, ±0,3 dex, y con la cota
de entrega recalculada en cada *fold* excluyendo el espécimen puntuado. **T7 y
T8 no se cargan en ningún momento.**

Cada candidato se evalúa con varias semillas, porque la dispersión entre
semillas de una misma configuración es del orden de la diferencia entre
configuraciones distintas. Sin eso, la búsqueda optimizaría el ruido.

El óptimo que encuentre **debe verificarse** con tandas de semillas
independientes antes de adoptarse: ``verificar`` hace esa comprobación.

Ejecutar:  python -m config1_cnn_pinn.busqueda_pesos
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from physics_calibration.metaheuristics import differential_evolution
from .cota_extrapolacion import cota_de_regla, entrenar_folds, maximos_por_especimen
from .data import build_specimen_batches, load_labels
from .evaluate import ABLATION, build_submission
from .select_prognosis import robustez
from .train import Config, LOSO_FOLDS, RESULTS, TRAIN_POOL, train_ensemble

# Variables de la búsqueda, en log10. El recorrido de cada una cubre los
# órdenes de magnitud en los que el barrido manual mostró actividad.
VARIABLES = [
    ("lambda_physics",     -1.0, 2.5),    # 0,1  a 316
    ("lambda_log_c_prior", -2.0, 1.0),    # 0,01 a 10
    ("lambda_monotonic",   -2.0, 1.0),    # 0,01 a 10
    ("anchor_weight",       0.5, 2.3),    # 3,2  a 200
]
LIMITES = [(b, c) for _, b, c in VARIABLES]
REGLA_COTA = "segunda mayor"


def descodificar(x) -> dict:
    """Vector en log10 -> ajustes de ``Config``."""
    return {nombre: float(10.0 ** v) for (nombre, _, _), v in zip(VARIABLES, x)}


def _base() -> dict:
    """Configuración de partida: todo salvo los pesos que se buscan."""
    base = dict(ABLATION["Configuración 1 (1D-CNN + PINN)"])
    for nombre, _, _ in VARIABLES:
        base.pop(nombre, None)
    base["data_weight"] = "anchor"
    return base


def construir_objetivo(n_seeds: int = 2, etiquetas=None, registro=None):
    """Objetivo: peor fold sobre la banda, con la cota excluyendo cada fold."""
    etiquetas = load_labels() if etiquetas is None else etiquetas
    maximos = maximos_por_especimen([n for n in TRAIN_POOL])
    cotas = {f: cota_de_regla(REGLA_COTA, maximos, excluir=f) for f in LOSO_FOLDS}
    base = _base()
    historial = [] if registro is None else registro

    def f(X):
        X = np.atleast_2d(X)
        salida = []
        for fila in X:
            pesos = descodificar(fila)
            cfg = Config(**{**base, **pesos})
            t0 = time.perf_counter()
            try:
                cache = entrenar_folds(cfg, n_seeds=n_seeds, etiquetas=etiquetas,
                                       verbose=False)
                r = robustez(cache, cfg.m_exponent, "cabeza", cotas_por_fold=cotas)
                valor = float(r["peor_fold"].max())
                media = float(r["media"].mean())
            except Exception as exc:                      # una configuración puede divergir
                valor, media = float("inf"), float("inf")
                print(f"    descartada: {exc}", flush=True)
            historial.append({**pesos, "peor_fold_banda": valor,
                              "media_banda": media,
                              "segundos": time.perf_counter() - t0})
            if len(historial) % 5 == 0:
                pd.DataFrame(historial).to_csv(
                    RESULTS / "busqueda_pesos_historial.csv", index=False)
            print(f"  [{len(historial):3d}] " + "  ".join(
                f"{k}={v:8.3f}" for k, v in pesos.items()) +
                f"   -> {valor:9.2f}   [{historial[-1]['segundos']:.0f}s]", flush=True)
            salida.append(valor)
        return np.array(salida)

    return f, historial


def verificar(pesos: dict, n_tandas: int = 3, n_seeds: int = 5, etiquetas=None):
    """Repite el óptimo sobre tandas independientes y lo entrega en T7/T8.

    La búsqueda puede haber encontrado una combinación afortunada para las
    semillas con las que se evaluó. Esta comprobación la repite con semillas
    distintas; sólo si la mejora sobrevive tiene sentido adoptarla.
    """
    etiquetas = load_labels() if etiquetas is None else etiquetas
    filas = []
    for etiqueta, extra in (("óptimo de la búsqueda", pesos),
                            ("configuración vigente", {})):
        for tanda in range(n_tandas):
            cfg = Config(**{**ABLATION["Configuración 1 (1D-CNN + PINN)"],
                            **extra, "seed": 1000 * tanda})
            lotes = build_specimen_batches(etiquetas, cfg.m_exponent)
            modelos = train_ensemble(cfg, lotes, [n for n in TRAIN_POOL if n in lotes],
                                     n_seeds=n_seeds)
            subs = [build_submission(modelos, lotes, cfg, n) for n in ("T7", "T8")]
            filas.append({"variante": etiqueta, "tanda": tanda,
                          "T7": subs[0]["penalizacion"], "T8": subs[1]["penalizacion"],
                          "total": subs[0]["penalizacion"] + subs[1]["penalizacion"]})
            print(f"  {etiqueta:24s} tanda {tanda}  T7 {filas[-1]['T7']:7.2f}  "
                  f"T8 {filas[-1]['T8']:8.2f}", flush=True)
    return pd.DataFrame(filas)


def main(max_evals: int = 160, pop_size: int = 16, n_seeds: int = 2, seed: int = 0):
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)
    print(f"Evolución diferencial sobre {len(VARIABLES)} pesos, "
          f"{max_evals} evaluaciones de {n_seeds} semillas cada una.")
    print("Criterio: peor fold de entrenamiento sobre la banda de ±0,3 dex. "
          "T7 y T8 no se cargan.\n")

    f, historial = construir_objetivo(n_seeds=n_seeds)
    t0 = time.perf_counter()
    res = differential_evolution(f, LIMITES, seed=seed, max_evals=max_evals,
                                 pop_size=pop_size)
    pd.DataFrame(historial).to_csv(RESULTS / "busqueda_pesos_historial.csv", index=False)

    mejor = descodificar(res.x)
    print("\n" + "=" * 90)
    print(f"ÓPTIMO tras {res.n_evals} evaluaciones en {(time.perf_counter()-t0)/3600:.1f} h")
    print("=" * 90)
    for k, v in mejor.items():
        print(f"  {k:20s} {v:10.4f}")
    print(f"  criterio (peor fold sobre la banda): {res.fun:.2f}")
    (RESULTS / "busqueda_pesos_optimo.json").write_text(
        json.dumps({"pesos": mejor, "criterio": res.fun,
                    "n_evaluaciones": res.n_evals}, indent=2))

    print("\nVerificación sobre tandas independientes:")
    tabla = verificar(mejor)
    tabla.to_csv(RESULTS / "busqueda_pesos_verificacion.csv", index=False)
    print()
    print(tabla.groupby("variante").agg(
        T7_med=("T7", "median"), T7_min=("T7", "min"), T7_max=("T7", "max"),
        T8_med=("T8", "median"), total_med=("total", "median")).round(2).to_string())


if __name__ == "__main__":
    main()
