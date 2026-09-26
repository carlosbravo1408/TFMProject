"""Cuánto de lo que mide la ablación es señal y cuánto es la semilla.

Con 87 ondas etiquetadas, un ensamblado de cinco semillas no es una constante:
cambia de una semilla a otra. Dos preguntas distintas comparten aquí el mismo
procedimiento, repetir sobre tandas de semillas independientes y mirar el
recorrido resultante.

:func:`dispersion_ablacion`
    Repite **cada variante de la ablación** sobre varias tandas. Es lo que
    permite decidir si dos filas de la tabla de ablación están realmente
    ordenadas o sólo lo parecen.

:func:`sensibilidad_pesos`
    Repite la Configuración 1 variando los dos pesos de la pérdida que **no
    entran en ninguna rejilla de selección**: la bisagra de monotonía y la
    fracción de calentamiento del término físico. Ambos son valores por
    defecto, y la pregunta es si mueven el resultado por encima del ruido de
    semilla.

Las dos devuelven un ``DataFrame`` y no escriben nada: quien las llama decide
si guardar. El notebook ``configuracion1.ipynb`` las invoca directamente.
"""
from __future__ import annotations

import pandas as pd

from .data import build_specimen_batches, load_labels
from .evaluate import ABLATION, build_submission
from .train import Config, TRAIN_POOL, train_ensemble

BARRIDOS_POR_DEFECTO = {"lambda_monotonic": (0.0, 0.5, 2.0),
                        "physics_warmup_frac": (0.0, 0.25, 0.5)}


def _penalizaciones(ajustes: dict, n_semillas: int, etiquetas) -> dict:
    """Entrena un ensamblado con ``ajustes`` y devuelve la penalización de T7 y T8."""
    cfg = Config(**ajustes)
    lotes = build_specimen_batches(etiquetas, cfg.m_exponent)
    nombres = [n for n in TRAIN_POOL if n in lotes]
    modelos = train_ensemble(cfg, lotes, nombres, n_seeds=n_semillas)
    entregas = {n: build_submission(modelos, lotes, cfg, n) for n in ("T7", "T8")}
    return {
        "T7": entregas["T7"]["penalizacion"],
        "T8": entregas["T8"]["penalizacion"],
        "correcciones": sum(e["n_correcciones_monotonia"] for e in entregas.values()),
    }


def dispersion_ablacion(n_tandas: int = 6, n_semillas: int = 5, etiquetas=None,
                        variantes=None, verbose: bool = True) -> pd.DataFrame:
    """Recorrido de cada variante de la ablación sobre tandas de semillas.

    ``variantes`` permite pasar la ablación construida sobre la configuración
    que la seleccion acaba de elegir, en vez de la cableada en ``ABLATION``.
    """
    etiquetas = load_labels() if etiquetas is None else etiquetas
    variantes = ABLATION if variantes is None else variantes
    filas = []
    for nombre, ajustes in variantes.items():
        for tanda in range(n_tandas):
            pen = _penalizaciones({**ajustes, "seed": 1000 * tanda}, n_semillas, etiquetas)
            filas.append({"variante": nombre, "tanda": tanda,
                          "T7": pen["T7"], "T8": pen["T8"],
                          "total": pen["T7"] + pen["T8"]})
            if verbose:
                print(f"  {nombre[:42]:44s} tanda {tanda}  T7 {pen['T7']:8.2f}", flush=True)
    return pd.DataFrame(filas)


def sensibilidad_pesos(barridos: dict | None = None, n_tandas: int = 3,
                       n_semillas: int = 5, etiquetas=None, base=None,
                       verbose: bool = True) -> pd.DataFrame:
    """Efecto de los pesos que no se eligen, contra el ruido de semilla.

    ``base`` acepta una ``Config`` ya construida o un diccionario de ajustes;
    por omision se usa la Configuracion 1 cableada en ``ABLATION``.
    """
    from dataclasses import asdict, is_dataclass

    barridos = BARRIDOS_POR_DEFECTO if barridos is None else barridos
    etiquetas = load_labels() if etiquetas is None else etiquetas
    if base is None:
        base = ABLATION["Configuración 1 (1D-CNN + PINN)"]
    elif is_dataclass(base):
        base = {k: v for k, v in asdict(base).items() if k != "log_c_prior"}
    filas = []
    for peso, valores in barridos.items():
        for valor in valores:
            for tanda in range(n_tandas):
                pen = _penalizaciones({**base, peso: valor, "seed": 1000 * tanda},
                                      n_semillas, etiquetas)
                filas.append({"peso": peso, "valor": valor, "tanda": tanda, **pen})
                if verbose:
                    print(f"  {peso:22s} = {valor:<5} tanda {tanda}  "
                          f"T7 {pen['T7']:8.2f}", flush=True)
    return pd.DataFrame(filas)
