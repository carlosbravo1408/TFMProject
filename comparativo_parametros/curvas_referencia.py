from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "PHMDC2019_Data"

PUESTOS = {
    "1.º Youn et al. (2019)": "1st_place_baseline",
    "2.º Kong et al. (2020)": "2nd_place_baseline",
    "3.º Rao et al. (2020)": "3rd_place_baseline",
}

# The three baselines define modules with the same names, so they are unloaded
# between imports.
_COLISIONAN = ("pipeline", "scoring", "data_loader", "features", "signal_processing",
               "crack_estimator", "physics_models", "paris_law", "trans_fitting",
               "variable_loading")


def _entrar(carpeta: str):
    for nombre in [m for m in list(sys.modules) if m in _COLISIONAN]:
        del sys.modules[nombre]
    sys.path.insert(0, str(RAIZ / carpeta))


def _salir():
    sys.path.pop(0)


def curvas_publicadas() -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    constantes = {"1st_place_baseline": ("PAPER_TABLES", "predicted_mm"),
                  "2nd_place_baseline": ("PAPER_PREDICTIONS", None),
                  "3rd_place_baseline": ("PAPER_TABLE_A", "estimated_mm")}
    salida = {}
    for puesto, carpeta in PUESTOS.items():
        nombre, campo = constantes[carpeta]
        _entrar(carpeta)
        tabla = getattr(__import__("pipeline"), nombre)
        salida[puesto] = {}
        for esp in ("T7", "T8"):
            # The 2nd place stores {cycle: mm}.
            if campo is None:
                ciclos = sorted(tabla[esp])
                valores = [tabla[esp][c] for c in ciclos]
            else:
                ciclos, valores = tabla[esp]["cycle"], tabla[esp][campo]
            salida[puesto][esp] = (np.asarray(ciclos, float), np.asarray(valores, float))
        _salir()
    return salida


def curvas_replicas() -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    salida: dict[str, dict] = {}

    def anotar(puesto, esp, ciclos, valores):
        salida.setdefault(puesto, {})[esp] = (np.asarray(ciclos, float),
                                              np.asarray(valores, float))

    _entrar("3rd_place_baseline")
    import pipeline as p3
    for esp, df in p3.Angler3rdPlacePipeline(DATOS).run().items():
        anotar("3.º Rao et al. (2020)", esp, df["cycle"], df["estimated_crack_mm"])
    _salir()

    _entrar("2nd_place_baseline")
    import pipeline as p2
    for esp, df in p2.HybridPipeline(DATOS).run().groupby("specimen"):
        anotar("2.º Kong et al. (2020)", esp, df["cycle"], df["estimated_crack_mm"])
    _salir()

    _entrar("1st_place_baseline")
    import pipeline as p1
    from data_loader import load_all, TRAINING_SPECIMENS
    from variable_loading import (fit_paris_law_exponent, equivalent_stress_ratio,
                                  variable_loading_exponent)
    entrenado = p1.train_estimator(DATOS)
    esp7 = load_all(DATOS, TRAINING_SPECIMENS + ["T7"])
    r7 = p1.predict_specimen(entrenado, esp7["T7"], ["T3", "T4"], esp7,
                             lam_translocate=30000.0)
    esp8 = load_all(DATOS, TRAINING_SPECIMENS + ["T8"])
    m_aj, _ = fit_paris_law_exponent(load_all(DATOS, TRAINING_SPECIMENS))
    expo = variable_loading_exponent(
        m_aj, equivalent_stress_ratio(esp8["T1"].loading_profile(),
                                      esp8["T8"].loading_profile()))
    r8 = p1.predict_specimen(entrenado, esp8["T8"], ["T3", "T4"], esp8,
                             lam_translocate=30000.0, variable_loading_exponent=expo,
                             baseline_override=50000)
    for esp, r in (("T7", r7), ("T8", r8)):
        anotar("1.º Youn et al. (2019)", esp, r["score_table"]["cycle"],
               r["score_table"]["estimated"])
    _salir()
    return salida
