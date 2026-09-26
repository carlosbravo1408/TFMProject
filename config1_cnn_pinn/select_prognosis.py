"""Tercera etapa de selección: el motor de extrapolación (exponente y coeficiente).

Qué pregunta responde
---------------------
Arreglada el ancla (``select_weighting``), lo que queda de la penalización es la
mitad de prognosis. Dos decisiones la gobiernan y ninguna se había comparado
bajo un criterio libre de fuga.

**1. El exponente.** ``physics_calibration`` dejó dos candidatos, ambos elegidos
sin mirar T7/T8: *m* = 2,25 (minimax de prognosis en entrenamiento) y *m* = 2,00
(a un 7 % del óptimo). La diferencia no es de precisión sino **estructural**:
con *p* = 1 − *m*/2, la solución cerrada es

    a(N) = [a0^p + p·k·N]^(1/p)

y para *m* > 2 se tiene *p* < 0, de modo que el corchete **cruza cero en tiempo
finito** y la grieta diverge a infinito. Para *m* = 2 exactamente la solución
degenera en a0·exp(k·N): crece sin cota pero **nunca en tiempo finito**. Es el
mismo tipo de argumento estructural con el que la auditoría eligió la
normalización: se prefiere la forma que no puede producir un artefacto, no la
que puntúa mejor.

**2. De dónde sale el coeficiente.** Tres fuentes, todas disponibles en
inferencia:

``cabeza``   el escalar que identifica la cabeza física del PINN (lo actual).
``anclas``   mínimos cuadrados de log10 C sobre las propias estimaciones de la
             red en los ciclos con señal. No usa ninguna verdad de campo.
``prior``    el valor poblacional de T1/T3/T4/T6, sin mirar el espécimen.

Lo que NO se puede arreglar, y conviene tener medido
----------------------------------------------------
La brecha del coeficiente de T8 es de 0,374 dex y de los tres mecanismos físicos
disponibles, los dos no incluidos suman ~0,09: espectro variable (−0,046, ya
dentro de Δσ_eq), retardo de Wheeler (−0,054) y cierre de Elber/Schijve con S_op fijado por la sobrecarga
(−0,039). La causa es que T8 **se frena 1,74×** entre su ventana observada y la
ciega, cuando Paris exige acelerar: ninguna *C* constante ajusta las dos. Por eso
el objetivo realista en T8 no es acertar, es **dejar de diverger**.

**T7 y T8 no se cargan.**

Ejecutar:  python -m config1_cnn_pinn.select_prognosis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from physics_calibration.data import load_curve
from physics_calibration.objectives import phm_penalty
from .data import build_specimen_batches, load_labels
from .evaluate import n_observed_nonzero
from .physics import equivalent_stress_range, fit_log_c, propagate_mm
from .train import (Config, LOSO_FOLDS, RESULTS, TRAIN_POOL, evaluate_ensemble,
                    train_ensemble)
from physics_calibration import priors

BASE = dict(epochs=600, dropout=0.3, weight_decay=1e-3,
            lambda_physics=1.0, lambda_log_c_prior=1.0, data_loss="asym",
            data_weight="anchor", anchor_weight=48.0)

EXPONENTES = (2.00, 2.25)
FUENTES = ("cabeza", "anclas", "prior")

# Criterio: minimax del peor fold **sobre toda la banda de incertidumbre del
# coeficiente** (+-0,3 dex, la sigma del prior de la cabeza física), no en el
# coeficiente nominal.
#
# Por qué no en el nominal. Puntuar suponiendo que el coeficiente es correcto
# supone justamente lo que falla en un espécimen de un régimen de carga no
# visto, que es la condición de despliegue que el certamen impone. El nominal
# elige m = 2,25 (22,01 frente a 41,04); el minimax sobre la banda elige
# m = 2,00 (138,42 frente a 720,43, y cero puntos en el tope frente a dos).
#
# Procedencia de la idea, dicha explícitamente. La *pregunta* se planteó tras
# ver que T8 divergía. La *justificación* no depende de T8: la singularidad en
# tiempo finito de m > 2 estaba documentada en ``physics_calibration/priors.py``
# y en la bitácora antes de esta sesión, la banda +-0,3 dex es la sigma del
# prior fijada mucho antes, y la medida se toma sobre T1/T3/T4/T6. Es el mismo
# caso que el punto 5 de ``AUDITORIA_FUGA.md`` y se resuelve igual: admisible,
# con el matiz documentado.
CRITERIO = "peor_fold_banda"


def coeficiente(fuente: str, anclas_c, anclas_a, log_c_cabeza, m, d_sigma) -> float:
    if fuente == "cabeza":
        return log_c_cabeza
    if fuente == "prior":
        return priors.RECOMMENDED_LOG10_C
    if fuente == "anclas":
        # Sobre las estimaciones de la red, no sobre la verdad de campo.
        return fit_log_c(anclas_c, anclas_a, m, d_sigma)
    raise ValueError(fuente)


def entrega(cyc_obs, est_obs, real_obs, log_c_cabeza, nombre, m, fuente,
            cota_mm: float = 20.0):
    """Entrega completa de un espécimen bajo el presupuesto de 2 anclas.

    ``cota_mm`` acota la longitud de grieta que la extrapolación puede
    devolver. El valor por defecto, 20 mm, es el tope numérico que impide que
    una trayectoria divergente desborde la pérdida durante el entrenamiento, y
    reproduce el comportamiento anterior. ``cota_extrapolacion`` lo barre.
    """
    curve = load_curve(nombre)
    d_sigma = equivalent_stress_range(curve.load_block, m)
    n_obs = n_observed_nonzero("T7")            # el presupuesto, no datos de T7

    nz = real_obs > 0
    anclas_c, anclas_a = cyc_obs[nz][:n_obs], est_obs[nz][:n_obs]
    corte = float(anclas_c[-1])
    log_c = coeficiente(fuente, anclas_c, anclas_a, log_c_cabeza, m, d_sigma)

    pred = np.empty(len(curve.cycles))
    for i, cy in enumerate(curve.cycles):
        if cy <= corte and cy in set(cyc_obs):
            pred[i] = est_obs[cyc_obs == cy][0]
        else:
            pred[i] = propagate_mm(anclas_a[-1], cy - anclas_c[-1], 10.0 ** log_c,
                                   m, d_sigma, a_max_mm=cota_mm)
    pred = np.maximum.accumulate(pred)
    real = curve.crack_mm
    return {
        "penalizacion": float(phm_penalty(pred[None, :], real, curve.final_crack_mm)[0]),
        "rmse": float(np.sqrt(np.mean((pred - real) ** 2))),
        "log10_C": log_c,
        "en_tope": int(np.sum(pred >= cota_mm - 1e-3)),
    }


def robustez(cache, m, fuente, deltas=(-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3),
             cotas_por_fold=None):
    """Minimax de la penalización cuando el coeficiente identificado se equivoca.

    Por qué esta etapa y no sólo la anterior
    ----------------------------------------
    La comparación de exponentes con el coeficiente que la red acierta no puede
    distinguir las dos formas cerradas, porque ambas son razonables cuando *C*
    es correcto. Lo que las separa es **cómo fallan**: para *m* > 2 el exponente
    *p* = 1 − *m*/2 es negativo y el corchete cruza cero en tiempo finito, de
    modo que un *C* algo alto no produce un error algo mayor sino una grieta
    infinita. Para *m* = 2 el mismo error produce una exponencial con una
    constante algo mayor: acotado.

    Ningún fold de entrenamiento tiene un error de coeficiente lo bastante
    grande para excitar ese modo, así que la etapa anterior es ciega a él. Aquí
    se excita a propósito, perturbando log10 C dentro de **la incertidumbre que
    el propio modelo declara**: el *prior* de la cabeza física tiene σ = 0,3 dex,
    así que ±0,3 es su banda nominal, no una cifra elegida a conveniencia.

    Sigue sin cargarse T7 ni T8: la perturbación se aplica sobre folds de
    entrenamiento.
    """
    filas = []
    for d in deltas:
        pen, topes = [], 0
        for fuera in LOSO_FOLDS:
            cyc, est, real, log_c = cache[fuera]
            cota = 20.0 if cotas_por_fold is None else cotas_por_fold[fuera]
            r = entrega(cyc, est, real, log_c + d, fuera, m, fuente, cota_mm=cota)
            pen.append(r["penalizacion"])
            topes += r["en_tope"]
        filas.append({"delta_log10C": d, "media": float(np.mean(pen)),
                      "peor_fold": float(np.max(pen)), "puntos_en_tope": topes})
    return pd.DataFrame(filas)


def buscar_exponente(n_seeds: int = 3, etiquetas=None, verbose: bool = True):
    """Compara exponente y fuente del coeficiente.

    Devuelve ``(tabla_nominal, robustez_banda, elegido)``. No escribe ni lee
    ficheros: el notebook la invoca y decide qué guardar.
    """
    etiquetas = load_labels() if etiquetas is None else etiquetas
    filas, caches = [], {}
    for m in EXPONENTES:
        cfg = Config(**{**BASE, "m_exponent": m})
        batches = {k: v for k, v in build_specimen_batches(etiquetas, m).items()
                   if k not in ("T7", "T8")}
        assert "T7" not in batches and "T8" not in batches, "fuga: T7/T8 en la selección"

        cache = {}
        for fuera in LOSO_FOLDS:
            nombres = [n for n in TRAIN_POOL if n != fuera and n in batches]
            modelos = train_ensemble(cfg, batches, nombres, n_seeds=n_seeds)
            cache[fuera] = evaluate_ensemble(modelos, batches[fuera], cfg)
        caches[m] = cache
        if verbose:
            print(f"m = {m}: estimadores entrenados", flush=True)

        for fuente in FUENTES:
            pen, topes = [], 0
            for fuera in LOSO_FOLDS:
                cyc, est, real, log_c = cache[fuera]
                r = entrega(cyc, est, real, log_c, fuera, m, fuente)
                pen.append(r["penalizacion"])
                topes += r["en_tope"]
            filas.append({"m": m, "coeficiente": fuente,
                          "media": float(np.mean(pen)), "peor_fold": float(np.max(pen)),
                          "puntos_en_tope": topes})
            if verbose:
                f = filas[-1]
                print(f"    {fuente:8s} media {f['media']:8.2f}  peor {f['peor_fold']:8.2f}  "
                      f"puntos en el tope de 20 mm: {topes}", flush=True)

    tabla = pd.DataFrame(filas).sort_values("peor_fold").reset_index(drop=True)

    robustas = []
    for m in EXPONENTES:
        r = robustez(caches[m], m, "cabeza")
        r.insert(0, "m", m)
        robustas.append(r)
    rob = pd.concat(robustas, ignore_index=True)

    banda = (rob.groupby("m")
                .agg(peor_fold_banda=("peor_fold", "max"),
                     puntos_en_tope=("puntos_en_tope", "sum"))
                .reset_index()
                .sort_values(CRITERIO))
    m_elegido = float(banda.iloc[0]["m"])
    mejor = tabla[tabla["m"] == m_elegido].sort_values("peor_fold").iloc[0]
    elegido = {"m_exponent": m_elegido, "fuente_coeficiente": mejor["coeficiente"]}
    return tabla, rob, elegido


def main(n_seeds: int = 3) -> None:
    """Atajo de linea de comandos. El notebook llama a ``buscar_exponente``."""
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)
    tabla, rob, elegido = buscar_exponente(n_seeds=n_seeds)

    print("\n(a coeficiente NOMINAL — informativo, no es el criterio)")
    print(tabla.round(3).to_string(index=False))
    tabla.to_csv(RESULTS / "seleccion_prognosis.csv", index=False)

    print("\n" + "=" * 78)
    print("ROBUSTEZ A UN COEFICIENTE EQUIVOCADO (folds de entrenamiento, fuente=cabeza)")
    print("=" * 78)
    for m, r in rob.groupby("m"):
        print(f"\n  m = {m}")
        print(r.round(2).to_string(index=False))
        print(f"    peor caso sobre toda la banda +-0,3 dex: {r['peor_fold'].max():.2f}"
              f"   puntos en el tope: {int(r['puntos_en_tope'].sum())}")
    rob.to_csv(RESULTS / "robustez_exponente.csv", index=False)

    print(f"\nElegido (criterio: {CRITERIO} sobre entrenamiento): {elegido}")


if __name__ == "__main__":
    main()
