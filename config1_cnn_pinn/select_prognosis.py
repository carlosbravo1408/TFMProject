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

# Worst fold over the +/-0.3 dex coefficient band (the prior's sigma), not at the
# nominal coefficient: for m > 2 a slightly high C diverges in finite time.
CRITERIO = "peor_fold_banda"


def coeficiente(fuente: str, anclas_c, anclas_a, log_c_cabeza, m, d_sigma) -> float:
    if fuente == "cabeza":
        return log_c_cabeza
    if fuente == "prior":
        return priors.RECOMMENDED_LOG10_C
    if fuente == "anclas":
        # From the network's estimates, not the ground truth.
        return fit_log_c(anclas_c, anclas_a, m, d_sigma)
    raise ValueError(fuente)


def entrega(cyc_obs, est_obs, real_obs, log_c_cabeza, nombre, m, fuente,
            cota_mm: float = 20.0):
    curve = load_curve(nombre)
    d_sigma = equivalent_stress_range(curve.load_block, m)
    # The protocol's budget; no T7 data is read.
    n_obs = n_observed_nonzero("T7")

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
