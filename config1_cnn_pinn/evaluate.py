from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from physics_calibration.data import load_curve
from physics_calibration.objectives import phm_penalty
from . import signals as S
from .data import build_specimen_batches, load_labels
from .physics import equivalent_stress_range, fit_log_c, propagate_mm
from .train import Config, RESULTS, TRAIN_POOL, evaluate_ensemble, train_ensemble
from src.referencias_phm import PUBLICADO, REPLICA

REFERENCES = {
    **{f"{k} (publicado)": {"T7": v["T7"], "T8": v["T8"], "total": v["total"]}
       for k, v in PUBLICADO.items()},
    **{f"{k} (réplica local)": dict(v) for k, v in REPLICA.items()},
}

# Kept for compatibility with earlier result files.
WINNERS = {k: v["total"] for k, v in REFERENCES.items() if "publicado" in k}


def cota_entrega(nombres=TRAIN_POOL) -> float:
    # Second largest rather than largest: both tie on the worst training fold and the
    # second wins on the mean over the coefficient band.
    maximos = sorted(float(load_curve(n).crack_mm.max()) for n in nombres)
    return maximos[-2]


COTA_ENTREGA = cota_entrega()


def last_signal_cycle(name: str) -> int:
    return max(S.signal_cycles(name=name))


def n_observed_nonzero(name: str) -> int:
    curve = load_curve(name)
    return int(np.sum(curve.cycles <= last_signal_cycle(name)))


def build_submission(models, batches, cfg: Config, name: str,
                     n_observed: int | None = None) -> dict:
    curve = load_curve(name)
    d_sigma = equivalent_stress_range(curve.load_block, cfg.m_exponent)
    if n_observed is None:
        n_observed = n_observed_nonzero(name)

    cota = cfg.cota_entrega_mm if cfg.cota_entrega_mm is not None else 20.0
    cycles_obs, est_obs, true_obs, log_c = evaluate_ensemble(models, batches[name], cfg)

    # A zero label only means the crack was below the optical threshold.
    keep = true_obs > 0
    anchor_cycles, anchor_est = cycles_obs[keep][:n_observed], est_obs[keep][:n_observed]
    cutoff = float(anchor_cycles[-1])

    target_cycles = curve.cycles
    predicted = np.empty(len(target_cycles))
    source = []
    for i, cyc in enumerate(target_cycles):
        if cyc <= cutoff and cyc in set(cycles_obs):
            predicted[i] = est_obs[cycles_obs == cyc][0]
            source.append("estimación 1D-CNN")
        else:
            predicted[i] = propagate_mm(anchor_est[-1], cyc - anchor_cycles[-1],
                                        10.0 ** log_c, cfg.m_exponent, d_sigma,
                                        a_max_mm=cota)
            source.append("extrapolación Paris")

    # Only the estimated half can decrease; the analytic extrapolation is monotone.
    monotone = np.maximum.accumulate(predicted)
    n_fixes = int(np.sum(monotone > predicted + 1e-9))

    true = curve.crack_mm
    return {
        "specimen": name,
        "cycles": target_cycles.tolist(),
        "true_mm": true.tolist(),
        "pred_mm": monotone.tolist(),
        "pred_raw_mm": predicted.tolist(),
        "source": source,
        "cutoff_cycle": int(cutoff),
        "log10_C_predicho": log_c,
        "log10_C_implicado": fit_log_c(curve.cycles, curve.crack_mm,
                                       cfg.m_exponent, d_sigma),
        "n_correcciones_monotonia": n_fixes,
        "rmse_mm": float(np.sqrt(np.mean((monotone - true) ** 2))),
        "penalizacion": float(phm_penalty(monotone[None, :], true, curve.final_crack_mm)[0]),
        "penalizacion_estimacion": float(phm_penalty(
            monotone[None, :][:, np.array(source) == "estimación 1D-CNN"],
            true[np.array(source) == "estimación 1D-CNN"], curve.final_crack_mm)[0]),
        "penalizacion_prognosis": float(phm_penalty(
            monotone[None, :][:, np.array(source) == "extrapolación Paris"],
            true[np.array(source) == "extrapolación Paris"], curve.final_crack_mm)[0]),
    }


def loso_challenge_score(cfg: Config, batches, folds=("T1", "T3", "T4", "T6"),
                         n_seeds: int = 1) -> pd.DataFrame:
    rows = []
    for held_out in folds:
        train_names = [n for n in TRAIN_POOL if n != held_out and n in batches]
        models = train_ensemble(cfg, batches, train_names, n_seeds=n_seeds)
        sub = build_submission(models, batches, cfg, held_out,
                               n_observed=n_observed_nonzero("T7"))
        rows.append({
            "fold": held_out,
            "penalizacion": sub["penalizacion"],
            "penalizacion_estimacion": sub["penalizacion_estimacion"],
            "penalizacion_prognosis": sub["penalizacion_prognosis"],
            "rmse_mm": sub["rmse_mm"],
            "log10_C_predicho": sub["log10_C_predicho"],
            "log10_C_implicado": sub["log10_C_implicado"],
        })
    return pd.DataFrame(rows)


def coefficient_diagnostic(models, batches, cfg: Config, name: str) -> pd.DataFrame:
    curve = load_curve(name)
    d_sigma = equivalent_stress_range(curve.load_block, cfg.m_exponent)
    cota = cfg.cota_entrega_mm if cfg.cota_entrega_mm is not None else 20.0
    sub = build_submission(models, batches, cfg, name)
    anchors = [i for i, s in enumerate(sub["source"]) if s == "estimación 1D-CNN"]
    a0, n0 = sub["pred_mm"][anchors[-1]], curve.cycles[anchors[-1]]

    rows = []
    for label, log_c in (("C predicho por el PINN", sub["log10_C_predicho"]),
                         ("C implicado por la verdad (oráculo)", sub["log10_C_implicado"])):
        pred = np.array(sub["pred_mm"], dtype=float).copy()
        for i, cycle in enumerate(curve.cycles):
            if i > anchors[-1]:
                pred[i] = propagate_mm(a0, cycle - n0, 10.0 ** log_c,
                                       cfg.m_exponent, d_sigma, a_max_mm=cota)
        pred = np.maximum.accumulate(pred)
        rows.append({
            "espécimen": name, "coeficiente": label, "log10 C": round(log_c, 3),
            "penalización": round(float(phm_penalty(
                pred[None, :], curve.crack_mm, curve.final_crack_mm)[0]), 2),
            "RMSE (mm)": round(float(np.sqrt(np.mean((pred - curve.crack_mm) ** 2))), 3),
        })
    return pd.DataFrame(rows)


def run(cfg: Config | None = None, n_seeds: int = 5, verbose: bool = True):
    cfg = cfg or Config()
    labels = load_labels()
    batches = build_specimen_batches(labels, cfg.m_exponent)

    train_names = [n for n in TRAIN_POOL if n in batches]
    if verbose:
        print(f"Entrenando el ensamblado final ({n_seeds} semillas) sobre "
              f"{', '.join(train_names)}…")
    models = train_ensemble(cfg, batches, train_names, n_seeds=n_seeds)

    submissions = [build_submission(models, batches, cfg, name) for name in ("T7", "T8")]
    return models, submissions, batches


def summarise(submissions) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "espécimen": s["specimen"],
            "RMSE (mm)": round(s["rmse_mm"], 4),
            "penalización": round(s["penalizacion"], 3),
            "de la estimación": round(s["penalizacion_estimacion"], 3),
            "de la prognosis": round(s["penalizacion_prognosis"], 3),
            "log10 C predicho": round(s["log10_C_predicho"], 3),
            "log10 C implicado": round(s["log10_C_implicado"], 3),
            "correcciones monotonía": s["n_correcciones_monotonia"],
        }
        for s in submissions
    ])


# data_weight/anchor_weight, m_exponent and cota_entrega_mm come from the second,
# third and fourth selection stages (select_weighting, select_prognosis and
# cota_extrapolacion), all scored on training folds.
BASE = dict(epochs=600, dropout=0.3, weight_decay=1e-3,
            data_weight="anchor", anchor_weight=48.0, m_exponent=2.00,
            cota_entrega_mm=COTA_ENTREGA)

# Chosen by the first selection stage (select).
LAMBDA_PHYSICS = 1.0

ABLATION = {
    "Configuración 1 (1D-CNN + PINN)":
        dict(**BASE, lambda_physics=LAMBDA_PHYSICS, lambda_log_c_prior=1.0, data_loss="asym"),
    "Control A — 1D-CNN sola (sin física)":
        dict(**BASE, lambda_physics=0.0, lambda_log_c_prior=0.0, data_loss="asym"),
    "Control B — 1D-CNN + PINN, pérdida simétrica (MSE)":
        dict(**BASE, lambda_physics=LAMBDA_PHYSICS, lambda_log_c_prior=1.0, data_loss="mse"),
    "Control C — PINN sin prior sobre C":
        dict(**BASE, lambda_physics=LAMBDA_PHYSICS, lambda_log_c_prior=0.0, data_loss="asym"),
}


def ejecutar_ablacion(n_seeds: int = 5, variantes=None, verbose: bool = True):
    variantes = ABLATION if variantes is None else variantes
    rows, detalle, diagnostico = [], {}, None
    for label, overrides in variantes.items():
        cfg = Config(**overrides)
        if verbose:
            print(f"\n=== {label} ===")
        models, submissions, batches = run(cfg, n_seeds=n_seeds, verbose=verbose)
        total = sum(s["penalizacion"] for s in submissions)
        if verbose:
            print(summarise(submissions).to_string(index=False))
            print(f"  penalización T7+T8 = {total:.2f}")
        rows.append({
            "variante": label,
            "penalización T7+T8": round(total, 2),
            "T7": round(submissions[0]["penalizacion"], 2),
            "T8": round(submissions[1]["penalizacion"], 2),
            "RMSE T7 (mm)": round(submissions[0]["rmse_mm"], 3),
            "RMSE T8 (mm)": round(submissions[1]["rmse_mm"], 3),
            "|Δlog10 C| T7": round(abs(submissions[0]["log10_C_predicho"]
                                       - submissions[0]["log10_C_implicado"]), 3),
            "|Δlog10 C| T8": round(abs(submissions[1]["log10_C_predicho"]
                                       - submissions[1]["log10_C_implicado"]), 3),
        })
        detalle[label] = {"config": asdict(cfg), "submissions": submissions,
                          "models": models, "batches": batches}
        if label.startswith("Configuración 1"):
            diagnostico = pd.concat(
                [coefficient_diagnostic(models, batches, cfg, n) for n in ("T7", "T8")],
                ignore_index=True)
            if verbose:
                print("\n  Diagnóstico — ¿estimador o coeficiente?")
                print(diagnostico.to_string(index=False))

    return pd.DataFrame(rows), detalle, diagnostico


def tabla_comparativa(ablacion: pd.DataFrame) -> str:
    cfg1 = ablacion.iloc[0]
    lineas = [f"  {'':34s} {'T7':>9s} {'T8':>10s} {'total':>10s}",
              f"  {'Configuración 1 (este trabajo)':34s} {cfg1['T7']:9.2f} "
              f"{cfg1['T8']:10.2f} {cfg1['penalización T7+T8']:10.2f}"]
    for label, v in REFERENCES.items():
        t7 = f"{v['T7']:9.2f}" if v["T7"] is not None else f"{'n/d':>9s}"
        t8 = f"{v['T8']:10.2f}" if v["T8"] is not None else f"{'n/d':>10s}"
        lineas.append(f"  {label:34s} {t7} {t8} {v['total']:10.2f}")
    lineas.append("\n  n/d: el paper del 1.º publica su score en mm crudos, no la")
    lineas.append("       penalización normalizada por espécimen.")
    return "\n".join(lineas)


def main() -> None:
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)

    ablation, detalle, diag = ejecutar_ablacion(n_seeds=5)
    if diag is not None:
        diag.to_csv(RESULTS / "diagnostico_coeficiente.csv", index=False)

    print("\n" + "=" * 100)
    print("ABLACIÓN — penalización oficial sobre T7 y T8 (protocolo del certamen)")
    print("=" * 100)
    print(ablation.to_string(index=False))
    print("\nComparación por espécimen (la única válida):")
    print(tabla_comparativa(ablation))

    ablation.to_csv(RESULTS / "ablacion.csv", index=False)
    (RESULTS / "entrega.json").write_text(json.dumps(
        {k: {"config": v["config"], "submissions": v["submissions"]}
         for k, v in detalle.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
