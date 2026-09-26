"""Challenge-protocol evaluation of Configuration 1 on T7 and T8.

Two protocols are reported side by side, because they answer different
questions and conflating them is how prognosis papers end up incomparable.

**LOSO** (in ``train.py``) — train on the other training specimens, test on a
held-out one. The honest estimate of how the estimator generalises to a *new*
specimen. It is strictly harder than what the challenge asked.

**Challenge protocol** (here) — train on all of T1-T6, then produce the full
submission for T7 and T8 and score it with the official penalty. This is the
number that is comparable with the published 7.36 / 7.63 / 16.14.

The submission has two halves, which is precisely the structure the challenge
imposed:

*Estimation.* For the cycles that still have Lamb-wave signals (T7 up to 47022,
T8 up to 76931) the 1D-CNN estimates the crack length directly.

*Prognosis.* After that the signals stop. Configuration 1 has no multi-step
integrator — that is component (c), and it is what Configuration 2 adds — so it
extrapolates with the **closed-form analytic solution** of the Paris law, using
the coefficient its own PINN head identified from the waveforms. The comparison
between this and Configuration 2's RK4 integration is exactly the ablation
step: it isolates what explicit numerical integration over the true load block
buys over the analytic constant-amplitude solution.

Monotonicity comes for free: with the identified exponent m = 2 the analytic
solution is a growing exponential, so the extrapolated half of the submission
can never violate the challenge's M(i) factor. The estimated half can, and is
therefore reported separately.
"""
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
from referencias_phm import PUBLICADO, REPLICA

# Reference scores, **per specimen**, from the single table every notebook
# shares. Quoting only the T7+T8 totals invites the error of comparing a
# per-specimen result against a total, which is not a comparison at all.
REFERENCES = {
    **{f"{k} (publicado)": {"T7": v["T7"], "T8": v["T8"], "total": v["total"]}
       for k, v in PUBLICADO.items()},
    **{f"{k} (réplica local)": dict(v) for k, v in REPLICA.items()},
}

# Kept for backwards compatibility with earlier result files.
WINNERS = {k: v["total"] for k, v in REFERENCES.items() if "publicado" in k}


def cota_entrega(nombres=TRAIN_POOL) -> float:
    """Cota de dominio para la extrapolacion: segunda mayor grieta observada.

    Es una **regla**, no un numero: sobre T1-T6 vale 7.24 mm, y con otro
    conjunto de especimenes valdria otra cosa. Se toma la segunda mayor y no la
    mayor porque, sometidas al criterio sin fuga sobre los folds de
    entrenamiento, ambas empatan en el peor caso y la segunda mayor queda por
    delante en la media sobre la banda de incertidumbre del coeficiente.

    Al puntuar un fold, ``cota_extrapolacion`` recalcula esta regla **excluyendo
    el especimen puntuado**, para que la cota no conozca la respuesta que se le
    pide. Para T7 y T8 la exclusion no procede: no son de entrenamiento.
    """
    maximos = sorted(float(load_curve(n).crack_mm.max()) for n in nombres)
    return maximos[-2]


COTA_ENTREGA = cota_entrega()


def last_signal_cycle(name: str) -> int:
    return max(S.signal_cycles(name=name))


def n_observed_nonzero(name: str) -> int:
    """Non-zero crack measurements whose cycle still has Lamb-wave signals.

    Two for both validation specimens (T7 up to cycle 47022, T8 up to 76931).
    Applying the *same* budget to a held-out training specimen is what makes a
    LOSO fold a faithful rehearsal of the challenge rather than a much easier
    problem.
    """
    curve = load_curve(name)
    return int(np.sum(curve.cycles <= last_signal_cycle(name)))


def build_submission(models, batches, cfg: Config, name: str,
                     n_observed: int | None = None) -> dict:
    """Estimate where signals exist, extrapolate analytically afterwards.

    ``n_observed`` is the number of *non-zero* crack measurements the estimator
    is allowed to see. ``None`` means "whatever the released signals actually
    provide", which is the real challenge condition.
    """
    curve = load_curve(name)
    d_sigma = equivalent_stress_range(curve.load_block, cfg.m_exponent)
    if n_observed is None:
        n_observed = n_observed_nonzero(name)

    cota = cfg.cota_entrega_mm if cfg.cota_entrega_mm is not None else 20.0
    cycles_obs, est_obs, true_obs, log_c = evaluate_ensemble(models, batches[name], cfg)

    # Only cycles with a non-zero label take part in the propagation: the growth
    # law describes propagation, and a zero label just means the crack was below
    # the optical detection threshold.
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

    # The estimated half is not monotone by construction; enforce it there so
    # the submission never trips M(i). This is the one post-hoc heuristic in the
    # pipeline, and it is confined to the estimated cycles.
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
    """Rehearse the full challenge protocol on held-out *training* specimens.

    Selecting Configuration 1 on estimation RMSE alone actively penalises its
    own physics term: the PINN head exists to make the coefficient identifiable
    for the extrapolated half of the submission, and estimation RMSE never
    looks at that half. Worse, with ``lambda_physics = 0`` the coefficient head
    receives no gradient at all, so a selection driven by RMSE would happily
    pick a configuration whose PINN is inert.

    This scorer instead runs the whole pipeline on each held-out specimen under
    the same two-anchor budget the validation specimens impose, and reports the
    official penalty. It never touches T7 or T8.
    """
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
    """Is a bad submission the estimator's fault or the coefficient's?

    Re-runs the extrapolation from the network's own anchors twice: once with
    the coefficient its PINN head predicted, once with the coefficient the
    ground-truth curve implies. Any gap between the two is attributable purely
    to the identified physics, and any error remaining in the oracle row is
    attributable to the estimator. Without this split, a diverging T8 could be
    blamed on either half of the model.

    Ambas filas se construyen **bajo la misma cota de entrega** que la entrega
    oficial. Evaluar el oraculo sin ella exageraria lo que aporta acertar el
    coeficiente, porque le atribuiria tambien el dano que la cota ya evita: los
    dos mecanismos actuan sobre la misma sobrestimacion y se solapan.
    """
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


# The ablation actually reported. The capacity hyper-parameters are the ones
# selected on the training folds; the variants differ only in the terms whose
# contribution is the question.
#
# ``data_weight``/``anchor_weight`` vienen de la segunda etapa de seleccion
# (``select_weighting``), puntuada por el criterio sin fuga: el peor fold de
# entrenamiento bajo el protocolo completo del certamen, sobre toda la banda de
# incertidumbre del coeficiente. A 600 epocas ese criterio elige ``ancla x48``
# (430,4 frente a 1102,4 de ``x12`` y 1368,3 de copiar T(i) del certamen). Son
# parte del *estimador*, no de la pregunta de la ablacion, asi que todas las
# variantes de abajo los heredan.
#
# ``m_exponent`` viene de la tercera etapa (``select_prognosis``), por minimax
# sobre la banda de incertidumbre del coeficiente en folds de entrenamiento.
# Sobreescribe a ``priors.RECOMMENDED_EXPONENT`` (2,25), que se eligio al
# coeficiente nominal.
#
# ``cota_entrega_mm`` viene de la cuarta etapa (``cota_extrapolacion``): la
# extrapolacion de Paris no tiene cota superior, de modo que un coeficiente algo
# alto no da un error algo mayor sino una grieta arbitrariamente larga. La regla
# elegida sobre folds de entrenamiento, con exclusion del fold puntuado, lleva
# el peor caso sobre la banda de 510,4 a 141,2.
BASE = dict(epochs=600, dropout=0.3, weight_decay=1e-3,
            data_weight="anchor", anchor_weight=48.0, m_exponent=2.00,
            cota_entrega_mm=COTA_ENTREGA)

# ``lambda_physics`` lo elige la primera etapa (``select``). A 600 epocas la
# mejor combinacion tiene el termino fisico **activo** (82,2 de penalizacion
# media frente a 84,0 con la fisica apagada), de modo que ya no hace falta
# fijarlo por definicion como ocurria a 100 epocas, donde el criterio prefería
# apagarlo. La diferencia es pequeña y conviene decirlo asi.
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
    """Entrena y evalua cada variante de la ablacion sobre T7 y T8.

    Devuelve ``(tabla, detalle, diagnostico)``:

    ``tabla``        una fila por variante, con la penalizacion oficial.
    ``detalle``      ``{variante: {"config", "submissions", "models", "batches"}}``,
                     para que quien llame pueda seguir trabajando con el modelo
                     entrenado (el barrido de coeficiente, por ejemplo) sin
                     volver a entrenarlo.
    ``diagnostico``  el desglose estimador/coeficiente de la Configuracion 1.

    No escribe ni lee ficheros.
    """
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
    """Comparacion por especimen contra las referencias publicadas y replicadas."""
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
    """Atajo de linea de comandos. El notebook llama a ``ejecutar_ablacion``."""
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
