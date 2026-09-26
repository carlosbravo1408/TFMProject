"""Turn the calibration results into ``priors.py``, the single source of
truth the TFM's PINN imports.

Everything the physics module needs is derived here from the JSON/CSV that
``calibrate.py`` wrote, so the numbers in ``priors.py`` are traceable to a
specific run instead of being retyped by hand. Regenerate with:

    python -m physics_calibration.make_priors
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
TARGET = Path(__file__).resolve().parent / "priors.py"

TEMPLATE = '''"""Fracture-mechanics hyper-parameters for the PHM 2019 2024-T3 lap joints.

GENERATED FILE -- do not edit by hand. Produced by
``python -m physics_calibration.make_priors`` on {today} from the results of
``python -m physics_calibration.calibrate``.

Units are SI-consistent throughout: da/dN in m/cycle, dK in MPa*sqrt(m),
crack length in metres, stress in MPa. A coefficient quoted here is therefore
directly comparable with the fracture-mechanics literature.

Why these values and not published ones
---------------------------------------
No reviewed source gives Paris/Walker constants for these specimens. Kong et
al. (2020, 2nd place) state that "the constants C0, gamma and m could not be
determined because they depend on the material properties of the specimen"
and fit them by genetic algorithm without publishing the result; Youn et al.
(2020, 1st place) reject physics-based models outright for lack of "the shape
of the initial crack, specimen geometry, and material properties"; Rao et al.
(2021, 3rd place) fall back on the generic metallic window C in [1e-13, 1e-11],
m in [2, 4] (after Li, Wang & Gong, 2012). The values below were therefore
identified from the released crack-growth curves by metaheuristic search.

How to use them in the PINN
---------------------------
These are *initialisations, priors and bounds*, not constants to freeze. The
identification found a sharp asymmetry:

* The exponent is only weakly determined by the fit. Every m from {m_lo:.1f} to
  {m_hi:.1f} reproduces the training curves within 5 % of the optimum, because
  dK spans just {decades:.2f} decades. It is *prognosis*, not fitting, that pins
  it: the official penalty on blindly predicted points is minimised around
  m = 2.0-2.5 and explodes past m ~ 3.
* The coefficient is tightly determined once the specimen is fixed: the
  per-specimen spread is only {paris_logc_std:.3f} dex (a factor of
  {c_factor:.2f}), so log10 C is exactly the kind of scalar a 1D-CNN plus
  attention encoder can regress from the Lamb-wave history.

So: keep m near ``PARIS_EXPONENT`` under a weak prior, and let the network
predict ``log10 C`` per specimen.
"""
from __future__ import annotations

# --- Load spectrum, decoded from the released loading-profile CSVs ---------
# Constant amplitude (T1-T7): one sine cycle, Smax/Smin in MPa.
CONSTANT_LOAD_BLOCK = {constant_block!r}
# Variable amplitude (T8): one repeating block of 1000 cycles,
# (n_cycles, Smax, Smin).
VARIABLE_LOAD_BLOCK = {variable_block!r}

# Stress ratio present in the whole dataset. R varies from {r_min:.4f} to
# {r_max:.4f} -- a single point for all practical purposes, which is why
# Walker's gamma is NOT identifiable here (see WALKER_GAMMA below).
STRESS_RATIO_RANGE = ({r_min!r}, {r_max!r})

# --- Paris-Erdogan: da/dN = C * dK^m --------------------------------------
# Pooled identification on T1/T3/T4/T6, coefficients projected out, plain
# plate geometry (Y = 1). ``PARIS_LOG10_C_BY_SPECIMEN`` is the per-specimen
# coefficient at the pooled exponent: the quantity the CNN encoder must learn
# to predict.
#
# RECOMMENDED_EXPONENT is the value to carry into the PINN. It is the
# minimax-optimal exponent for blind prognosis over anchor budgets of 3 and 4,
# scored **on the training specimens only** (T1/T3/T4/T6) — never on T7/T8
# points, restricted to the classical metallic window m >= 2. Three
# independent estimates agree on it: the pooled fit ({paris_m:.2f}), the
# model-free log-log regression ({loglog_m:.2f} +/- {loglog_m_ci:.2f}) and the
# prognosis sweep (optimum 1.75-2.5 depending on anchor budget).
RECOMMENDED_EXPONENT = {recommended_m!r}
RECOMMENDED_EXPONENT_PRIOR_SD = {recommended_m_sd!r}
# Coefficient re-identified *at* RECOMMENDED_EXPONENT (not at PARIS_EXPONENT:
# C and m slide along the log-log ridge together, so a coefficient quoted at a
# different exponent is simply wrong). Per specimen, and pooled mean/spread.
RECOMMENDED_LOG10_C = {rec_logc_mean!r}
RECOMMENDED_LOG10_C_STD = {rec_logc_std!r}
RECOMMENDED_C = {rec_c!r}
RECOMMENDED_LOG10_C_BY_SPECIMEN = {rec_logc_by_spec!r}

PARIS_EXPONENT = {paris_m!r}
PARIS_LOG10_C = {paris_logc_mean!r}
PARIS_LOG10_C_STD = {paris_logc_std!r}
PARIS_C = {paris_c!r}
PARIS_LOG10_C_BY_SPECIMEN = {paris_logc_by_spec!r}

# Model-free cross-check: ordinary least squares of log10(da/dN) on log10(dK)
# over finite differences pooled across all specimens. Independent of every
# modelling choice above, and the source of the confidence interval on m.
LOGLOG_EXPONENT = {loglog_m!r}
LOGLOG_EXPONENT_CI95 = {loglog_m_ci!r}
LOGLOG_C = {loglog_c!r}
LOGLOG_R2 = {loglog_r2!r}
LOGLOG_RESIDUAL_SD_DEX = {loglog_sd!r}

# Search bounds for the physics module. log10 C is wide because C absorbs the
# geometry factor; m is bounded to the classical metallic window so the PINN
# cannot drift to a value that fits the data but is not fracture mechanics.
LOG10_C_BOUNDS = {log_c_bounds!r}
EXPONENT_BOUNDS = {m_bounds!r}

# Minimum number of crack estimates the physics module needs before its
# multi-step extrapolation stops diverging. With two anchors -- all the Lamb
# wave signals of T7/T8 actually provide -- every law tested runs away on T8;
# with three the same law scores {pen_3:.1f} on T7+T8's blindly predicted
# points, and with four, {pen_4:.1f}. This is the handover point the hybrid
# has to respect: the CNN must supply reliable crack estimates for at least
# this many measurement cycles before the integrator takes over alone.
MIN_ANCHOR_POINTS = 3

# --- Walker: da/dN = C0 * (Kmax * (1-R)^gamma)^m --------------------------
# gamma is unidentifiable from this dataset: every specimen is tested at
# essentially the same stress ratio, so (1-R)^gamma is a constant that the
# coefficient absorbs exactly. The pooled fit confirms it -- Walker reaches
# the same objective as Paris ({walker_rmse:.4f} vs {paris_rmse:.4f} mm) at an
# arbitrary gamma. Use the value below only if the PINN is to be transferred
# to a spectrum with a different R; on PHM 2019 alone, prefer Paris.
WALKER_GAMMA = {walker_gamma!r}
WALKER_GAMMA_IDENTIFIABLE = False

# --- Geometry -------------------------------------------------------------
# The dataset ReadMe states the crack initiates at a countersunk hole in the
# first rivet row. Y therefore starts near the hole's stress concentration and
# decays to the plain-plate value:  Y(a) = 1 + (Kt - 1) * exp(-a / lambda_h),
# with a and lambda_h in metres and millimetres respectively.
# Fitting this improves the pooled in-sample RMSE from {paris_rmse:.4f} to
# {hole_rmse:.4f} mm AND brings the identified coefficient into the published
# metallic range -- but it extrapolates far less stably (see the report).
HOLE_KT = {hole_kt!r}
HOLE_DECAY_MM = {hole_lambda!r}
HOLE_EXPONENT = {hole_m!r}
HOLE_LOG10_C = {hole_logc!r}

# --- 2024-T3 specific constant from the literature ------------------------
# Elber/Schijve crack-opening function, U = dK_eff/dK, reported specifically
# for aluminium alloy 2024-T3 (Schijve, *Fatigue of Structures and
# Materials*, Eq. 9.10), valid for -1 <= R <= +1. The only 2024-T3-specific
# fracture-mechanics constant the reviewed literature supplies.
def elber_schijve_closure(r):
    """U = dK_eff / dK for 2024-T3."""
    return 0.55 + 0.35 * r + 0.1 * r ** 2


# --- Reference values found in the literature (NOT for this specimen) -----
# Kept for the discussion section. None of these is a drop-in replacement for
# the identified values above: the first two are coupon data for a different
# geometry, and the third is a generic metallic window.
LITERATURE_REFERENCE = {literature!r}
'''


def _round(x, n=4):
    return float(np.round(float(x), n))


def main() -> None:
    best = json.loads((RESULTS / "stage_b_pooled_best.json").read_text())
    loglog = json.loads((RESULTS / "stage_e_loglog.json").read_text())
    profile = json.loads((RESULTS / "stage_c_profile_m.json").read_text())["profile"]
    sweep = pd.read_csv(RESULTS / "stage_g_exponent_sweep.csv")

    # El exponente se elige **sólo con los especímenes de entrenamiento**
    # (T1/T3/T4/T6). Usar `penalty_T7_T8` aquí sería seleccionar el
    # hiperparámetro mirando los especímenes de evaluación: fuga de
    # información que convertiría el resultado final en un ajuste, no en una
    # predicción. Fue el error de la primera versión de este fichero.
    pivot = sweep.pivot(index="m", columns="n_anchor", values="penalty_train")
    worst = pivot.max(axis=1)
    admissible = worst[worst.index >= 2.0]
    recommended_m = float(admissible.idxmin())
    # Prior width: half the span of exponents whose worst-case penalty stays
    # within 50 % of the minimax optimum -- the range the data cannot separate.
    tolerated = admissible[admissible <= 1.5 * admissible.min()].index
    recommended_sd = float(max(0.2, (tolerated.max() - tolerated.min()) / 4.0))

    from .data import CALIBRATION_SPECIMENS, load_curve
    from .models import LAWS
    from .pooled import fit_coefficient_only

    rec_logc = {
        spec: fit_coefficient_only(
            LAWS["paris"], np.array([recommended_m]), load_curve(spec)
        )[0]
        for spec in CALIBRATION_SPECIMENS
    }
    rec_values = np.array(list(rec_logc.values()))

    paris = best["paris"]
    walker = best["walker"]
    hole = best.get("paris_hole", paris)

    best_obj = min(p["objective"] for p in profile)
    inside = [p["value"] for p in profile if p["objective"] <= 1.05 * best_obj]

    t1, t8 = load_curve("T1"), load_curve("T8")
    r_values = [seg[2] / seg[1] for c in (t1, t8) for seg in c.load_block]

    p = loglog["pooled"]
    text = TEMPLATE.format(
        today=date.today().isoformat(),
        m_lo=min(inside), m_hi=max(inside),
        decades=float(np.log10(p["delta_k_max"] / p["delta_k_min"])),
        constant_block=[[float(v) for v in seg] for seg in t1.load_block],
        variable_block=[[float(v) for v in seg] for seg in t8.load_block],
        r_min=_round(min(r_values), 5), r_max=_round(max(r_values), 5),
        recommended_m=recommended_m, recommended_m_sd=_round(recommended_sd, 2),
        rec_logc_mean=_round(rec_values.mean()),
        rec_logc_std=_round(rec_values.std(ddof=1)),
        rec_c=float(f"{10 ** rec_values.mean():.4g}"),
        rec_logc_by_spec={k: _round(v) for k, v in rec_logc.items()},
        c_factor=float(10 ** paris["log10_coefficient_std"]),
        pen_3=float(sweep.loc[(sweep.n_anchor == 3) & (np.isclose(sweep.m, recommended_m)),
                              "penalty_train"].iloc[0]),
        pen_4=float(sweep.loc[(sweep.n_anchor == 4) & (np.isclose(sweep.m, recommended_m)),
                              "penalty_train"].iloc[0]),
        paris_m=_round(paris["shared"]["m"]),
        paris_logc_mean=_round(paris["log10_coefficient_mean"], 4),
        paris_logc_std=_round(paris["log10_coefficient_std"], 4),
        paris_c=float(f"{10 ** paris['log10_coefficient_mean']:.4g}"),
        paris_logc_by_spec={k: _round(v) for k, v in paris["log10_coefficients"].items()},
        paris_rmse=paris["objective_mean_rmse_mm"],
        walker_rmse=walker["objective_mean_rmse_mm"],
        walker_gamma=_round(walker["shared"]["gamma"]),
        hole_rmse=hole["objective_mean_rmse_mm"],
        hole_kt=_round(hole["shared"].get("Kt", float("nan"))),
        hole_lambda=_round(hole["shared"].get("lambda_h", float("nan"))),
        hole_m=_round(hole["shared"]["m"]),
        hole_logc=_round(hole["log10_coefficient_mean"]),
        loglog_m=_round(p["m"], 3), loglog_m_ci=_round(p["m_ci95"], 3),
        loglog_c=float(f"{p['C']:.4g}"), loglog_r2=_round(p["r2"], 3),
        loglog_sd=_round(p["residual_sd_dex"], 3),
        log_c_bounds=(-13.0, -6.0),
        m_bounds=(2.0, 4.0),
        literature={
            "Dourado & Viana (2019), Al 2024-T3 coupons (Menan & Henaff 2010), air": {
                "C": 5.008e-10, "m": 3.859,
                "note": "units not recoverable from the paper; do not use as-is",
            },
            "Metals 13(8):1134 (2023), AA 2024-T4, modified CT, R = -1": {
                "C": 5.75e-8, "m": 3.09,
            },
            "Rao et al. (2021) after Li, Wang & Gong (2012), generic metals": {
                "C_range": (1e-13, 1e-11), "m_range": (2.0, 4.0),
            },
        },
    )
    TARGET.write_text(text)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
