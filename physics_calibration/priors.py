"""Fracture-mechanics hyper-parameters for the PHM 2019 2024-T3 lap joints.

GENERATED FILE -- do not edit by hand. Produced by
``python -m physics_calibration.make_priors`` on 2026-09-16 from the results of
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

* The exponent is only weakly determined by the fit. Every m from 1.0 to
  7.0 reproduces the training curves within 5 % of the optimum, because
  dK spans just 0.31 decades. It is *prognosis*, not fitting, that pins
  it: the official penalty on blindly predicted points is minimised around
  m = 2.0-2.5 and explodes past m ~ 3.
* The coefficient is tightly determined once the specimen is fixed: the
  per-specimen spread is only 0.085 dex (a factor of
  1.22), so log10 C is exactly the kind of scalar a 1D-CNN plus
  attention encoder can regress from the Lamb-wave history.

So: keep m near ``PARIS_EXPONENT`` under a weak prior, and let the network
predict ``log10 C`` per specimen.
"""
from __future__ import annotations

# --- Load spectrum, decoded from the released loading-profile CSVs ---------
# Constant amplitude (T1-T7): one sine cycle, Smax/Smin in MPa.
CONSTANT_LOAD_BLOCK = [[1.0, 100.21, 4.77]]
# Variable amplitude (T8): one repeating block of 1000 cycles,
# (n_cycles, Smax, Smin).
VARIABLE_LOAD_BLOCK = [[500.0, 90.0, 4.77], [500.0, 100.21, 4.77]]

# Stress ratio present in the whole dataset. R varies from 0.0476 to
# 0.0530 -- a single point for all practical purposes, which is why
# Walker's gamma is NOT identifiable here (see WALKER_GAMMA below).
STRESS_RATIO_RANGE = (0.0476, 0.053)

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
# independent estimates agree on it: the pooled fit (2.06), the
# model-free log-log regression (2.57 +/- 1.30) and the
# prognosis sweep (optimum 1.75-2.5 depending on anchor budget).
RECOMMENDED_EXPONENT = 2.25
RECOMMENDED_EXPONENT_PRIOR_SD = 0.2
# Coefficient re-identified *at* RECOMMENDED_EXPONENT (not at PARIS_EXPONENT:
# C and m slide along the log-log ridge together, so a coefficient quoted at a
# different exponent is simply wrong). Per specimen, and pooled mean/spread.
RECOMMENDED_LOG10_C = -8.6675
RECOMMENDED_LOG10_C_STD = 0.0932
RECOMMENDED_C = 2.15e-09
RECOMMENDED_LOG10_C_BY_SPECIMEN = {'T1': -8.72, 'T3': -8.69, 'T4': -8.73, 'T6': -8.53}

PARIS_EXPONENT = 2.0632
PARIS_LOG10_C = -8.4872
PARIS_LOG10_C_STD = 0.0849
PARIS_C = 3.257e-09
PARIS_LOG10_C_BY_SPECIMEN = {'T1': -8.5346, 'T3': -8.4971, 'T4': -8.5525, 'T6': -8.3647}

# Model-free cross-check: ordinary least squares of log10(da/dN) on log10(dK)
# over finite differences pooled across all specimens. Independent of every
# modelling choice above, and the source of the confidence interval on m.
LOGLOG_EXPONENT = 2.566
LOGLOG_EXPONENT_CI95 = 1.302
LOGLOG_C = 8.729e-10
LOGLOG_R2 = 0.332
LOGLOG_RESIDUAL_SD_DEX = 0.298

# Search bounds for the physics module. log10 C is wide because C absorbs the
# geometry factor; m is bounded to the classical metallic window so the PINN
# cannot drift to a value that fits the data but is not fracture mechanics.
LOG10_C_BOUNDS = (-13.0, -6.0)
EXPONENT_BOUNDS = (2.0, 4.0)

# Minimum number of crack estimates the physics module needs before its
# multi-step extrapolation stops diverging. With two anchors -- all the Lamb
# wave signals of T7/T8 actually provide -- every law tested runs away on T8;
# with three the same law scores 117.4 on T7+T8's blindly predicted
# points, and with four, 119.3. This is the handover point the hybrid
# has to respect: the CNN must supply reliable crack estimates for at least
# this many measurement cycles before the integrator takes over alone.
MIN_ANCHOR_POINTS = 3

# --- Walker: da/dN = C0 * (Kmax * (1-R)^gamma)^m --------------------------
# gamma is unidentifiable from this dataset: every specimen is tested at
# essentially the same stress ratio, so (1-R)^gamma is a constant that the
# coefficient absorbs exactly. The pooled fit confirms it -- Walker reaches
# the same objective as Paris (0.4278 vs 0.4278 mm) at an
# arbitrary gamma. Use the value below only if the PINN is to be transferred
# to a spectrum with a different R; on PHM 2019 alone, prefer Paris.
WALKER_GAMMA = 0.4004
WALKER_GAMMA_IDENTIFIABLE = False

# --- Geometry -------------------------------------------------------------
# The dataset ReadMe states the crack initiates at a countersunk hole in the
# first rivet row. Y therefore starts near the hole's stress concentration and
# decays to the plain-plate value:  Y(a) = 1 + (Kt - 1) * exp(-a / lambda_h),
# with a and lambda_h in metres and millimetres respectively.
# Fitting this improves the pooled in-sample RMSE from 0.4278 to
# 0.3740 mm AND brings the identified coefficient into the published
# metallic range -- but it extrapolates far less stably (see the report).
HOLE_KT = 8.0
HOLE_DECAY_MM = 0.4688
HOLE_EXPONENT = 4.4846
HOLE_LOG10_C = -10.9441

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
LITERATURE_REFERENCE = {'Dourado & Viana (2019), Al 2024-T3 coupons (Menan & Henaff 2010), air': {'C': 5.008e-10, 'm': 3.859, 'note': 'units not recoverable from the paper; do not use as-is'}, 'Metals 13(8):1134 (2023), AA 2024-T4, modified CT, R = -1': {'C': 5.75e-08, 'm': 3.09}, 'Rao et al. (2021) after Li, Wang & Gong (2012), generic metals': {'C_range': (1e-13, 1e-11), 'm_range': (2.0, 4.0)}}
