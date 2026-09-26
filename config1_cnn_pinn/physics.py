"""Differentiable Paris-Erdogan physics for the PINN head (componente b).

Scope note — what belongs to Configuration 1 and what does not
-------------------------------------------------------------
Configuration 1 is 1D-CNN (a) + PINN (b); the explicit multi-step numerical
integrator is component (c) and belongs to Configuration 2. The physics here is
therefore the **closed-form analytic solution** of the Paris law under a
constant (or block-equivalent) stress range, used for two things only:

1. a *one-step-ahead consistency residual* between consecutive **observed**
   cycles, which is what makes the loss physics-informed and what makes the
   specimen coefficient identifiable at all;
2. the analytic extrapolation that turns the estimator into a full challenge
   submission for the cycles after the Lamb-wave signals stop.

Configuration 2 will replace both by RK4 integration over the true load block,
so the ablation isolates exactly what the numerical integrator buys: correct
handling of a variable-amplitude spectrum and numerical stability over long
horizons.

Units follow ``physics_calibration``: da/dN in m/cycle, dK in MPa*sqrt(m),
crack length in metres inside the equations and millimetres at the interface.

Closed form
-----------
With dK = Y*dsigma*sqrt(pi*a) and Y = 1, da/dN = C*dK^m integrates to

    p = 1 - m/2,     k = C * (dsigma*sqrt(pi))^m
    a(N0 + dN) = [ a0^p + p*k*dN ]^(1/p)                      (m != 2)
    a(N0 + dN) = a0 * exp(k*dN)                               (m == 2)

The identified exponent is m = 2.00 exactly (see
``physics_calibration/priors.py``), which lands on the second branch. That is
convenient rather than awkward: the exponential form has no singularity, is
unconditionally positive and monotonically increasing, and so satisfies the
challenge's monotonicity factor M(i) *by construction* — one of the two
structural constraints the state of the art leaves unsolved.
"""
from __future__ import annotations

import numpy as np
import torch

MM_PER_M = 1000.0
_M_SINGULAR_TOL = 1e-3


def equivalent_stress_range(load_block, m: float) -> float:
    """Block-equivalent stress range, MPa.

    For a repeating spectrum, the cycle-averaged growth rate at a frozen crack
    length is C*(sqrt(pi*a))^m * mean_i(dsigma_i^m), so a single equivalent
    range dsigma_eq = (mean_i dsigma_i^m)^(1/m) reproduces it exactly. This is
    the analytic counterpart of Kong et al. (2020) Eq. 11.
    """
    block = np.asarray(load_block, dtype=float)
    n, s_max, s_min = block[:, 0], block[:, 1], block[:, 2]
    d_sigma = s_max - s_min
    return float((np.sum(n * d_sigma ** m) / np.sum(n)) ** (1.0 / m))


def pooled_log_c(m: float, specimens=None) -> float:
    """log10 *C* poblacional **re-identificado al exponente en uso**.

    Por qué no basta con leer una constante de ``priors``
    ----------------------------------------------------
    *C* y *m* no son independientes: se desplazan juntos a lo largo de la cresta
    log-log del ajuste, así que un coeficiente citado a un exponente distinto
    del que se está usando es sencillamente incorrecto. Con estos especímenes la
    dependencia es fuerte —el coeficiente agrupado pasa de −8,426 en *m* = 2,00
    a −8,667 en *m* = 2,25, casi un cuarto de década— de modo que reutilizar
    ``priors.RECOMMENDED_LOG10_C`` (identificado a 2,25) con *m* = 2,00 mete un
    sesgo de 0,24 dex en el *prior* de la cabeza física, del mismo orden que el
    error que la cabeza tiene que corregir.

    Se calcula sobre ``CALIBRATION_SPECIMENS`` = T1/T3/T4/T6, los mismos con los
    que se identificó todo lo demás. **No entra ningún dato de T7 ni de T8.**
    """
    from physics_calibration.data import CALIBRATION_SPECIMENS, load_curve
    nombres = specimens if specimens is not None else CALIBRATION_SPECIMENS
    valores = []
    for s in nombres:
        curve = load_curve(s)
        valores.append(fit_log_c(curve.cycles, curve.crack_mm, m,
                                 equivalent_stress_range(curve.load_block, m)))
    return float(np.mean(valores))


def growth_constant(d_sigma_mpa: float, C, m: float):
    """k = C * (dsigma*sqrt(pi))^m, the only way C and the load enter."""
    return C * (d_sigma_mpa * np.sqrt(np.pi)) ** m


def propagate_mm(a0_mm, delta_cycles, C, m: float, d_sigma_mpa: float, a_max_mm: float = 20.0):
    """Analytic Paris propagation. Torch-differentiable w.r.t. ``a0_mm`` and ``C``.

    ``a_max_mm`` clamps a diverging trajectory so the loss stays finite; 20 mm
    is nearly three times the longest crack ever measured in this dataset
    (7.46 mm on T1), so it never touches a physically meaningful curve.
    """
    is_torch = torch.is_tensor(a0_mm) or torch.is_tensor(C)
    exp_fn, pow_fn, clamp = (torch.exp, torch.pow, torch.clamp) if is_torch else (
        np.exp, np.power, np.clip)

    a0_m = a0_mm / MM_PER_M
    k = growth_constant(d_sigma_mpa, C, m)
    if abs(m - 2.0) < _M_SINGULAR_TOL:
        a_m = a0_m * exp_fn(k * delta_cycles)
    else:
        p = 1.0 - m / 2.0
        base = pow_fn(a0_m, p) + p * k * delta_cycles
        floor = 1e-9
        base = clamp(base, floor, None) if is_torch else np.maximum(base, floor)
        a_m = pow_fn(base, 1.0 / p)
    a_mm = a_m * MM_PER_M
    return clamp(a_mm, 0.0, a_max_mm) if is_torch else np.clip(a_mm, 0.0, a_max_mm)


def reconciled_anchor_mm(cycles, estimates_mm, C, m: float, d_sigma_mpa: float,
                         n_grid: int = 20001, max_mm: float = 20.0) -> float:
    """Ancla en el último ciclo observado, **coherente con la propia ley**.

    El problema que resuelve
    ------------------------
    La entrega toma como ancla la última estimación suelta de la red y propaga
    desde ahí. Pero la red produce *varias* estimaciones en la ventana con
    señal, y nada obliga a que sean coherentes entre sí bajo el coeficiente que
    su propia cabeza física identificó. En T8 no lo son: estima un avance de
    0,99 mm en 2048 ciclos cuando su log10 C implica 0,56 mm, un factor 1,8 de
    incoherencia interna. Anclar en la última estimación hereda ese error
    entero y lo amplifica durante 24 000 ciclos.

    El residuo de Paris ya penaliza esa incoherencia **durante el
    entrenamiento**, pero en inferencia no actúa: es un término de pérdida, no
    una restricción. Esta función lo convierte en restricción, que es lo que un
    modelo físicamente informado debería hacer en las dos fases.

    Cómo
    ----
    Mínimos cuadrados de una sola incógnita: el valor ``a_corte`` en el último
    ciclo observado que, propagado hacia atrás y hacia delante con la ley, mejor
    reproduce **todas** las estimaciones de la ventana observada. Se resuelve
    por rejilla porque el problema es unidimensional y acotado.

    Veredicto: MEDIDA Y RECHAZADA
    -----------------------------
    La idea es correcta pero **no funciona en este dataset**, y el criterio
    libre de fuga la rechaza sin ambigüedad: sobre T1/T3/T4/T6 el peor fold pasa
    de 48,81 a 86,23 en nominal y de 381,20 a 528,20 sobre la banda de ±0,3 dex.

    La razón es la asimetría de la métrica. Reconciliar *mueve* el ancla en la
    dirección que dicte el ajuste, y en T4 la mueve hacia abajo (1,93 → 1,71 mm,
    con verdad 2,17): mejora la coherencia interna y empeora la penalización,
    porque subestimar cuesta 2,5 veces más que sobreestimar. Coherencia física
    y coste operacional no apuntan al mismo sitio.

    Se conserva la función porque el resultado negativo es reportable y porque
    la coherencia interna sí es la magnitud correcta a vigilar; **no está
    cableada en la entrega**.

    No usa ninguna verdad de campo: sólo las salidas de la red y el coeficiente
    que ella misma identificó.
    """
    cycles = np.asarray(cycles, dtype=float)
    est = np.asarray(estimates_mm, dtype=float)
    if len(est) < 2:
        return float(est[-1])
    n_cut = cycles[-1]
    rejilla = np.linspace(1e-3, max(max_mm, float(est.max()) * 2.0), n_grid)[:, None]
    pred = propagate_mm(rejilla, (cycles - n_cut)[None, :], C, m, d_sigma_mpa,
                        a_max_mm=max_mm)
    sse = np.sum((pred - est[None, :]) ** 2, axis=1)
    return float(rejilla[int(np.argmin(sse)), 0])


def residual(a_from_mm, a_to_mm, delta_cycles, C, m: float, d_sigma_mpa: float):
    """One-step Paris consistency residual between two consecutive estimates.

    ``a_from_mm`` and ``a_to_mm`` are the network's own crack estimates at two
    consecutive *observed* cycles. The residual is the mismatch, in mm, between
    the estimate at the later cycle and what the growth law predicts from the
    earlier one with the specimen's identified coefficient. Detaching the
    anchor is deliberate: without it the cheapest way to zero the residual is
    to collapse both estimates onto the same value, which would fight the data
    term instead of regularising it.
    """
    anchor = a_from_mm.detach() if torch.is_tensor(a_from_mm) else a_from_mm
    predicted = propagate_mm(anchor, delta_cycles, C, m, d_sigma_mpa)
    return a_to_mm - predicted


def fit_log_c(cycles, cracks_mm, m: float, d_sigma_mpa: float,
              bounds=(-13.0, -6.0), grid: int = 4001) -> float:
    """Least-squares log10 C for one specimen given (cycle, crack) anchors.

    Used at evaluation time to compare the coefficient the network *predicts*
    from the waveforms against the one the crack measurements *imply* — the
    check that the PINN head learned physics rather than a constant.
    """
    cycles = np.asarray(cycles, dtype=float)
    cracks = np.asarray(cracks_mm, dtype=float)
    log_c = np.linspace(*bounds, grid)
    pred = propagate_mm(cracks[0], (cycles - cycles[0])[None, :],
                        (10.0 ** log_c)[:, None], m, d_sigma_mpa)
    sse = np.sum((pred - cracks[None, :]) ** 2, axis=1)
    return float(log_c[int(np.argmin(sse))])
