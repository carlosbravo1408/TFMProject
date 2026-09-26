"""Loss terms for Configuration 1.

The state of the art's Research Gap 1 asks, in so many words, whether making
the *training* loss asymmetric — instead of correcting for the asymmetry after
the fact — reduces the bias of the identified fracture parameters and the final
penalty. Both losses are therefore implemented and selectable, so the question
is answered by an experiment rather than asserted.

``mse``
    Plain mean squared error in mm. What Youn et al. (2020) and Rao et al.
    (2021) optimised, and the symmetric control of the hypothesis test.

``asym``
    A differentiable surrogate of the official penalty
    S(i) = T(i) * A(i) * M(i), evaluated on crack lengths normalised by the
    specimen's own final measured crack, exactly as the challenge does:

        T(i) = 2 + 10*x_i                       (weights the end of life)
        A(i) = exp(|dx|/0.5) - 1  if over-estimating
               exp(|dx|/0.2) - 1  if under-estimating

    The exponent is clamped before exponentiating. That is not cosmetic: early
    in training the network is wrong by several normalised units, exp(|dx|/0.2)
    overflows, and the run dies on the first batch. The clamp turns the tail
    into a linear region so a badly wrong prediction produces a large but
    finite, still-informative gradient.

    M(i) is not included here. It depends on the *ordering* of predictions
    within a specimen, which the physics residual already enforces far more
    directly, and squeezing it into a per-sample loss would double-count.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_OVER, _UNDER = 0.5, 0.2
_MAX_EXPONENT = 6.0


def asymmetric_penalty_loss(pred_mm, true_mm, norm_mm, weight=None):
    """Surrogate diferenciable de T(i)*A(i) sobre grietas normalizadas.

    ``weight`` sustituye al factor temporal T(i) = 2 + 10*x. Por defecto se usa
    T(i), que es lo que hace el certamen **al puntuar**; ver
    :func:`sample_weights` para por qué al *entrenar* no es lo correcto.
    """
    x_true = true_mm / norm_mm
    x_hat = pred_mm / norm_mm
    diff = x_hat - x_true
    scale = torch.where(diff >= 0, torch.full_like(diff, _OVER), torch.full_like(diff, _UNDER))
    a = torch.exp(torch.clamp(diff.abs() / scale, max=_MAX_EXPONENT)) - 1.0
    w = (2.0 + 10.0 * x_true) if weight is None else weight
    return (w * a).mean()


def data_loss(pred_mm, true_mm, norm_mm, kind: str = "asym", weight=None):
    if kind == "mse":
        if weight is None:
            return F.mse_loss(pred_mm, true_mm)
        return (weight * (pred_mm - true_mm) ** 2).mean()
    if kind == "asym":
        return asymmetric_penalty_loss(pred_mm, true_mm, norm_mm, weight)
    raise ValueError(f"unknown data loss {kind!r}")


def sample_weights(crack_mm, cycles, norm_mm, kind: str = "challenge",
                   n_anchor: int = 2, anchor_weight: float = 6.0):
    """Peso por ciclo observado en la pérdida de datos.

    Por qué esto no es un detalle
    -----------------------------
    El factor temporal del certamen, T(i) = 2 + 10·x, pesa **el triple** una
    medida de fin de vida que una del principio. Como criterio de *puntuación*
    es correcto: equivocarse cerca de la rotura es más caro.

    Como criterio de *entrenamiento* es justo lo contrario de lo que hace falta,
    y el diagnóstico LOSO lo enseña con números. Bajo el protocolo del certamen
    la red sólo estima los ciclos que conservan señal —los **primeros** de la
    vida del espécimen— y todos los ciclos tardíos salen de extrapolar desde el
    último de ellos. Copiar T(i) en la pérdida quita peso precisamente a las
    medidas de las que la red es responsable y se lo da a ciclos que en
    inferencia no verá nunca. El resultado medido sobre folds de entrenamiento
    es un ancla con un error mediano del 28 % —hasta −42 % en T6— que arrastra
    toda la mitad extrapolada.

    Las tres opciones que el protocolo libre de fuga compara:

    ``challenge``  w = 2 + 10·x. La definición anterior; se conserva como
                   control, porque la hipótesis es que perjudica.
    ``flat``       w = 1. Todas las medidas pesan igual.
    ``anchor``     w = 1, salvo las ``n_anchor`` primeras medidas no nulas —las
                   que el protocolo convierte en anclas— que pesan
                   ``anchor_weight``. Es el peso proporcional a la *influencia*
                   de cada estimación sobre la entrega final, que para el ancla
                   es total.
    """
    x_true = crack_mm / norm_mm
    if kind == "challenge":
        return 2.0 + 10.0 * x_true
    if kind not in ("flat", "anchor"):
        raise ValueError(f"unknown weighting {kind!r}")
    w = torch.ones_like(x_true)
    if kind == "anchor":
        nz = crack_mm > 0
        if bool(nz.any()):
            anclas = sorted({float(c) for c in cycles[nz]})[:n_anchor]
            for c in anclas:
                w = torch.where(cycles == c, torch.full_like(w, anchor_weight), w)
    return w


def physics_loss(residual_mm):
    """Huber on the Paris one-step residual.

    Huber rather than squared error because the optical crack labels are
    coarse: several specimens show the measured growth rate jumping
    non-monotonically by factors of 3-5 between consecutive intervals, and a
    squared residual would let those few intervals dictate the coefficient.
    """
    if residual_mm.numel() == 0:
        return residual_mm.new_zeros(())
    return F.huber_loss(residual_mm, torch.zeros_like(residual_mm), delta=0.5)


def monotonicity_loss(pred_mm, order_index, specimen_index):
    """Hinge on any decrease between consecutive cycles of the same specimen.

    Redundant with the physics residual once the coefficient head is trained —
    the analytic Paris solution is monotone by construction — but it acts from
    the first epoch, before the physics term has anything sensible to say, and
    it costs nothing.
    """
    if pred_mm.numel() < 2:
        return pred_mm.new_zeros(())
    order = torch.argsort(specimen_index * 1_000_000 + order_index)
    p = pred_mm[order]
    same = specimen_index[order][1:] == specimen_index[order][:-1]
    if not bool(same.any()):
        return pred_mm.new_zeros(())
    drop = F.relu(p[:-1][same] - p[1:][same])
    return (drop ** 2).mean()
