"""Tensor dataset, augmentation and specimen-aware splits for Configuration 1.

With 87 labelled waveforms, every design choice here is about not wasting any
of them and about not letting the network memorise them:

* **Waveforms are cached in memory as tensors.** The whole corpus is ~3 MB, so
  the whole training loop runs off RAM and an epoch costs milliseconds.
* **Batches are whole specimens.** The physics residual is defined between
  consecutive cycles of the same specimen, and the coefficient head pools over
  a specimen, so a batch that splits a specimen would silently compute both on
  partial information. Sampling by specimen makes both terms exact.
* **Augmentation is physically conservative.** Trigger jitter (a shift of a few
  samples) and additive noise are things the acquisition chain genuinely does.
  Time warping or large shifts are not: the time of flight *is* the damage
  signal, and augmenting it away would destroy the very feature the encoder
  needs to find.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from physics_calibration.data import DEFAULT_ROOT, load_description
from . import signals as S

ALL_SPECIMENS = S.TRAINING_SPECIMENS + S.VALIDATION_SPECIMENS


def load_labels(root=DEFAULT_ROOT) -> dict[str, pd.DataFrame]:
    return {name: load_description(root, name) for name in ALL_SPECIMENS}


@dataclass
class SpecimenBatch:
    """Every labelled waveform of one specimen, ready for the model.

    ``a_prev`` and ``log_dn`` carry the **sequential context**: the crack length
    at the specimen's previous labelled cycle and the (log) cycle gap since it.
    They are what turns a per-waveform regression into a sequential estimator,
    and they are the single biggest lever available on this dataset — a
    the single biggest lever available on this dataset. The 2nd-place entry uses
    the same idea, listing ``prev_crack`` among its five optimal features.

    At training time ``a_prev`` is the *measured* previous crack (teacher
    forcing, with noise — see :func:`augment_previous`). At inference nothing of
    the sort is available for T7/T8, so the estimator is run **recursively** on
    its own previous output.
    """

    name: str
    x: torch.Tensor           # (n, 2, window)
    crack_mm: torch.Tensor    # (n,)
    cycles: torch.Tensor      # (n,)
    a_prev: torch.Tensor      # (n,) crack at the previous labelled cycle, 0 if none
    log_dn: torch.Tensor      # (n,) log10(1 + cycles since that measurement)
    dn: torch.Tensor          # (n,) cycles since that measurement (lineal, 0 si no hay)
    features: torch.Tensor    # (n, F) rasgos escalares clásicos
    norm_mm: float            # final measured crack, the challenge normaliser
    d_sigma_eq: float         # block-equivalent stress range, MPa
    pair_from: torch.Tensor   # indices into x, earlier cycle of each pair
    pair_to: torch.Tensor     # indices into x, later cycle of each pair
    pair_dn: torch.Tensor     # cycle gap of each pair

    def to(self, device):
        for f in ("x", "crack_mm", "cycles", "a_prev", "log_dn", "dn", "features",
                  "pair_from", "pair_to", "pair_dn"):
            setattr(self, f, getattr(self, f).to(device))
        return self


def _consecutive_pairs(cycles: np.ndarray, cracks: np.ndarray):
    """Index pairs of consecutive *distinct* cycles with a non-zero crack.

    Pairs are built on cycles, not on waveforms: the two repetitions of a
    measurement share a cycle, so their averaged estimate is what the physics
    residual should act on. Zero-crack cycles are excluded because the growth
    law describes propagation, and a zero label only means the crack was below
    the optical detection threshold.
    """
    order = np.argsort(cycles)
    unique_cycles = []
    for i in order:
        if cracks[i] <= 0:
            continue
        if not unique_cycles or cycles[i] != cycles[unique_cycles[-1][0]]:
            unique_cycles.append((i, [i]))
        else:
            unique_cycles[-1][1].append(i)
    reps = [(c, idxs) for c, idxs in unique_cycles]
    froms, tos, dns = [], [], []
    for (i0, _), (i1, _) in zip(reps[:-1], reps[1:]):
        froms.append(i0)
        tos.append(i1)
        dns.append(cycles[i1] - cycles[i0])
    return np.array(froms, int), np.array(tos, int), np.array(dns, float)


def build_specimen_batches(
    labels: dict[str, pd.DataFrame],
    m_exponent: float,
    specimens=ALL_SPECIMENS,
    root=None,
    max_cycle: dict[str, int] | None = None,
) -> dict[str, SpecimenBatch]:
    """Assemble one :class:`SpecimenBatch` per specimen.

    ``max_cycle`` optionally truncates a specimen to the cycles at or below a
    given value — used to reproduce the challenge condition in which the
    validation specimens stop having signals partway through.
    """
    from physics_calibration.data import load_curve
    from .physics import equivalent_stress_range

    samples = S.build_samples(labels, specimens, root, include_zero=True)
    out: dict[str, SpecimenBatch] = {}
    for name in specimens:
        rows = [s for s in samples if s.specimen == name]
        if max_cycle and name in max_cycle:
            rows = [s for s in rows if s.cycle <= max_cycle[name]]
        if not rows:
            continue
        x = np.stack([s.features(root) for s in rows])
        feats = np.stack([S.scalar_features(s.specimen, s.cycle, s.rep, root) for s in rows])
        cracks = np.array([s.crack_mm for s in rows], dtype=np.float32)
        cycles = np.array([s.cycle for s in rows], dtype=np.float64)
        curve = load_curve(name)

        # Contexto secuencial para el estimador recursivo.
        # El "anterior" es el último ciclo con grieta **detectada** (no nula).
        # Los ciclos de grieta cero no aportan contexto: la ley de crecimiento
        # describe propagación, y una etiqueta cero sólo dice que la grieta
        # estaba por debajo del umbral óptico. La inferencia aplica el mismo
        # criterio (ver ``train.evaluate_batch``), de modo que entrenamiento e
        # inferencia ven exactamente la misma definición de ΔN.
        crack_at = {float(c): float(a) for c, a in zip(curve.cycles, curve.crack_mm)}
        labelled = sorted(crack_at)
        a_prev, log_dn, dn = [], [], []
        for cyc in cycles:
            anteriores = [c for c in labelled if c < cyc]
            if anteriores:
                a_prev.append(crack_at[anteriores[-1]])
                dn.append(cyc - anteriores[-1])
                log_dn.append(np.log10(1.0 + cyc - anteriores[-1]))
            else:
                a_prev.append(0.0)
                dn.append(0.0)
                log_dn.append(0.0)
        froms, tos, dns = _consecutive_pairs(cycles, cracks)
        out[name] = SpecimenBatch(
            name=name,
            x=torch.from_numpy(x),
            crack_mm=torch.from_numpy(cracks),
            cycles=torch.from_numpy(cycles.astype(np.float32)),
            a_prev=torch.tensor(a_prev, dtype=torch.float32),
            log_dn=torch.tensor(log_dn, dtype=torch.float32),
            dn=torch.tensor(dn, dtype=torch.float32),
            features=torch.from_numpy(feats),
            norm_mm=float(curve.final_crack_mm),
            d_sigma_eq=equivalent_stress_range(curve.load_block, m_exponent),
            pair_from=torch.from_numpy(froms).long(),
            pair_to=torch.from_numpy(tos).long(),
            pair_dn=torch.from_numpy(dns.astype(np.float32)),
        )
    return out


def augment_previous(a_prev: torch.Tensor, generator: torch.Generator, noise: float = 0.4):
    """Perturba el contexto ``a_prev`` durante el entrenamiento.

    Mitiga el **sesgo de exposición**: en entrenamiento el estimador ve la
    grieta anterior *medida*, pero en inferencia sólo dispone de su propia
    estimación previa, que arrastra error. Entrenar con ``a_prev`` ruidoso lo
    obliga a ser robusto a esa deriva. En el control con Random Forest bajó el
    error LOSO recursivo de 0,955 a 0,870 mm.
    """
    ruido = noise * torch.randn(a_prev.shape, generator=generator, device=a_prev.device)
    return torch.clamp(a_prev + ruido, min=0.0)


def augment(x: torch.Tensor, generator: torch.Generator, shift: int = 10, noise: float = 0.05):
    """Trigger jitter + additive noise, both physically plausible.

    ``shift`` is +/- 10 samples = +/- 0.5 us at 20 MHz, an order of magnitude
    below the time-of-flight changes the crack produces, so it models
    acquisition jitter without erasing the damage signature.
    """
    n, _, length = x.shape
    offsets = torch.randint(-shift, shift + 1, (n, 1), generator=generator, device=x.device)
    # Vectorised circular shift: build the gather index once instead of looping
    # over the batch with torch.roll (which dominated the epoch time).
    idx = (torch.arange(length, device=x.device).unsqueeze(0) - offsets) % length
    out = torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))
    return out + noise * torch.randn(out.shape, generator=generator, device=x.device)
