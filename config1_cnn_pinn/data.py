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
    name: str
    x: torch.Tensor
    crack_mm: torch.Tensor
    cycles: torch.Tensor
    # Crack at the previous non-zero label, 0 if none.
    a_prev: torch.Tensor
    # log10(1 + cycles since that label).
    log_dn: torch.Tensor
    dn: torch.Tensor
    features: torch.Tensor
    norm_mm: float
    # MPa
    d_sigma_eq: float
    pair_from: torch.Tensor
    pair_to: torch.Tensor
    pair_dn: torch.Tensor

    def to(self, device):
        for f in ("x", "crack_mm", "cycles", "a_prev", "log_dn", "dn", "features",
                  "pair_from", "pair_to", "pair_dn"):
            setattr(self, f, getattr(self, f).to(device))
        return self


def _consecutive_pairs(cycles: np.ndarray, cracks: np.ndarray):
    # Pairs of distinct cycles: both repetitions of a measurement share one.
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

        # Context is the last non-zero label: a zero label only means the crack was below
        # the optical threshold. Inference uses the same definition.
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
    # Exposure bias: at inference a_prev is the model's own previous estimate.
    ruido = noise * torch.randn(a_prev.shape, generator=generator, device=a_prev.device)
    return torch.clamp(a_prev + ruido, min=0.0)


def augment(x: torch.Tensor, generator: torch.Generator, shift: int = 10, noise: float = 0.05):
    # +/-10 samples = +/-0.5 us, an order of magnitude below the time-of-flight
    # change the crack produces.
    n, _, length = x.shape
    offsets = torch.randint(-shift, shift + 1, (n, 1), generator=generator, device=x.device)
    # A single gather index instead of torch.roll per sample, which dominated the epoch time.
    idx = (torch.arange(length, device=x.device).unsqueeze(0) - offsets) % length
    out = torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))
    return out + noise * torch.randn(out.shape, generator=generator, device=x.device)
