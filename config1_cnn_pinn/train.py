"""Training and leave-one-specimen-out validation for Configuration 1.

Protocol
--------
Model selection is **leave-one-specimen-out (LOSO)** over the training
specimens, never leave-one-*cycle*-out. The two repetitions of a measurement
are near-duplicates and consecutive cycles of one specimen share its PZT
bonding, rivet fit-up and initiation site; a random split would leak all of
that and report an optimistic error that says nothing about a new specimen.
T7 and T8 are touched only at the very end.

T2 and T5 are kept as *training* data but never used as validation folds: two
non-zero measurements each cannot support a meaningful fold, and T5 is the
acknowledged outlier of the challenge.

The physics term is warmed up rather than applied from step zero: for the first
epochs the coefficient head has nothing to identify because the crack estimates
are still noise, and a strong residual at that point simply drags every
estimate onto one exponential.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from physics_calibration import priors
from . import losses
from .data import (SpecimenBatch, augment, augment_previous,
                   build_specimen_batches, load_labels)
from .model import CnnPinn
from .physics import residual

RESULTS = Path(__file__).resolve().parent / "results"

TRAIN_POOL = ("T1", "T2", "T3", "T4", "T5", "T6")
LOSO_FOLDS = ("T1", "T3", "T4", "T6")


@dataclass
class Config:
    """Everything that defines a run. Serialised with the results."""

    m_exponent: float = priors.RECOMMENDED_EXPONENT
    # ``None`` = re-identificar el coeficiente poblacional **al exponente en
    # uso** (ver ``physics.pooled_log_c``). Es el valor por defecto a propósito:
    # C y m se desplazan juntos por la cresta log-log, y fijar aquí una
    # constante de ``priors`` deja de ser correcta en cuanto ``m_exponent``
    # cambia. Puede darse un número explícito para reproducir una corrida vieja.
    log_c_prior: float | None = None
    log_c_range: float = 1.0
    # Gaussian prior pulling the identified coefficient back to the population
    # value. The pooled identification measured a between-specimen spread of
    # only 0.081 dex, so a coefficient head free to roam a full decade on five
    # training specimens does not identify physics, it fits noise — which is
    # exactly what the first sweep showed (|Dlog10 C| rose from 0.06 to 0.43 as
    # soon as the physics term was switched on). The prior width is set well
    # above the measured spread so a genuinely different specimen (T8, variable
    # amplitude) can still escape it.
    log_c_prior_sd: float = 0.3
    lambda_log_c_prior: float = 1.0
    data_loss: str = "asym"          # "asym" | "mse"
    # Peso por ciclo en la pérdida de datos: "challenge" (T(i) del certamen),
    # "flat" o "anchor". Ver ``losses.sample_weights``: copiar T(i) al entrenar
    # quita peso a las medidas tempranas, que son justamente las únicas que la
    # red estima bajo el protocolo y de las que cuelga toda la extrapolación.
    data_weight: str = "challenge"
    anchor_weight: float = 6.0
    n_anchor: int = 2
    lambda_physics: float = 1.0
    lambda_monotonic: float = 0.5
    # Fraction of the run spent before the physics residual switches on, and
    # the fraction over which it ramps up. Expressed as fractions rather than
    # epoch counts on purpose: an absolute warm-up silently disables the whole
    # PINN term whenever the run is shorter than it, which is exactly what
    # happened on the first sweep (the best "1D-CNN + PINN" configuration
    # turned out to have never activated its physics).
    physics_warmup_frac: float = 0.25
    physics_ramp_frac: float = 0.15
    epochs: int = 600
    # Parada temprana sobre un especimen retenido del propio conjunto de
    # entrenamiento. ``None`` desactiva la parada y agota ``epochs``.
    #
    # Por que un especimen y no una fraccion de ciclos: las dos repeticiones de
    # una medida son casi duplicados y los ciclos de un mismo especimen
    # comparten pegado del PZT, ajuste del remache y sitio de iniciacion, asi
    # que un corte por ciclos filtra todo eso y la parada se dispararia tarde.
    # ``train_ensemble`` rota el especimen retenido entre semillas, de modo que
    # el ensamblado en conjunto si ve todos los datos.
    #
    # Desactivada por defecto porque **medida, empeora**: con tres tandas de
    # semillas, 600 epocas sin parada dejan T7 en 6,15 y T8 en 6426, mientras
    # que con paciencia 50 y los pesos de la mejor epoca T7 sube a 31,78. La
    # perdida de validacion toca su minimo entre las epocas 6 y 43 y luego
    # sube, pero el modelo sigue mejorando en lo que el certamen puntua: la
    # senal que vigila la parada no esta alineada con el objetivo. Parar sin
    # restaurar (``restore_best=False``) evita el destrozo pero tampoco mejora
    # a agotar las 600 epocas. Se conserva la maquinaria porque el resultado
    # negativo es reportable y porque con mas especimenes cambiaria.
    patience: int | None = None
    min_delta: float = 0.0
    # Al parar, ¿devolver los pesos de la mejor epoca o los de la ultima?
    # Con cinco especimenes y un desplazamiento de dominio fuerte entre ellos,
    # la perdida de validacion toca su minimo muy pronto y despues sube aunque
    # el modelo siga mejorando en lo que el certamen puntua, asi que las dos
    # respuestas dan modelos muy distintos y conviene poder medir ambas.
    restore_best: bool = True
    lr: float = 3e-3
    weight_decay: float = 1e-3
    dropout: float = 0.3
    channels: tuple = (8, 16, 24, 32)
    latent: int = 32
    # Arquitectura: "cnn_pinn" (Configuración 1) o "cnn_attention_pinn"
    # (Configuración 3, con autoatención sobre la secuencia de ciclos).
    architecture: str = "cnn_pinn"
    n_heads: int = 2
    attn_dropout: float = 0.1
    # Cota de dominio sobre la grieta que la extrapolacion puede devolver, en
    # mm. ``None`` deja actuar solo el tope numerico de ``physics.propagate_mm``
    # (20 mm), que existe para que la perdida no desborde en entrenamiento y es
    # deliberadamente holgado. Para la entrega el criterio es el contrario, el
    # mayor valor que el fenomeno produce de verdad; ``evaluate.COTA_ENTREGA``
    # lo deriva de los especimenes de entrenamiento y ``cota_extrapolacion`` lo
    # calibra.
    cota_entrega_mm: float | None = None
    augment_shift: int = 10
    augment_noise: float = 0.05
    # Ruido sobre el contexto ``a_prev`` en entrenamiento (sesgo de exposición).
    prev_noise: float = 0.4
    seed: int = 0

    def __post_init__(self):
        if self.log_c_prior is None:
            from .physics import pooled_log_c
            object.__setattr__(self, "log_c_prior", pooled_log_c(self.m_exponent))


def build_model(cfg, n_features: int):
    """Fabrica el modelo indicado por ``cfg.architecture``."""
    comun = dict(log_c_prior=cfg.log_c_prior, log_c_range=cfg.log_c_range,
                 channels=cfg.channels, latent=cfg.latent, dropout=cfg.dropout,
                 n_features=n_features)
    if cfg.architecture == "cnn_pinn":
        return CnnPinn(**comun)
    if cfg.architecture == "cnn_attention_pinn":
        from config3_attention.model import CnnAttentionPinn
        return CnnAttentionPinn(n_heads=cfg.n_heads, attn_dropout=cfg.attn_dropout, **comun)
    raise ValueError(f"arquitectura desconocida: {cfg.architecture!r}")


def _log_c(model, z, specimen_index, n_specimens, cycles=None, log_dn=None):
    """Llama a la cabeza de coeficiente pasándole la secuencia si la admite."""
    from config3_attention.model import CnnAttentionPinn
    if isinstance(model, CnnAttentionPinn):
        return model.log_c(z, specimen_index, n_specimens, cycles, log_dn)
    return model.log_c(z, specimen_index, n_specimens)


def _forward(model, batches, indices, cfg, generator=None, training=True):
    """One optimisation step's worth of forward pass over whole specimens."""
    xs, prevs, dns, lins, fts, specimen_ids = [], [], [], [], [], []
    for slot, name in enumerate(indices):
        b = batches[name]
        x = augment(b.x, generator, cfg.augment_shift, cfg.augment_noise) if training else b.x
        prev = augment_previous(b.a_prev, generator, cfg.prev_noise) if training else b.a_prev
        xs.append(x)
        prevs.append(prev)
        dns.append(b.log_dn)
        lins.append(b.dn)
        fts.append(b.features)
        specimen_ids.append(torch.full((x.shape[0],), slot, dtype=torch.long))
    x = torch.cat(xs)
    a_prev = torch.cat(prevs)
    log_dn = torch.cat(dns)
    dn = torch.cat(lins)
    feats = torch.cat(fts)
    specimen_index = torch.cat(specimen_ids)

    z = model.encoder(x)
    crack = model.crack_mm(z, feats, a_prev, log_dn, dn)
    log_c = _log_c(model, z, specimen_index, len(indices), torch.cat([b.cycles for b in
                   (batches[n] for n in indices)]), log_dn)

    total = crack.new_zeros(())
    # Cada término se registra dos veces: el valor **crudo**, que mide el
    # desajuste, y su contribución **efectiva** al gradiente (``_eff``), es
    # decir ya multiplicado por su lambda y, en la física, por el peso del
    # warm-up. Sin esa distinción una curva del término físico aparenta estar
    # activa desde la época 0, cuando su peso todavía es exactamente 0 y no
    # toca el gradiente. ``data`` no necesita pareja: su coeficiente es 1.
    parts = {"data": 0.0, "physics": 0.0, "mono": 0.0, "prior": 0.0,
             "prior_eff": 0.0, "physics_eff": 0.0, "mono_eff": 0.0}

    if cfg.lambda_log_c_prior:
        l_prior = (((log_c - model.log_c_prior) / cfg.log_c_prior_sd) ** 2).mean()
        total = total + cfg.lambda_log_c_prior * l_prior
        parts["prior"] = float(l_prior.detach())
        parts["prior_eff"] = float(cfg.lambda_log_c_prior * l_prior.detach())

    offset = 0
    for slot, name in enumerate(indices):
        b = batches[name]
        n = b.x.shape[0]
        pred = crack[offset:offset + n]
        w = losses.sample_weights(b.crack_mm, b.cycles, b.norm_mm, cfg.data_weight,
                                  cfg.n_anchor, cfg.anchor_weight)
        total_data = losses.data_loss(pred, b.crack_mm, b.norm_mm, cfg.data_loss, w)
        total = total + total_data
        parts["data"] += float(total_data.detach())

        if len(b.pair_from):
            res = residual(
                pred[b.pair_from], pred[b.pair_to], b.pair_dn,
                torch.pow(10.0, log_c[slot]), cfg.m_exponent, b.d_sigma_eq,
            )
            l_phys = losses.physics_loss(res)
            total = total + cfg.lambda_physics * model.physics_weight * l_phys
            parts["physics"] += float(l_phys.detach())
            parts["physics_eff"] += float(
                cfg.lambda_physics * model.physics_weight * l_phys.detach())

        l_mono = losses.monotonicity_loss(pred, b.cycles, torch.full((n,), slot))
        total = total + cfg.lambda_monotonic * l_mono
        parts["mono"] += float(l_mono.detach())
        parts["mono_eff"] += float(cfg.lambda_monotonic * l_mono.detach())
        offset += n

    return total / len(indices), parts, crack, log_c


@torch.no_grad()
def evaluate_batch(model, batch: SpecimenBatch, cfg: Config):
    """Estimación **recursiva** por ciclo y ``log10 C`` identificado.

    Recursiva y no con *teacher forcing*: en T7/T8 no existe ninguna grieta
    medida que alimentar como contexto, así que el estimador debe consumir su
    propia salida anterior. Evaluar con la grieta real daría un número
    optimista que no se puede reproducir en el protocolo del certamen — en el
    protocolo del certamen. El control de ``control_rf.py`` se evalúa igual, de
    forma recursiva, para que la comparación sea de una sola variable.

    La salida se fuerza monótona (``max`` con la estimación previa): la grieta
    no puede decrecer, y arrastrar un retroceso al contexto del ciclo siguiente
    propagaría el error.
    """
    model.eval()
    z = model.encoder(batch.x)
    log_c = float(_log_c(model, z, torch.zeros(len(z), dtype=torch.long), 1,
                         batch.cycles, batch.log_dn)[0])

    cycles = batch.cycles.numpy()
    unique = np.unique(cycles)
    est, a_prev, cycle_prev = [], 0.0, None
    for cyc in unique:
        mask = torch.from_numpy(cycles == cyc)
        log_dn = 0.0 if cycle_prev is None else float(np.log10(1.0 + cyc - cycle_prev))
        n = int(mask.sum())
        dn_lin = 0.0 if cycle_prev is None else float(cyc - cycle_prev)
        pred = model.crack_mm(
            z[mask],
            batch.features[mask],
            torch.full((n,), a_prev, dtype=torch.float32),
            torch.full((n,), log_dn, dtype=torch.float32),
            torch.full((n,), dn_lin, dtype=torch.float32),
        )
        value = max(float(pred.mean()), a_prev)
        est.append(value)
        a_prev, cycle_prev = value, cyc

    true = np.array([float(batch.crack_mm[cycles == c][0]) for c in unique])
    return unique, np.array(est), true, log_c


@torch.no_grad()
def _val_data_loss(model, batches, val_names, cfg) -> float:
    """Perdida de datos sobre los especimenes retenidos, sin aumentado."""
    model.eval()
    _, parts, _, _ = _forward(model, batches, list(val_names), cfg, training=False)
    return parts["data"] / len(val_names)


def train_one(cfg: Config, batches, train_names, val_names=(), verbose=False):
    """Train a single model on ``train_names``. Returns the fitted model.

    Si se dan ``val_names`` y ``cfg.patience`` no es ``None``, se vigila la
    perdida de datos sobre esos especimenes al final de cada epoca y se detiene
    cuando pasan ``patience`` epocas sin mejorarla, devolviendo los pesos de la
    mejor epoca y no los de la ultima.
    """
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    model = build_model(cfg, n_features=batches[train_names[0]].features.shape[1])
    # Estandarización de rasgos con los especímenes de ENTRENAMIENTO de esta
    # ejecución: es lo que impide que estadísticos de T7/T8 entren en el modelo.
    todos = torch.cat([batches[n].features for n in train_names])
    model.feature_mean.copy_(todos.mean(dim=0))
    model.feature_std.copy_(todos.std(dim=0).clamp(min=1e-6))
    model.physics_weight = 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    history, history_terms = [], []
    vigilar = bool(val_names) and cfg.patience is not None
    mejor_val, mejor_epoca, mejor_estado = float("inf"), -1, None
    order = np.arange(len(train_names))
    rng = np.random.default_rng(cfg.seed + 2)
    for epoch in range(cfg.epochs):
        model.train()
        # Linear warm-up of the physics term (see module docstring).
        start = cfg.physics_warmup_frac * cfg.epochs
        ramp = max(1.0, cfg.physics_ramp_frac * cfg.epochs)
        model.physics_weight = float(np.clip((epoch - start) / ramp, 0.0, 1.0))
        lr_epoch = float(opt.param_groups[0]["lr"])
        # One optimisation step *per specimen*, in shuffled order, rather than
        # one full-batch step per epoch. With only five training specimens a
        # full-batch schedule spends the whole run on a few hundred gradient
        # steps and visibly underfits; stepping per specimen also keeps the
        # coefficient head pooling over exactly one specimen, which is what it
        # is defined to do.
        rng.shuffle(order)
        epoch_loss, epoch_parts = 0.0, {k: 0.0 for k in (
            "data", "physics", "mono", "prior",
            "prior_eff", "physics_eff", "mono_eff")}
        for j in order:
            opt.zero_grad()
            loss, parts, _, _ = _forward(
                model, batches, [train_names[j]], cfg, gen, training=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss)
            for k in epoch_parts:
                epoch_parts[k] += parts[k]
        sched.step()
        epoch_loss /= len(order)
        if verbose and epoch % 200 == 0:
            print(f"    epoch {epoch:4d}  loss {epoch_loss:.4f}  "
                  f"data {epoch_parts['data'] / len(order):.4f}  "
                  f"fis {epoch_parts['physics'] / len(order):.4f}")
        history.append(epoch_loss)
        # Historia por época descompuesta en los términos de la pérdida.
        # Es puramente observacional: no entra en el grafo ni en el
        # optimizador, de modo que registrarla no puede alterar ninguna cifra
        # ya publicada. ``physics_weight`` se guarda porque es lo que hace
        # visible en qué época entra la física al terminar el warm-up, que
        # §4.4.7 de la metodología explica pero ninguna figura muestra.
        val_loss = _val_data_loss(model, batches, val_names, cfg) if vigilar else float("nan")
        history_terms.append({
            "epoch": epoch,
            "total": epoch_loss,
            **{k: v / len(order) for k, v in epoch_parts.items()},
            "physics_weight": float(model.physics_weight),
            "lr": lr_epoch,
            "val_data": val_loss,
        })
        if vigilar:
            if val_loss < mejor_val - cfg.min_delta:
                mejor_val, mejor_epoca = val_loss, epoch
                mejor_estado = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - mejor_epoca >= cfg.patience:
                if verbose:
                    print(f"    parada temprana en la epoca {epoch} "
                          f"(mejor {mejor_epoca}, val {mejor_val:.4f})")
                break

    if vigilar and cfg.restore_best and mejor_estado is not None:
        model.load_state_dict(mejor_estado)
    model.history = history
    model.history_terms = history_terms
    model.val_specimens = tuple(val_names)
    model.stopped_epoch = len(history) - 1
    model.best_epoch = mejor_epoca if vigilar else len(history) - 1
    model.best_val = mejor_val if vigilar else float("nan")
    return model


def train_ensemble(cfg: Config, batches, train_names, n_seeds: int = 1, verbose=False):
    """``n_seeds`` independently initialised models.

    Averaging a handful of seeds is the single most reliable regulariser
    available at this sample size: individual runs on 60-70 waveforms land in
    visibly different minima, and the mean of their predictions is consistently
    better than any of them. It costs nothing at inference here (the encoder is
    ~10 k parameters) and it does not touch the validation specimens.
    """
    models = []
    rotar = cfg.patience is not None and len(train_names) >= 3
    for k in range(n_seeds):
        sub = Config(**{**asdict(cfg), "seed": cfg.seed + 100 * k})
        if rotar:
            # La semilla k retiene un especimen distinto: ninguno queda fuera
            # del ensamblado y la parada no depende de un solo espécimen.
            val = [train_names[k % len(train_names)]]
            entrena = [n for n in train_names if n not in val]
        else:
            val, entrena = [], list(train_names)
        models.append(train_one(sub, batches, entrena, val, verbose=verbose and k == 0))
    return models


@torch.no_grad()
def evaluate_ensemble(models, batch: SpecimenBatch, cfg: Config):
    """Per-cycle estimate and log10 C averaged over an ensemble."""
    ests, log_cs = [], []
    for model in models:
        cycles, est, true, log_c = evaluate_batch(model, batch, cfg)
        ests.append(est)
        log_cs.append(log_c)
    return cycles, np.mean(ests, axis=0), true, float(np.mean(log_cs))


def loso_cross_validation(cfg: Config, batches, folds=LOSO_FOLDS, n_seeds: int = 1, verbose=False):
    """Leave-one-specimen-out over ``folds``, training on the rest of the pool."""
    rows, curves = [], {}
    for held_out in folds:
        train_names = [n for n in TRAIN_POOL if n != held_out and n in batches]
        models = train_ensemble(cfg, batches, train_names, n_seeds, verbose=verbose)
        cycles, est, true, log_c = evaluate_ensemble(models, batches[held_out], cfg)
        nz = true > 0
        rows.append({
            "fold": held_out,
            "n_cycles": int(len(cycles)),
            "rmse_mm": float(np.sqrt(np.mean((est - true) ** 2))),
            "rmse_nonzero_mm": float(np.sqrt(np.mean((est[nz] - true[nz]) ** 2))) if nz.any() else np.nan,
            "mae_mm": float(np.mean(np.abs(est - true))),
            "bias_mm": float(np.mean(est - true)),
            "log10_C_predicho": log_c,
        })
        curves[held_out] = {"cycles": cycles.tolist(), "est": est.tolist(), "true": true.tolist()}
        if verbose:
            print(f"  fold {held_out}: RMSE {rows[-1]['rmse_mm']:.3f} mm  "
                  f"log10C {log_c:.3f}")
    return rows, curves


def main() -> None:
    import json

    RESULTS.mkdir(exist_ok=True)
    cfg = Config()
    labels = load_labels()
    batches = build_specimen_batches(labels, cfg.m_exponent)

    print(f"Configuración 1 — 1D-CNN + PINN")
    print(f"  m = {cfg.m_exponent}, log10 C prior = {cfg.log_c_prior}")
    model = build_model(cfg, n_features=13)
    print(f"  parámetros entrenables: {model.n_parameters():,}")
    print(f"  formas de onda etiquetadas: {sum(b.x.shape[0] for b in batches.values())}")

    print("\nValidación cruzada dejando un espécimen fuera (LOSO):")
    rows, curves = loso_cross_validation(cfg, batches, verbose=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    print(f"\n  RMSE media entre folds: {df.rmse_mm.mean():.4f} mm")

    df.to_csv(RESULTS / "loso_metrics.csv", index=False)
    (RESULTS / "loso_curves.json").write_text(json.dumps(curves, indent=2))
    (RESULTS / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))


if __name__ == "__main__":
    main()
