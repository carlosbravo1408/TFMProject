"""1D-CNN encoder (componente a) with the PINN parameter head (componente b).

Capacity is set by the data, not by ambition: 87 labelled waveforms exist in
the entire release. The encoder is therefore deliberately small (~10 k
parameters), strided rather than pooled so it stays cheap, and regularised with
batch norm, dropout and weight decay. A ResNet-scale 1D-CNN would memorise this
dataset in a handful of epochs and generalise to nothing.

Two heads sit on the shared latent:

``crack head`` (**híbrida**)
    Recibe el latente convolucional **y** un puñado de rasgos escalares
    clásicos (energía diferencial, retardo de fase, correlación…). Con 87 ondas
    etiquetadas el codificador no llega solo: un control con Random Forest
    sobre esos mismos rasgos alcanza 0,850 mm LOSO frente a 1,008 mm de la CNN
    sola. En vez de elegir entre representación aprendida y representación
    diseñada, la cabeza usa las dos.

    Crack length in mm, parameterised **incrementally** as
    ``a(k) = a(k-1) + softplus(...)`` from the waveform latent plus the
    sequential context (previous crack, cycle gap). Monotone increasing by
    construction, so the challenge's M(i) factor holds without post-processing,
    and the network only has to predict *how much it grew* rather than
    rediscover the absolute crack at every cycle.

``coefficient head`` (the PINN part)
    One ``log10 C`` per *specimen*, not per waveform: C is a property of the
    specimen (rivet fit-up, initiation site, local geometry), which is exactly
    what the pooled identification found — the exponent is shared across
    specimens while the coefficient carries all the specimen-to-specimen
    scatter, with a spread of only 0.081 dex. The head therefore reads a
    mean-pooled latent over all of a specimen's waveforms and emits a single
    scalar, parameterised as

        log10 C = prior_mean + range * tanh(u)

    so it is bounded by construction and starts at the identified population
    prior. This is what makes the module physics-*informed* rather than
    physics-*flavoured*: the network has to explain the waveforms with a
    fracture-mechanics coefficient that also has to reproduce the observed
    growth between cycles.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.norm = nn.BatchNorm1d(c_out)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


class CrackEncoder(nn.Module):
    """Shared 1D convolutional trunk over the two-channel waveform."""

    def __init__(self, channels=(8, 16, 24, 32), latent: int = 32, dropout: float = 0.3):
        super().__init__()
        kernels = (15, 9, 7, 5)
        strides = (4, 4, 4, 2)
        blocks, c_in = [], 2
        for c_out, k, s in zip(channels, kernels, strides):
            blocks.append(ConvBlock(c_in, c_out, k, s))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        # Average and max pooling carry complementary information here: mean
        # over the trace tracks the overall attenuation, max tracks the
        # strongest scattered echo.
        self.project = nn.Sequential(
            nn.Linear(2 * c_in, latent), nn.GELU(), nn.Dropout(dropout)
        )
        self.latent_dim = latent

    def forward(self, x):
        h = self.blocks(x)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.project(pooled)


class CnnPinn(nn.Module):
    """Configuration 1: 1D-CNN estimator + Paris coefficient head."""

    def __init__(
        self,
        log_c_prior: float,
        log_c_range: float = 1.0,
        channels=(8, 16, 24, 32),
        latent: int = 32,
        dropout: float = 0.3,
        n_features: int = 13,
    ):
        super().__init__()
        self.encoder = CrackEncoder(channels, latent, dropout)
        self.n_features = n_features
        # Estandarización de los rasgos escalares. Se rellena en el
        # entrenamiento **con los especímenes de entrenamiento de esa ejecución**
        # (ver ``train.fit_feature_scaler``): calcularla sobre el conjunto
        # completo mezclaría estadísticos de T7/T8 en el modelo.
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_std", torch.ones(n_features))
        # La cabeza de grieta recibe el latente de la onda MÁS el contexto
        # secuencial (grieta previa y hueco de ciclos). Es el mayor salto de
        # precisión disponible en este dataset: un control con Random Forest
        # es el mayor palanca disponible en este dataset. La cabeza de coeficiente,
        # en cambio, lee SÓLO el latente: C es una propiedad del espécimen y no
        # debe depender de dónde esté la grieta ahora mismo.
        self.rate_head = nn.Linear(latent + n_features + 2, 1)   # incremento de grieta
        self.initial_head = nn.Linear(latent, 1)       # grieta absoluta, primer ciclo
        self.coefficient_head = nn.Linear(latent, 1)
        self.register_buffer("log_c_prior", torch.tensor(float(log_c_prior)))
        self.log_c_range = float(log_c_range)
        # Start both heads at the population prior: zero crack offset and the
        # identified log10 C, so training begins from the physics rather than
        # from noise.
        # Arranque en "incremento pequeño y positivo": softplus(-1) ~ 0,31 mm,
        # del orden del crecimiento típico entre medidas consecutivas.
        nn.init.zeros_(self.rate_head.weight)
        nn.init.constant_(self.rate_head.bias, -1.0)     # ~0,31 mm por 1000 ciclos
        nn.init.zeros_(self.initial_head.weight)
        nn.init.constant_(self.initial_head.bias, 0.5)
        nn.init.zeros_(self.coefficient_head.weight)
        nn.init.zeros_(self.coefficient_head.bias)

    def forward(self, x, feats, a_prev, log_dn, dn):
        return self.crack_mm(self.encoder(x), feats, a_prev, log_dn, dn)

    def crack_mm(self, z, feats, a_prev, log_dn, dn, cycles_scale: float = 1000.0):
        """Crack length in mm por forma de onda, no negativa por construcción.

        ``a_prev`` y ``log_dn`` son el contexto secuencial. En entrenamiento
        ``a_prev`` es la grieta medida en el ciclo anterior (con ruido); en
        inferencia es la propia estimación previa del modelo, aplicada de forma
        recursiva.
        """
        feats_std = (feats - self.feature_mean) / self.feature_std.clamp(min=1e-6)
        h = torch.cat([z, feats_std, a_prev.unsqueeze(-1), log_dn.unsqueeze(-1)], dim=-1)
        # Parametrización **incremental**:  a(k) = a(k-1) + softplus(...),
        # con ΔN entre las entradas (no como multiplicador).
        #
        # Se llegó aquí por eliminación, y las dos alternativas descartadas son
        # instructivas:
        #
        # * Predecir la grieta **absoluta** hace que la red ignore el contexto
        #   y se sature en un valor constante — se estancaba en 2,6 mm con la
        #   grieta real en 7,5.
        # * Predecir una **tasa** y multiplicarla por ΔN parece lo físicamente
        #   correcto, pero amplifica cualquier error por ΔN/1000, que en este
        #   dataset va de 0,5 a 36 (el hueco 14000→50000 de T3). El LOSO se
        #   disparaba a 6,7 mm.
        #
        # Dejar ΔN como *entrada* permite a la red modular el incremento sin
        # que un hueco largo multiplique el error. Sigue siendo **monótona
        # creciente por construcción** (softplus ≥ 0), así que el factor M(i)
        # del certamen se cumple sin post-proceso. En el primer ciclo, sin
        # grieta previa, actúa la cabeza absoluta.
        # Salida ABSOLUTA descartada: con la cabeza híbrida sube a 1,66 mm
        # LOSO (frente a 0,97) porque la red se estanca — T1 se quedaba en 5,07
        # con la grieta real en 7,46. La forma incremental gana con claridad.
        incremental = a_prev + F.softplus(self.rate_head(h)).squeeze(-1)
        absoluta = F.softplus(self.initial_head(z)).squeeze(-1)
        return torch.where(dn > 0, incremental, absoluta)

    def log_c(self, z, specimen_index: torch.Tensor, n_specimens: int):
        """One bounded ``log10 C`` per specimen from its mean-pooled latent.

        ``specimen_index`` maps each waveform in the batch to its specimen, so
        the pooling is done inside the graph and the coefficient stays a
        specimen-level quantity even when batches mix specimens.
        """
        pooled = torch.zeros(n_specimens, z.shape[1], device=z.device, dtype=z.dtype)
        counts = torch.zeros(n_specimens, device=z.device, dtype=z.dtype)
        pooled.index_add_(0, specimen_index, z)
        counts.index_add_(0, specimen_index, torch.ones_like(specimen_index, dtype=z.dtype))
        pooled = pooled / counts.clamp(min=1.0).unsqueeze(-1)
        u = self.coefficient_head(pooled).squeeze(-1)
        return self.log_c_prior + self.log_c_range * torch.tanh(u)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
