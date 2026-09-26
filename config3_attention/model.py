"""Configuración 3 — 1D-CNN (a) + PINN (b) + autoatención (d).

Dónde entra la atención, y por qué ahí
--------------------------------------
En la Configuración 1 la cabeza de coeficiente promedia el latente de **todas**
las ondas del espécimen con **peso igual**:

    log10 C = f( mean_k z_k )

Ese promedio es el eslabón débil. El diagnóstico de la Configuración 1 lo
localiza: el estimador de grieta funciona y lo que se desvía es ese escalar, que
en T8 cae fuera del valor implicado por la curva por algo menos de cuatro
décimas de década. Las cifras vigentes las calcula el cuaderno; no se copian
aquí para que no queden desfasadas.

Promediar con peso igual trata por igual al ciclo de referencia sano —que por
construcción no contiene daño— y a los ciclos que sí informan de la velocidad
de crecimiento. La autoatención sustituye ese promedio por una **ponderación
aprendida sobre la secuencia de ciclos del espécimen**:

    log10 C = f( Attn( z_1, ..., z_K ) )

Qué se espera de ella, y qué límite tiene
-----------------------------------------
La hipótesis del TFM es que la atención sobre el historial completo permita
**detectar que un espécimen está en otro régimen de carga**. El conjunto de
datos no permite contrastarla: **ningún espécimen de entrenamiento es de
amplitud variable** y el único, T8, es el de evaluación. Es la misma restricción
de conteo que limita al retardo en la Configuración 2, y ninguna arquitectura la
sortea.

Lo que sí puede aprenderse con estos datos, y es medible sin tocar T7 ni T8, es
a **ponderar los ciclos por su informatividad**: descontar la referencia sana y
dar peso a los tramos donde la grieta crece de forma consistente.

**Medido, no ocurre.** Al leer los pesos de la consulta aprendida, la atención
pondera **más** los ciclos sin daño que los que informan del crecimiento, y lo
hace en todos los especímenes. La expectativa anterior queda refutada por medida
directa. No es necesariamente un defecto —el registro sano es la referencia
contra la que se define el canal diferencial de todos los demás, así que puede
portar información de normalización—, pero **no puede invocarse como explicación
de ninguna ventaja**, y encaja con que la atención no mueva el coeficiente de
T8: el resumen recibe sobre todo ondas sin crecimiento. El apartado
correspondiente del cuaderno da las razones medidas.

Decisiones de diseño impuestas por el tamaño de la muestra
----------------------------------------------------------
* **Secuencias de 3 a 9 ciclos** (T7 y T8 tienen 4). A este tamaño se usa **una
  sola capa, dos cabezales**, sobre la dimensión del latente ya existente (32).
* **Un token por ciclo**, no por onda: las dos repeticiones de una medida son
  casi idénticas y darían al mecanismo dos entradas redundantes.
* **La atención sólo alimenta la cabeza de coeficiente.** La cabeza de grieta se
  deja exactamente como en la Configuración 1. Así la ablación aísla la
  contribución de (d) al problema que (d) debe resolver, en vez de mezclarla
  con cambios en la estimación.
* La atención es **bidireccional** (cada ciclo ve a todos). Es transductivo
  —lee ondas posteriores del espécimen bajo prueba— pero admisible bajo el
  protocolo del certamen, que entregaba el conjunto completo de señales de cada
  espécimen de validación de una vez. Se declara como limitación, igual que la
  normalización por espécimen.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config1_cnn_pinn.model import CnnPinn


class CycleAttention(nn.Module):
    """Autoatención de una capa sobre la secuencia de ciclos de un espécimen."""

    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        # Consulta aprendida que resume la secuencia en un vector: es la que
        # decide a qué ciclos mirar para estimar el coeficiente.
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """``seq`` de forma (1, K, d) -> resumen (1, d)."""
        h, _ = self.attn(seq, seq, seq, need_weights=False)
        h = self.norm1(seq + h)
        h = self.norm2(h + self.ff(h))
        resumen, pesos = nn.functional.multi_head_attention_forward(
            self.query.transpose(0, 1), h.transpose(0, 1), h.transpose(0, 1),
            h.shape[-1], 1,
            None, None, None, None, False, 0.0,
            torch.eye(h.shape[-1], device=h.device), None,
            use_separate_proj_weight=True,
            q_proj_weight=torch.eye(h.shape[-1], device=h.device),
            k_proj_weight=torch.eye(h.shape[-1], device=h.device),
            v_proj_weight=torch.eye(h.shape[-1], device=h.device),
            training=self.training,
        )
        self.last_weights = pesos.detach()
        return resumen.transpose(0, 1).squeeze(1)


class CnnAttentionPinn(CnnPinn):
    """Configuración 1 con la cabeza de coeficiente movida a autoatención."""

    def __init__(self, *args, n_heads: int = 2, attn_dropout: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.encoder.latent_dim
        # Codificación posicional: la secuencia no es uniforme en ciclos, así
        # que la posición útil no es el índice sino el hueco temporal.
        self.position = nn.Linear(2, d)
        self.attention = CycleAttention(d, n_heads, attn_dropout)
        nn.init.zeros_(self.position.weight)
        nn.init.zeros_(self.position.bias)

    def log_c(self, z, specimen_index, n_specimens, cycles=None, log_dn=None):
        """``log10 C`` por espécimen, resumiendo su secuencia con atención."""
        if cycles is None:                      # sin contexto de secuencia: como Config 1
            return super().log_c(z, specimen_index, n_specimens)

        salidas = []
        for s in range(n_specimens):
            mask = specimen_index == s
            zs, cs, ds = z[mask], cycles[mask], log_dn[mask]
            # Un token por ciclo: promedio de las repeticiones de esa medida.
            unicos, inverso = torch.unique(cs, return_inverse=True)
            tokens = torch.zeros(len(unicos), zs.shape[1], device=z.device, dtype=z.dtype)
            cuenta = torch.zeros(len(unicos), device=z.device, dtype=z.dtype)
            tokens.index_add_(0, inverso, zs)
            cuenta.index_add_(0, inverso, torch.ones_like(inverso, dtype=z.dtype))
            tokens = tokens / cuenta.clamp(min=1.0).unsqueeze(-1)

            hueco = torch.zeros(len(unicos), device=z.device, dtype=z.dtype)
            hueco.index_add_(0, inverso, ds)
            hueco = hueco / cuenta.clamp(min=1.0)
            avance = torch.linspace(0.0, 1.0, len(unicos), device=z.device, dtype=z.dtype)
            tokens = tokens + self.position(torch.stack([hueco, avance], dim=-1))

            resumen = self.attention(tokens.unsqueeze(0))
            salidas.append(self.coefficient_head(resumen).squeeze(-1))
        u = torch.cat(salidas)
        return self.log_c_prior + self.log_c_range * torch.tanh(u)
