from __future__ import annotations

import numpy as np

MM_PER_M = 1000.0
_A_MIN_M, _A_MAX_M = 1e-6, 0.02


def paris_rate(a_m, C, m, d_sigma_mpa, geometry=1.0):
    a = np.clip(a_m, _A_MIN_M, _A_MAX_M)
    return C * (geometry * d_sigma_mpa * np.sqrt(np.pi * a)) ** m


def wheeler_factor(a_m, a_overload_m, plastic_zone_m, exponent):
    frontera = a_overload_m + plastic_zone_m
    dentro = a_m < frontera
    restante = np.maximum(frontera - a_m, 1e-12)
    razon = np.clip(plastic_zone_m / restante, 0.0, 1.0)
    return np.where(dentro, razon ** exponent, 1.0)


# a_max_mm must be the closed form's delivery bound, or the comparison would
# measure the bound instead of the integrator.
def integrate_block_rk4(a0_mm, target_cycles, C, m, load_block,
                        max_step: float = 250.0, geometry: float = 1.0,
                        a_max_mm: float = 20.0):
    bloque = np.asarray(load_block, dtype=float)
    n_ciclos, s_max, s_min = bloque[:, 0], bloque[:, 1], bloque[:, 2]
    d_sigma = s_max - s_min
    peso = n_ciclos / n_ciclos.sum()

    def tasa(a_m):
        return sum(w * paris_rate(a_m, C, m, ds, geometry) for w, ds in zip(peso, d_sigma))

    a_max = min(a_max_mm / MM_PER_M, _A_MAX_M)
    a = np.atleast_1d(np.asarray(a0_mm, dtype=float)).astype(float) / MM_PER_M
    objetivos = np.asarray(target_cycles, dtype=float)
    salida = np.empty((len(objetivos),) + a.shape)
    ciclo = 0.0
    for i, objetivo in enumerate(objetivos):
        tramo = objetivo - ciclo
        if tramo > 0:
            pasos = max(1, int(np.ceil(tramo / max_step)))
            h = tramo / pasos
            for _ in range(pasos):
                k1 = tasa(a)
                k2 = tasa(a + 0.5 * h * k1)
                k3 = tasa(a + 0.5 * h * k2)
                k4 = tasa(a + h * k3)
                a = np.clip(a + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), _A_MIN_M, a_max)
            ciclo = objetivo
        salida[i] = a
    return salida * MM_PER_M
