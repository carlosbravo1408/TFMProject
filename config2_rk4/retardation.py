from __future__ import annotations

import numpy as np

MM_PER_M = 1000.0

# Handbook yield strength of thin 2024-T3 sheet, MPa.
YIELD_STRENGTH_MPA = 345.0
# 2 = plane stress (1.6 mm sheet); 6 would be plane strain.
PLANE_STRESS_ALPHA = 2.0


def plastic_zone_m(k_max, sigma_y=YIELD_STRENGTH_MPA, alpha=PLANE_STRESS_ALPHA):
    return (1.0 / (alpha * np.pi)) * (np.asarray(k_max) / sigma_y) ** 2


def k_max(a_m, s_max_mpa, geometry=1.0):
    return geometry * s_max_mpa * np.sqrt(np.pi * np.clip(a_m, 1e-9, None))


def wheeler_phi(a_m, a_ol_m, r_ol_m, r_y_m, exponent):
    frontera = a_ol_m + r_ol_m
    restante = np.maximum(frontera - a_m, 1e-12)
    dentro = (a_m + r_y_m) < frontera
    return np.where(dentro, np.clip(r_y_m / restante, 0.0, 1.0) ** exponent, 1.0)


def block_rate_with_retardation(a_m, C, m, load_block, exponent,
                                sigma_y=YIELD_STRENGTH_MPA, geometry=1.0):
    bloque = np.asarray(load_block, dtype=float)
    n_ciclos, s_max, s_min = bloque[:, 0], bloque[:, 1], bloque[:, 2]
    peso = n_ciclos / n_ciclos.sum()

    # The highest segment is the overload, evaluated at the current a: growth within
    # a block (~0.05 mm) is negligible against the plastic zone.
    i_ol = int(np.argmax(s_max))
    r_ol = plastic_zone_m(k_max(a_m, s_max[i_ol], geometry), sigma_y)

    total = 0.0
    for j, (w, smax_j, smin_j) in enumerate(zip(peso, s_max, s_min)):
        d_sigma = smax_j - smin_j
        tasa = C * (geometry * d_sigma * np.sqrt(np.pi * np.clip(a_m, 1e-9, None))) ** m
        if j == i_ol:
            phi = 1.0
        else:
            r_y = plastic_zone_m(k_max(a_m, smax_j, geometry), sigma_y)
            phi = wheeler_phi(a_m, a_m, r_ol, r_y, exponent)
        total = total + w * phi * tasa
    return total


def integrate_with_retardation(a0_mm, target_cycles, C, m, load_block, exponent,
                               max_step: float = 250.0, sigma_y=YIELD_STRENGTH_MPA,
                               geometry: float = 1.0, a_max_mm: float = 20.0):
    a = float(a0_mm) / MM_PER_M
    objetivos = np.asarray(target_cycles, dtype=float)
    salida = np.empty(len(objetivos))
    ciclo = 0.0
    f = lambda x: block_rate_with_retardation(x, C, m, load_block, exponent, sigma_y, geometry)
    for i, objetivo in enumerate(objetivos):
        tramo = objetivo - ciclo
        if tramo > 0:
            pasos = max(1, int(np.ceil(tramo / max_step)))
            h = tramo / pasos
            for _ in range(pasos):
                k1 = f(a); k2 = f(a + 0.5 * h * k1)
                k3 = f(a + 0.5 * h * k2); k4 = f(a + h * k3)
                a = min(a + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), a_max_mm / MM_PER_M)
            ciclo = objetivo
        salida[i] = a
    return salida * MM_PER_M


def retardation_summary(a_mm, load_block, exponent, sigma_y=YIELD_STRENGTH_MPA):
    a_m = a_mm / MM_PER_M
    bloque = np.asarray(load_block, dtype=float)
    n_ciclos, s_max, s_min = bloque[:, 0], bloque[:, 1], bloque[:, 2]
    i_ol = int(np.argmax(s_max))
    r_ol = plastic_zone_m(k_max(a_m, s_max[i_ol]), sigma_y)
    filas = []
    for j in range(len(s_max)):
        r_y = plastic_zone_m(k_max(a_m, s_max[j]), sigma_y)
        phi = 1.0 if j == i_ol else float(wheeler_phi(a_m, a_m, r_ol, r_y, exponent))
        filas.append({
            "segmento": j, "n_ciclos": int(n_ciclos[j]), "Smax": s_max[j],
            "Kmax": float(k_max(a_m, s_max[j])),
            "zona_plastica_mm": float(r_y * MM_PER_M),
            "phi": phi, "sobrecarga": j == i_ol,
        })
    return filas
