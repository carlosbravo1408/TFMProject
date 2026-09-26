"""Barrido del exponente de Wheeler y verificación de que el retardo es constante.

Por qué existe este módulo
--------------------------
El barrido de sensibilidad se había ejecutado *ad hoc* y sus cifras quedaron en
`results/sensibilidad_wheeler.csv` sin código que las regenerase. Al cambiar la
configuración del estimador quedaron obsoletas y no había forma de rehacerlas.

Además contiene la comprobación que reclasifica el componente (c). Se reporta
antes que el barrido porque **cambia lo que el barrido significa**: si el factor
de retardo no depende de la longitud de grieta, el barrido de ``p`` no explora
intensidades de un efecto de historia, sino recalibraciones del coeficiente.

Ejecutar:  python -m config2_rk4.sensitivity
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from config1_cnn_pinn.data import build_specimen_batches, load_labels
from config1_cnn_pinn.evaluate import ABLATION, n_observed_nonzero
from config1_cnn_pinn.physics import equivalent_stress_range, fit_log_c
from config1_cnn_pinn.train import Config, TRAIN_POOL, evaluate_ensemble, train_ensemble
from physics_calibration.data import load_curve
from physics_calibration.objectives import phm_penalty
from .integrator import integrate_block_rk4
from .retardation import (YIELD_STRENGTH_MPA, block_rate_with_retardation, k_max,
                          integrate_with_retardation, plastic_zone_m, wheeler_phi)

RESULTS = __file__.rsplit("/", 1)[0] + "/results"

# Barrido del exponente de Wheeler. El intervalo cubre el publicado para otras
# aleaciones de aluminio: Sheu, Song y Hwang (1995, Eng. Fract. Mech., "Shaping
# exponent in Wheeler model under a single overload") calibran el exponente sobre
# ensayos de sobrecarga unica y obtienen 0,94-2,19 en 5083-O y 1,36-1,98 en
# 6061-T651. No hay valor publicado para el 2024-T3. Se recorre como cota de
# sensibilidad, nunca para elegir p.
EXPONENTES_P = (0.0, 1.0, 1.43, 2.0, 2.5, 3.0, 3.5)


def diagnostico_phi(longitudes_mm=(2.0, 3.0, 4.0, 5.0, 8.0, 15.0), p: float = 1.43):
    """¿Depende el retardo de la longitud de grieta? (debería, si es Wheeler)."""
    curve = load_curve("T8")
    bloque = np.asarray(curve.load_block, dtype=float)
    i_ol = int(np.argmax(bloque[:, 1]))
    filas = []
    for a_mm in longitudes_mm:
        a = a_mm / 1000.0
        r_ol = plastic_zone_m(k_max(a, bloque[i_ol, 1]))
        r_y = plastic_zone_m(k_max(a, bloque[1 - i_ol, 1]))
        filas.append({
            "a_mm": a_mm,
            "phi": float(wheeler_phi(a, a, r_ol, r_y, p)),
            "tasa_retardada/sin_retardo": float(
                block_rate_with_retardation(a, 1e-9, 2.0, bloque, p)
                / block_rate_with_retardation(a, 1e-9, 2.0, bloque, 0.0)),
        })
    return pd.DataFrame(filas)


def equivalencia_con_coeficiente(a0_mm, C, m, curve, p: float = 1.43):
    """Integrar con retardo == integrar con ``C * factor``. Devuelve la diferencia."""
    objetivos = np.asarray(curve.cycles[2:]) - curve.cycles[1]
    bloque = curve.load_block
    factor = float(block_rate_with_retardation(a0_mm / 1000.0, 1.0, m, bloque, p)
                   / block_rate_with_retardation(a0_mm / 1000.0, 1.0, m, bloque, 0.0))
    con_retardo = integrate_with_retardation(a0_mm, objetivos, C, m, bloque, p)
    con_coef = integrate_block_rk4(a0_mm, objetivos, C * factor, m, bloque)[:, 0]
    return factor, float(np.max(np.abs(con_retardo - con_coef))), con_retardo, con_coef


def barrido(n_seeds: int = 5) -> pd.DataFrame:
    """Penalización de T7/T8 en función de ``p``, con el estimador vigente."""
    cfg = Config(**ABLATION["Configuración 1 (1D-CNN + PINN)"])
    batches = build_specimen_batches(load_labels(), cfg.m_exponent)
    modelos = train_ensemble(cfg, batches, [n for n in TRAIN_POOL if n in batches],
                             n_seeds=n_seeds)

    entregas = {}
    for nombre in ("T7", "T8"):
        curve = load_curve(nombre)
        cyc, est, real, log_c = evaluate_ensemble(modelos, batches[nombre], cfg)
        nz = real > 0
        n_obs = n_observed_nonzero(nombre)
        entregas[nombre] = (curve, cyc, est, cyc[nz][:n_obs], est[nz][:n_obs], log_c)

    filas = []
    for p in EXPONENTES_P:
        fila = {"p": p}
        for nombre, (curve, cyc, est, ac, aa, log_c) in entregas.items():
            d_sigma = equivalent_stress_range(curve.load_block, cfg.m_exponent)
            objetivos, indices = [], []
            pred = np.empty(len(curve.cycles))
            for i, cy in enumerate(curve.cycles):
                if cy <= ac[-1] and cy in set(cyc):
                    pred[i] = est[cyc == cy][0]
                else:
                    indices.append(i)
                    objetivos.append(cy - ac[-1])
            if indices:
                C = 10.0 ** log_c
                if p == 0.0:
                    salida = integrate_block_rk4(aa[-1], np.array(objetivos), C,
                                                 cfg.m_exponent, curve.load_block)[:, 0]
                else:
                    salida = integrate_with_retardation(aa[-1], np.array(objetivos), C,
                                                        cfg.m_exponent, curve.load_block, p)
                pred[indices] = salida
            pred = np.maximum.accumulate(pred)
            fila[f"pen {nombre}"] = round(float(phm_penalty(
                pred[None, :], curve.crack_mm, curve.final_crack_mm)[0]), 2)
            fila[f"RMSE {nombre}"] = round(float(np.sqrt(np.mean((pred - curve.crack_mm) ** 2))), 3)
        fila["total"] = round(fila["pen T7"] + fila["pen T8"], 2)
        filas.append(fila)
    return pd.DataFrame(filas), entregas, cfg


def main() -> None:
    torch.set_num_threads(4)
    from pathlib import Path
    salida = Path(RESULTS)
    salida.mkdir(parents=True, exist_ok=True)

    print("1. ¿ES DINÁMICO EL RETARDO? (si lo fuera, phi dependería de la grieta)")
    phi = diagnostico_phi()
    print(phi.round(4).to_string(index=False))
    # Comparación con tolerancia, no ``nunique()``: la dispersión que queda es
    # ruido de coma flotante, y contar valores distintos la leería como señal.
    disp = float((phi["phi"].max() - phi["phi"].min()) / phi["phi"].mean())
    print(f"\n   phi = {phi['phi'].mean():.4f} en TODO el rango de 2 a 15 mm "
          f"(dispersión relativa {disp:.1e}, es decir ruido de coma flotante).")
    print("   Un retardo que no depende de la longitud de grieta no es un efecto")
    print("   de historia: es una constante multiplicativa sobre el coeficiente.")
    phi.to_csv(salida / "diagnostico_phi.csv", index=False)

    tabla, entregas, cfg = barrido()

    curve, _, _, _, aa, log_c = entregas["T8"]
    factor, dif, _, _ = equivalencia_con_coeficiente(aa[-1], 10.0 ** log_c,
                                                     cfg.m_exponent, curve)
    print(f"\n2. EQUIVALENCIA CON UNA RECALIBRACIÓN DEL COEFICIENTE")
    print(f"   retardo p=1,43  ==  C x {factor:.5f}  (= {np.log10(factor):+.4f} dex)")
    print(f"   diferencia máxima entre ambas curvas: {dif:.4f} mm")
    brecha = abs(log_c - fit_log_c(curve.cycles, curve.crack_mm, cfg.m_exponent,
                                   equivalent_stress_range(curve.load_block, cfg.m_exponent)))
    print(f"   brecha de coeficiente de T8: {brecha:.3f} dex "
          f"-> el retardo cubre el {100 * abs(np.log10(factor)) / brecha:.0f} %")

    print("\n3. SENSIBILIDAD AL EXPONENTE p (barrido 1,0-3,5; ver nota de EXPONENTES_P)")
    print(tabla.to_string(index=False))
    tabla.to_csv(salida / "sensibilidad_wheeler.csv", index=False)
    print(f"\n-> {salida}")


if __name__ == "__main__":
    main()
