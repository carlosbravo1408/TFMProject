from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from physics_calibration.data import load_curve
from physics_calibration.objectives import phm_penalty
from .data import build_specimen_batches, load_labels
from .evaluate import ABLATION, build_submission
from .physics import equivalent_stress_range, propagate_mm
from .train import Config, RESULTS, TRAIN_POOL, train_ensemble


def sweep_coefficient(models, batches, cfg: Config, name: str,
                      log_c_grid: np.ndarray | None = None) -> pd.DataFrame:
    curve = load_curve(name)
    d_sigma = equivalent_stress_range(curve.load_block, cfg.m_exponent)
    cota = cfg.cota_entrega_mm if cfg.cota_entrega_mm is not None else 20.0
    sub = build_submission(models, batches, cfg, name)

    anchors = [i for i, s in enumerate(sub["source"]) if s == "estimación 1D-CNN"]
    last = anchors[-1]
    a0, n0 = sub["pred_mm"][last], curve.cycles[last]

    if log_c_grid is None:
        log_c_grid = np.linspace(-9.6, -7.8, 361)

    rows = []
    for log_c in log_c_grid:
        pred = np.array(sub["pred_mm"], dtype=float).copy()
        for i, cycle in enumerate(curve.cycles):
            if i > last:
                pred[i] = propagate_mm(a0, cycle - n0, 10.0 ** log_c,
                                       cfg.m_exponent, d_sigma, a_max_mm=cota)
        pred = np.maximum.accumulate(pred)
        rows.append({
            "log10_C": float(log_c),
            "rmse_mm": float(np.sqrt(np.mean((pred - curve.crack_mm) ** 2))),
            "penalizacion": float(phm_penalty(pred[None, :], curve.crack_mm,
                                              curve.final_crack_mm)[0]),
        })
    out = pd.DataFrame(rows)
    out.attrs["submission"] = sub
    return out


def trajectory(models, batches, cfg: Config, name: str, log_c: float) -> np.ndarray:
    curve = load_curve(name)
    d_sigma = equivalent_stress_range(curve.load_block, cfg.m_exponent)
    cota = cfg.cota_entrega_mm if cfg.cota_entrega_mm is not None else 20.0
    sub = build_submission(models, batches, cfg, name)
    anchors = [i for i, s in enumerate(sub["source"]) if s == "estimación 1D-CNN"]
    last = anchors[-1]
    a0, n0 = sub["pred_mm"][last], curve.cycles[last]
    pred = np.array(sub["pred_mm"], dtype=float).copy()
    for i, cycle in enumerate(curve.cycles):
        if i > last:
            pred[i] = propagate_mm(a0, cycle - n0, 10.0 ** log_c, cfg.m_exponent,
                                   d_sigma, a_max_mm=cota)
    return np.maximum.accumulate(pred)


def verificar_coeficiente(models, batches, cfg: Config, nombres=("T7", "T8"),
                          verbose: bool = True):
    resumen, barridos, curvas = [], {}, {}
    for name in nombres:
        barrido = sweep_coefficient(models, batches, cfg, name)
        sub = barrido.attrs["submission"]
        mejor = barrido.loc[barrido["rmse_mm"].idxmin()]
        mejor_pen = barrido.loc[barrido["penalizacion"].idxmin()]

        # The margin the coefficient head would have to meet.
        banda = barrido.loc[barrido["rmse_mm"] <= 1.0, "log10_C"]

        if verbose:
            print(f"\n=== {name} ===")
            print(f"  log10 C predicho por el modelo : {sub['log10_C_predicho']:+.3f}  "
                  f"-> RMSE {sub['rmse_mm']:6.3f} mm, penalización {sub['penalizacion']:9.2f}")
            print(f"  log10 C óptimo (barrido)       : {mejor['log10_C']:+.3f}  "
                  f"-> RMSE {mejor['rmse_mm']:6.3f} mm, penalización {mejor['penalizacion']:9.2f}")
            print(f"  log10 C de mínima penalización : {mejor_pen['log10_C']:+.3f}  "
                  f"-> RMSE {mejor_pen['rmse_mm']:6.3f} mm, penalización {mejor_pen['penalizacion']:9.2f}")
        if len(banda):
            fuera = max(0.0, banda.min() - sub["log10_C_predicho"],
                        sub["log10_C_predicho"] - banda.max())
            if verbose:
                print(f"  banda con RMSE <= 1 mm         : [{banda.min():+.3f}, {banda.max():+.3f}]  "
                      f"(anchura {banda.max() - banda.min():.3f} dex)")
                print(f"  el modelo cae {'DENTRO' if fuera == 0 else f'FUERA por {fuera:.3f} dex'} "
                      f"de esa banda")
        elif verbose:
            print("  ningún coeficiente baja el RMSE de 1 mm")

        barridos[name] = barrido
        resumen.append({
            "espécimen": name,
            "log10 C predicho": round(sub["log10_C_predicho"], 3),
            "RMSE con el predicho (mm)": round(sub["rmse_mm"], 3),
            "penalización con el predicho": round(sub["penalizacion"], 2),
            "log10 C óptimo": round(float(mejor["log10_C"]), 3),
            "RMSE con el óptimo (mm)": round(float(mejor["rmse_mm"]), 3),
            "penalización con el óptimo": round(float(mejor["penalizacion"]), 2),
            "banda RMSE<=1mm (dex)": round(float(banda.max() - banda.min()), 3) if len(banda) else None,
        })
        curvas[name] = {
            "cycles": load_curve(name).cycles.tolist(),
            "true_mm": load_curve(name).crack_mm.tolist(),
            "pred_predicho": trajectory(models, batches, cfg, name,
                                        sub["log10_C_predicho"]).tolist(),
            "pred_optimo": trajectory(models, batches, cfg, name,
                                      float(mejor["log10_C"])).tolist(),
            "cutoff_cycle": sub["cutoff_cycle"],
        }

    return pd.DataFrame(resumen), barridos, curvas


def main() -> None:
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)

    cfg = Config(**ABLATION["Configuración 1 (1D-CNN + PINN)"])
    batches = build_specimen_batches(load_labels(), cfg.m_exponent)
    print("Entrenando el modelo de la Configuración 1 (5 semillas)…")
    models = train_ensemble(cfg, batches, [n for n in TRAIN_POOL if n in batches], n_seeds=5)

    tabla, barridos, curvas = verificar_coeficiente(models, batches, cfg)
    for name, barrido in barridos.items():
        barrido.to_csv(RESULTS / f"barrido_coeficiente_{name}.csv", index=False)

    print("\n" + "=" * 100)
    print("VERIFICACIÓN SOBRE EL MODELO CONSTRUIDO — sólo cambia el coeficiente")
    print("=" * 100)
    print(tabla.to_string(index=False))

    tabla.to_csv(RESULTS / "verificacion_coeficiente.csv", index=False)
    (RESULTS / "verificacion_curvas.json").write_text(json.dumps(curvas, indent=2))


if __name__ == "__main__":
    main()
