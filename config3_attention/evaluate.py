from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config1_cnn_pinn.data import build_specimen_batches, load_labels
from config1_cnn_pinn.evaluate import ABLATION, REFERENCES, build_submission
from config1_cnn_pinn.train import Config, TRAIN_POOL, train_ensemble

RESULTS = Path(__file__).resolve().parent / "results"

VARIANTES = {
    "Configuración 1 — promedio de peso igual": "cnn_pinn",
    "Configuración 3 — autoatención": "cnn_attention_pinn",
}


def main() -> None:
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)
    base = dict(ABLATION["Configuración 1 (1D-CNN + PINN)"])
    labels = load_labels()

    filas, detalle = [], {}
    for etiqueta, arq in VARIANTES.items():
        cfg = Config(**{**base, "architecture": arq})
        batches = build_specimen_batches(labels, cfg.m_exponent)
        print(f"\n=== {etiqueta} ===")
        modelos = train_ensemble(cfg, batches, [n for n in TRAIN_POOL if n in batches], n_seeds=5)
        entregas = [build_submission(modelos, batches, cfg, s) for s in ("T7", "T8")]
        for e in entregas:
            print(f"  {e['specimen']}: penalización {e['penalizacion']:9.2f}  "
                  f"RMSE {e['rmse_mm']:6.3f} mm  "
                  f"log10C {e['log10_C_predicho']:.3f} (correcto {e['log10_C_implicado']:.3f})  "
                  f"|Δ| {abs(e['log10_C_predicho']-e['log10_C_implicado']):.3f}")
        filas.append({
            "variante": etiqueta,
            "T7": round(entregas[0]["penalizacion"], 2),
            "T8": round(entregas[1]["penalizacion"], 2),
            "total": round(sum(e["penalizacion"] for e in entregas), 2),
            "RMSE T7": round(entregas[0]["rmse_mm"], 3),
            "RMSE T8": round(entregas[1]["rmse_mm"], 3),
            "|Δlog10C| T7": round(abs(entregas[0]["log10_C_predicho"]-entregas[0]["log10_C_implicado"]), 3),
            "|Δlog10C| T8": round(abs(entregas[1]["log10_C_predicho"]-entregas[1]["log10_C_implicado"]), 3),
        })
        detalle[etiqueta] = entregas

    tabla = pd.DataFrame(filas)
    print("\n" + "=" * 105)
    print("CONFIGURACIÓN 3 frente a CONFIGURACIÓN 1 — única diferencia: cómo se resume la secuencia")
    print("=" * 105)
    print(tabla.to_string(index=False))
    print("\n  Objetivo heredado del diagnóstico: |Δlog10C| de T8 por debajo de 0.11")
    tabla.to_csv(RESULTS / "config3_vs_config1.csv", index=False)
    (RESULTS / "entregas.json").write_text(json.dumps(detalle, indent=2, default=str))


if __name__ == "__main__":
    main()
