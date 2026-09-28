from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import build_specimen_batches, load_labels
from .hyperparameter_sources import LOG_C_FAA, M_FAA
from .select_prognosis import BASE, robustez
from .train import Config, LOSO_FOLDS, RESULTS, TRAIN_POOL, evaluate_ensemble, train_ensemble

VARIANTES = {
    "Identificado, m = 2,00": dict(m_exponent=2.00),
    "FAA 2024-T3 chapa (Forman et al., 2005)": dict(m_exponent=M_FAA, log_c_prior=LOG_C_FAA),
}


def robustez_constantes(n_seeds: int = 3, etiquetas=None, verbose: bool = True):
    etiquetas = load_labels() if etiquetas is None else etiquetas
    resumen, detalle = [], []
    for nombre, extra in VARIANTES.items():
        cfg = Config(**{**BASE, **extra})
        m = cfg.m_exponent
        batches = {k: v for k, v in build_specimen_batches(etiquetas, m).items()
                   if k not in ("T7", "T8")}
        assert "T7" not in batches and "T8" not in batches, "fuga: T7/T8 en la selección"
        cache = {}
        for fuera in LOSO_FOLDS:
            nombres = [n for n in TRAIN_POOL if n != fuera and n in batches]
            cache[fuera] = evaluate_ensemble(
                train_ensemble(cfg, batches, nombres, n_seeds=n_seeds), batches[fuera], cfg)
        if verbose:
            print(f"\n{nombre}: estimadores entrenados (m = {m:.3f})", flush=True)

        coeficientes = {"cabeza": cache}
        if "log_c_prior" in extra:
            coeficientes["fijo"] = {f: (*c[:3], extra["log_c_prior"]) for f, c in cache.items()}
        for coef, cch in coeficientes.items():
            r = robustez(cch, m, "cabeza")
            nominal = r[r["delta_log10C"] == 0.0].iloc[0]
            resumen.append({"procedencia": nombre, "coeficiente": coef, "m": round(m, 3),
                            "media_nominal": nominal["media"],
                            "peor_fold_nominal": nominal["peor_fold"],
                            "peor_fold_banda": r["peor_fold"].max(),
                            "puntos_en_tope_banda": int(r["puntos_en_tope"].sum())})
            r.insert(0, "coeficiente", coef)
            r.insert(0, "procedencia", nombre)
            detalle.append(r)
            if verbose:
                print(f"  [{coef}]\n" + r.round(2).to_string(index=False), flush=True)

    tabla = pd.DataFrame(resumen).sort_values("peor_fold_banda").reset_index(drop=True)
    return tabla, pd.concat(detalle, ignore_index=True)


def main(n_seeds: int = 3) -> None:
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)
    tabla, detalle = robustez_constantes(n_seeds=n_seeds)
    detalle.to_csv(RESULTS / "robustez_literatura_detalle.csv", index=False)
    tabla.to_csv(RESULTS / "robustez_literatura.csv", index=False)
    print("\n" + "=" * 100)
    print("CRITERIO SIN FUGA — minimax del peor fold de entrenamiento sobre la banda ±0,3 dex")
    print("=" * 100)
    print(tabla.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
