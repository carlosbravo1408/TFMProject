"""¿Resisten las constantes publicadas del 2024-T3 el criterio de selección sin fuga?

``hyperparameter_sources`` mostró que el ajuste de Walker de la FAA para chapa de
2024-T3 (Forman et al., 2005, DOT/FAA/AR-05/15) **pierde en T7** (169,45 frente a
12,01) pero **gana en T8** (12,77 frente a 8631,76). Adoptarlo *por* su resultado en
T8 sería elegir con un espécimen de validación: fuga. Aquí se somete al mismo
criterio con el que se eligió el exponente —minimax del peor fold de entrenamiento
sobre la banda de ±0,3 dex del coeficiente— **sin cargar T7 ni T8**.

Variantes, todas con la configuración de ``select_prognosis.BASE``:

``Identificado, m = 2,00 / cabeza``   control; debe reproducir el resultado que
                                      ``select_prognosis.buscar_exponente`` da a m = 2,00.
``FAA / cabeza``                      la cabeza física parte del *prior* de la FAA y lo
                                      ajusta por espécimen.
``FAA / fijo``                        el coeficiente publicado tal cual, sin adaptar.

Ejecutar:  python -m config1_cnn_pinn.robustez_literatura
"""
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
    """Somete cada procedencia al criterio sin fuga. Devuelve ``(resumen, detalle)``.

    No escribe ni lee ficheros. El control de reproducibilidad frente a
    ``select_prognosis`` se obtiene comparando la fila ``Identificado, m = 2,00``
    de este resumen con la salida de ``buscar_exponente`` en la misma sesión,
    no leyendo un CSV de una ejecucion anterior.
    """
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
            r = robustez(cch, m, "cabeza")   # «cabeza» usa el log C que trae la caché
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
    """Atajo de linea de comandos. El notebook llama a ``robustez_constantes``."""
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
