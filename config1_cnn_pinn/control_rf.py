from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import signals as S
from .data import load_labels
from .train import TRAIN_POOL

SEED = 0
N_TREES = 200
# With 89 waveforms, depth is the dominant regulariser.
MAX_DEPTH = 4


def _tabla(labels) -> pd.DataFrame:
    filas = []
    for nombre, df in labels.items():
        etiquetas = dict(zip(df["cycle"].to_numpy(), df["crack_length_mm"].to_numpy()))
        for ciclo in S.signal_cycles(name=nombre):
            # Same exclusions as the network, so the comparison has a single variable.
            if ciclo not in etiquetas or (nombre, ciclo) in S.EXCLUDED_RECORDS:
                continue
            for rep in S.available_repetitions(S.DEFAULT_ROOT, nombre, ciclo):
                filas.append({
                    "especimen": nombre, "ciclo": ciclo, "rep": rep,
                    "grieta_mm": float(etiquetas[ciclo]),
                    "rasgos": S.scalar_features(nombre, ciclo, rep),
                })
    return pd.DataFrame(filas)


def _ajustar(tabla: pd.DataFrame, nombres) -> RandomForestRegressor:
    sub = tabla[tabla["especimen"].isin(nombres)]
    X, y = [], []
    for nombre, grupo in sub.groupby("especimen"):
        grupo = grupo.sort_values(["ciclo", "rep"])
        previa = 0.0
        for ciclo, ciclo_grupo in grupo.groupby("ciclo", sort=True):
            for _, fila in ciclo_grupo.iterrows():
                X.append(np.concatenate([fila["rasgos"], [previa]]))
                y.append(fila["grieta_mm"])
            previa = float(ciclo_grupo["grieta_mm"].iloc[0])
    modelo = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                   random_state=SEED, n_jobs=1)
    modelo.fit(np.asarray(X), np.asarray(y))
    return modelo


def _estimar(modelo, tabla: pd.DataFrame, nombre: str):
    grupo = tabla[tabla["especimen"] == nombre].sort_values(["ciclo", "rep"])
    ciclos, est, real, previa = [], [], [], 0.0
    for ciclo, ciclo_grupo in grupo.groupby("ciclo", sort=True):
        X = np.stack([np.concatenate([f, [previa]]) for f in ciclo_grupo["rasgos"]])
        valor = max(float(np.mean(modelo.predict(X))), previa)
        ciclos.append(ciclo)
        est.append(valor)
        real.append(float(ciclo_grupo["grieta_mm"].iloc[0]))
        previa = valor
    return np.array(ciclos), np.array(est), np.array(real)


def loso(tabla: pd.DataFrame, folds=("T1", "T3", "T4", "T6")) -> pd.DataFrame:
    filas = []
    for fuera in folds:
        modelo = _ajustar(tabla, [n for n in TRAIN_POOL if n != fuera])
        _, est, real = _estimar(modelo, tabla, fuera)
        nz = real > 0
        filas.append({"fold": fuera, "n_ciclos": int(nz.sum()),
                      "rmse_mm": float(np.sqrt(np.mean((est[nz] - real[nz]) ** 2)))})
    return pd.DataFrame(filas)


def certamen(tabla: pd.DataFrame) -> pd.DataFrame:
    modelo = _ajustar(tabla, list(TRAIN_POOL))
    filas = []
    for nombre in ("T7", "T8"):
        ciclos, est, real = _estimar(modelo, tabla, nombre)
        nz = real > 0
        filas.append({
            "especimen": nombre, "n_ciclos": int(nz.sum()),
            "rmse_estimacion_mm": float(np.sqrt(np.mean((est[nz] - real[nz]) ** 2))),
            "estimado": np.round(est[nz], 3).tolist(),
            "real": np.round(real[nz], 3).tolist(),
        })
    return pd.DataFrame(filas)


def main() -> None:
    tabla = _tabla(load_labels())
    print(f"{len(tabla)} ondas etiquetadas, {tabla['especimen'].nunique()} especímenes\n")

    l = loso(tabla)
    print("LOSO sobre especímenes de entrenamiento (RMSE de estimación, mm)")
    print(l.to_string(index=False))
    print(f"  media {l['rmse_mm'].mean():.3f} mm\n")

    c = certamen(tabla)
    print("Protocolo del certamen — sólo la mitad de ESTIMACIÓN de T7/T8")
    print(c.to_string(index=False))

    from .train import RESULTS
    RESULTS.mkdir(exist_ok=True)
    l.to_csv(RESULTS / "control_rf_loso.csv", index=False)
    c.drop(columns=["estimado", "real"]).to_csv(
        RESULTS / "control_rf_certamen.csv", index=False)


if __name__ == "__main__":
    main()
