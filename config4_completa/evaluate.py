from __future__ import annotations

import pandas as pd

# (extrapolation engine, latent summary)
COMBINACIONES = {
    "Config 1  (a+b)":     ("analítica", "media"),
    "Config 2  (a+b+c)":   ("paso a paso", "media"),
    "Config 3  (a+b+d)":   ("analítica", "ponderado"),
    "Config 4  (a+b+c+d)": ("paso a paso", "ponderado"),
}

MOTORES = ("analítica", "paso a paso", "ponderado")


def tabla_contrastes(detalle: dict, especimen: str) -> pd.DataFrame:
    filas: dict[str, dict[str, float]] = {}
    for etiqueta, (motor, resumen) in COMBINACIONES.items():
        filas.setdefault(motor, {})[resumen] = detalle[etiqueta][especimen]["penalizacion"]
    return pd.DataFrame(filas).T[["media", "ponderado"]]


def aportaciones(detalle: dict, especimen: str) -> pd.DataFrame:
    # Each component's effect is the mean of its two simple contrasts; the joint effect
    # is half their difference. Negative means the penalty drops.
    t = tabla_contrastes(detalle, especimen)
    c_media = t.loc["paso a paso", "media"] - t.loc["analítica", "media"]
    c_ponderado = t.loc["paso a paso", "ponderado"] - t.loc["analítica", "ponderado"]
    d_analitica = t.loc["analítica", "ponderado"] - t.loc["analítica", "media"]
    d_paso = t.loc["paso a paso", "ponderado"] - t.loc["paso a paso", "media"]
    return pd.DataFrame([
        {"término": "(c) integración paso a paso",
         "valor": 0.5 * (c_media + c_ponderado),
         "detalle": f"{c_media:+.3f} con resumen por media, "
                    f"{c_ponderado:+.3f} con resumen ponderado"},
        {"término": "(d) ponderación aprendida",
         "valor": 0.5 * (d_analitica + d_paso),
         "detalle": f"{d_analitica:+.3f} con extrapolación analítica, "
                    f"{d_paso:+.3f} con integración paso a paso"},
        {"término": "aportación conjunta de (c) y (d)",
         "valor": 0.5 * (c_ponderado - c_media),
         "detalle": "mitad de la diferencia entre los dos contrastes de (c)"},
    ])
