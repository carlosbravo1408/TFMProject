from __future__ import annotations

FINAL_CRACK_MM = {"T7": 7.22, "T8": 5.52}

PUESTOS = ("1.º Youn et al. (2019)", "2.º Kong et al. (2020)", "3.º Rao et al. (2020)")

# No paper publishes per-specimen penalties: the 3rd gives Table A cycle by cycle,
# and the 1st and 2nd are rebuilt by applying the official formula to their tables.
PUBLICADO = {
    "1.º Youn et al. (2019)": {
        "T7": 3.415, "T8": 3.944, "total": 7.359,
        "origen": "derivada de sus Tablas 5, 7, 8 y 11",
    },
    "2.º Kong et al. (2020)": {
        "T7": 3.543, "T8": 4.088, "total": 7.631,
        "origen": "derivada de su Tabla 3",
    },
    "3.º Rao et al. (2020)": {
        "T7": 9.229, "T8": 6.864, "total": 16.093,
        "origen": "derivada de su Tabla A",
    },
}

REPLICA = {
    "1.º Youn et al. (2019)": {"T7": 22.708, "T8": 91.600, "total": 114.309},
    "2.º Kong et al. (2020)": {"T7": 83.252, "T8": 148.344, "total": 231.596},
    "3.º Rao et al. (2020)": {"T7": 215.123, "T8": 25.787, "total": 240.910},
}


def tabla_comparativa(puesto: str, propia: dict[str, float] | None = None):
    import pandas as pd

    if puesto not in PUBLICADO:
        raise KeyError(f"{puesto!r} no es uno de {PUESTOS}")

    filas = [
        {"entrada": k, "T7": v["T7"], "T8": v["T8"], "total": v["total"],
         "naturaleza": f"publicada, {v['origen']}"}
        for k, v in PUBLICADO.items()
    ]
    for k, v in REPLICA.items():
        propio = k == puesto
        valores = dict(v)
        if propio and propia is not None:
            valores = {"T7": round(propia["T7"], 3), "T8": round(propia["T8"], 3),
                       "total": round(propia["T7"] + propia["T8"], 3)}
        filas.append({
            "entrada": f"{k}, réplica",
            "T7": round(valores["T7"], 3),
            "T8": round(valores["T8"], 3),
            "total": round(valores["total"], 3),
            "naturaleza": "reproducible, este cuaderno" if propio else "reproducible, su cuaderno",
        })
    return pd.DataFrame(filas).set_index("entrada")


def desviacion(puesto: str, propia: dict[str, float]) -> float:
    return abs((propia["T7"] + propia["T8"]) - REPLICA[puesto]["total"])
