"""Retardo por interacción de cargas (modelo de Wheeler) para el bloque de T8.

Por qué este modelo y no otro
-----------------------------
El diagnóstico previo dejó una firma clara: la tasa medida de T8 dibuja una
**U** — rápida (0,273 µm/ciclo), se frena hasta un mínimo (0,055) y se recupera
(0,245) — mientras la ley de Paris con coeficiente constante exige crecimiento
monótonamente acelerado. Frenado transitorio seguido de recuperación es la
firma del **retardo por sobrecarga**: la zona plástica que deja el ciclo de
mayor amplitud mantiene la grieta parcialmente cerrada mientras la punta avanza
dentro de ella, y el efecto desaparece al salir.

De los modelos de retardo disponibles se elige el de **Wheeler (1972)** porque
tiene **un solo parámetro libre** (el exponente ``p``). No hay ningún espécimen
de entrenamiento bajo amplitud variable con el que identificar algo más rico:
cualquier modelo con más grados de libertad sería inidentificable por
construcción.

Formulación
-----------
Zona plástica en tensión plana (chapa de 1,6 mm), con ``alpha = 2``:

    r_y = (1 / (alpha * pi)) * (K_max / sigma_y)^2

Tras una sobrecarga a longitud ``a_ol`` que crea ``r_ol``, mientras la punta
siga dentro de esa zona:

    phi = ( r_y / (a_ol + r_ol - a) )^p        si  a + r_y < a_ol + r_ol
    phi = 1                                     en caso contrario

y la tasa retardada es ``phi * da/dN``.

Parámetros y su procedencia
---------------------------
``sigma_y = 345 MPa`` es el límite elástico convencional del 2024-T3 en chapa
fina (valor de manual).

``p`` **no es determinable con estos datos**: sólo deja huella bajo amplitud
variable, y el único espécimen de ese régimen es T8, el de evaluación. Los seis
de entrenamiento son de amplitud constante y en ellos el exponente no altera la
curva. El valor ``p = 1,43`` que usa la tabla de ablación procede de la
literatura sobre un acero (Wheeler, 1972) y queda dentro del intervalo publicado
para otras aleaciones de aluminio, 0,94-2,19 en 5083-O y 1,36-1,98 en 6061-T651
segun Sheu, Song y Hwang (1995). **No hay valor publicado para el 2024-T3.**
En consecuencia la variante que lo emplea es un diagnóstico, no un componente
predictivo: mide cuánto llegaría a explicar el mecanismo si estuviera bien
parametrizado. El barrido de
``sensitivity.py`` recorre el exponente con ese único fin.

Cota que conviene tener presente
--------------------------------
El bloque de T8 es 500 ciclos a 90,00 MPa + 500 a 100,21 MPa. Los segundos
**son** la sobrecarga, así que nunca están retardados: aportan su tasa completa
pase lo que pase. En el límite de frenado total de la primera mitad, la tasa
media del bloque sólo puede caer hasta la mitad de la contribución no retardada
— es decir, Wheeler tiene un techo de reducción de 1,80x (con m = 2,00). El
coeficiente de T8 necesita bajar 2,3x (0,363 dex). Está en el mismo orden, pero
el modelo no puede cubrirlo entero; medir cuánto cubre es el objeto de este
módulo. (No comparar el techo con el frenado medido de T8, 2,16x o 1,74x según
la ventana: es un cociente de tasas observadas, no la corrección de C.)
"""
from __future__ import annotations

import numpy as np

MM_PER_M = 1000.0

# Límite elástico convencional del Al 2024-T3 en chapa fina, MPa (valor de manual).
YIELD_STRENGTH_MPA = 345.0
# Constrenimiento: 2 = tensión plana (chapa de 1,6 mm); 6 sería deformación plana.
PLANE_STRESS_ALPHA = 2.0


def plastic_zone_m(k_max, sigma_y=YIELD_STRENGTH_MPA, alpha=PLANE_STRESS_ALPHA):
    """Radio de la zona plástica monótona, en metros. ``k_max`` en MPa*sqrt(m)."""
    return (1.0 / (alpha * np.pi)) * (np.asarray(k_max) / sigma_y) ** 2


def k_max(a_m, s_max_mpa, geometry=1.0):
    return geometry * s_max_mpa * np.sqrt(np.pi * np.clip(a_m, 1e-9, None))


def wheeler_phi(a_m, a_ol_m, r_ol_m, r_y_m, exponent):
    """Factor de retardo phi en [0, 1]."""
    frontera = a_ol_m + r_ol_m
    restante = np.maximum(frontera - a_m, 1e-12)
    dentro = (a_m + r_y_m) < frontera
    return np.where(dentro, np.clip(r_y_m / restante, 0.0, 1.0) ** exponent, 1.0)


def block_rate_with_retardation(a_m, C, m, load_block, exponent,
                                sigma_y=YIELD_STRENGTH_MPA, geometry=1.0):
    """Tasa media por ciclo sobre un bloque, con retardo de Wheeler.

    El segmento de mayor ``Smax`` del bloque es la sobrecarga: define la zona
    plástica que retarda a los demás. Como el avance de la grieta dentro de un
    bloque (~0,05 mm) es despreciable frente al tamaño de la zona plástica, se
    evalúa la sobrecarga en la ``a`` actual.
    """
    bloque = np.asarray(load_block, dtype=float)
    n_ciclos, s_max, s_min = bloque[:, 0], bloque[:, 1], bloque[:, 2]
    peso = n_ciclos / n_ciclos.sum()

    i_ol = int(np.argmax(s_max))
    r_ol = plastic_zone_m(k_max(a_m, s_max[i_ol], geometry), sigma_y)

    total = 0.0
    for j, (w, smax_j, smin_j) in enumerate(zip(peso, s_max, s_min)):
        d_sigma = smax_j - smin_j
        tasa = C * (geometry * d_sigma * np.sqrt(np.pi * np.clip(a_m, 1e-9, None))) ** m
        if j == i_ol:
            phi = 1.0                      # el propio ciclo de sobrecarga no se retarda
        else:
            r_y = plastic_zone_m(k_max(a_m, smax_j, geometry), sigma_y)
            phi = wheeler_phi(a_m, a_m, r_ol, r_y, exponent)
        total = total + w * phi * tasa
    return total


def integrate_with_retardation(a0_mm, target_cycles, C, m, load_block, exponent,
                               max_step: float = 250.0, sigma_y=YIELD_STRENGTH_MPA,
                               geometry: float = 1.0, a_max_mm: float = 20.0):
    """RK4 explícito sobre el bloque real con retardo de Wheeler."""
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
    """Desglose legible del efecto a una longitud de grieta dada."""
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
