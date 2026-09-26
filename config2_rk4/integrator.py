"""Predictor multi-paso por integración numérica explícita (componente c).

Qué añade sobre la Configuración 1
----------------------------------
La Configuración 1 extrapola con la **solución analítica cerrada** de Paris bajo
un rango de tensión equivalente por bloque. Eso tiene dos limitaciones que sólo
un integrador explícito puede levantar:

1. **El espectro real, segmento a segmento.** El bloque de T8 son 500 ciclos a
   90,00 MPa seguidos de 500 a 100,21 MPa. La forma cerrada los colapsa en un
   único Δσ_eq y congela la longitud de grieta dentro del bloque; RK4 integra
   cada segmento con la ``a`` que hay en ese momento.
2. **Interacción de cargas.** El fenómeno que la forma cerrada **no puede**
   representar en absoluto: tras un ciclo de sobrecarga, la zona plástica
   residual en la punta de la grieta la mantiene parcialmente cerrada y la
   propagación se **retarda** durante los ciclos siguientes. Es un efecto de
   historia, no de estado, así que exige integrar paso a paso.

Por qué importa aquí, con números
---------------------------------
Es el problema real de T8. Su tasa medida se **desacelera** un factor 2,16 al
pasar de la ventana observada a la ciega (0,273 -> 0,127 µm/ciclo), cuando Paris
con coeficiente constante exige que **acelere** al crecer la grieta. T7, bajo
amplitud constante, sí acelera (1,4x) y por eso la forma cerrada le basta.

Esa desaceleración bajo espectro variable es la firma clásica del retardo por
sobrecarga, y es justo lo que el Estado del Arte (Eje 5) señala como no
resuelto: *«el integrador tiende a subestimar el tiempo de retardo en el
crecimiento de la grieta causado por la zona plástica residual»*.

Unidades: ``a`` en metros dentro de las ecuaciones, milímetros en la interfaz;
tensiones en MPa; ``dK`` en MPa*sqrt(m); ``da/dN`` en m/ciclo.
"""
from __future__ import annotations

import numpy as np

MM_PER_M = 1000.0
_A_MIN_M, _A_MAX_M = 1e-6, 0.02


def paris_rate(a_m, C, m, d_sigma_mpa, geometry=1.0):
    """da/dN = C (Y dsigma sqrt(pi a))^m, en m/ciclo."""
    a = np.clip(a_m, _A_MIN_M, _A_MAX_M)
    return C * (geometry * d_sigma_mpa * np.sqrt(np.pi * a)) ** m


def wheeler_factor(a_m, a_overload_m, plastic_zone_m, exponent):
    """Factor de retardo de Wheeler, en [0, 1].

    Mientras la punta de la grieta sigue dentro de la zona plástica creada por
    la sobrecarga, el crecimiento se frena por un factor
    ``(r_p / (a_ol + r_ol - a))^p``; al salir de ella, vale 1 y no hay efecto.

    Es el modelo de retardo con menos parámetros libres que existe (uno solo,
    el exponente ``p``), razón por la que se elige aquí: no hay ningún
    espécimen de entrenamiento bajo amplitud variable con el que identificar
    algo más rico.
    """
    frontera = a_overload_m + plastic_zone_m
    dentro = a_m < frontera
    restante = np.maximum(frontera - a_m, 1e-12)
    razon = np.clip(plastic_zone_m / restante, 0.0, 1.0)
    return np.where(dentro, razon ** exponent, 1.0)


def integrate_block_rk4(a0_mm, target_cycles, C, m, load_block,
                        max_step: float = 250.0, geometry: float = 1.0,
                        a_max_mm: float = 20.0):
    """RK4 explícito sobre el bloque de carga real, segmento a segmento.

    ``load_block`` es ``(n_segmentos, 3)`` con ``(n_ciclos, Smax, Smin)``. La
    tasa en cada instante es la media ponderada por ciclos de las tasas de cada
    segmento evaluadas en la ``a`` actual — es decir, se resuelve la ODE con el
    espectro real en vez de con un rango equivalente congelado.

    ``a_max_mm`` es la cota de entrega de la Configuración 1, que recorta la
    extrapolación al dominio cubierto por los especímenes de entrenamiento. Es
    el mismo recorte que ``physics.propagate_mm`` aplica a la forma cerrada, y
    debe pasarse aquí para que las dos extrapolaciones sean comparables: sin él
    la fila del integrador estaría midiendo el recorte y no el integrador.
    """
    bloque = np.asarray(load_block, dtype=float)
    n_ciclos, s_max, s_min = bloque[:, 0], bloque[:, 1], bloque[:, 2]
    d_sigma = s_max - s_min
    peso = n_ciclos / n_ciclos.sum()

    def tasa(a_m):
        return sum(w * paris_rate(a_m, C, m, ds, geometry) for w, ds in zip(peso, d_sigma))

    a_max = min(a_max_mm / MM_PER_M, _A_MAX_M)
    a = np.atleast_1d(np.asarray(a0_mm, dtype=float)).astype(float) / MM_PER_M
    objetivos = np.asarray(target_cycles, dtype=float)
    salida = np.empty((len(objetivos),) + a.shape)
    ciclo = 0.0
    for i, objetivo in enumerate(objetivos):
        tramo = objetivo - ciclo
        if tramo > 0:
            pasos = max(1, int(np.ceil(tramo / max_step)))
            h = tramo / pasos
            for _ in range(pasos):
                k1 = tasa(a)
                k2 = tasa(a + 0.5 * h * k1)
                k3 = tasa(a + 0.5 * h * k2)
                k4 = tasa(a + h * k3)
                a = np.clip(a + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), _A_MIN_M, a_max)
            ciclo = objetivo
        salida[i] = a
    return salida * MM_PER_M
