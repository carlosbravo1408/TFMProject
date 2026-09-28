from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "models"
FORMATO = 2


def ruta_de(nombre: str) -> Path:
    return (MODELOS / nombre).with_suffix(".pt")


def _estado(modelo):
    return {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}


def guardar(nombre: str, cfg, modelos, segundos: float | None = None,
            entrenados_sobre=None, notas: str | None = None) -> Path:
    ruta = ruta_de(nombre)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "formato": FORMATO,
        "nombre": nombre,
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "config": asdict(cfg),
        "n_features": int(modelos[0].feature_mean.numel()),
        "segundos": segundos,
        "entrenados_sobre": list(entrenados_sobre) if entrenados_sobre else None,
        "notas": notas,
        "pesos": [_estado(m) for m in modelos],
    }, ruta)
    return ruta


def cargar(nombre: str):
    from config1_cnn_pinn.train import Config, build_model

    registro = torch.load(ruta_de(nombre), map_location="cpu", weights_only=False)
    cfg = Config(**registro["config"])
    modelos = []
    for estado in registro["pesos"]:
        m = build_model(cfg, n_features=registro["n_features"])
        m.load_state_dict(estado)
        m.eval()
        modelos.append(m)
    return modelos, registro.get("segundos"), registro


def cargar_o_entrenar(nombre: str, cfg, entrenar, *, forzar: bool = False,
                      entrenados_sobre=None, notas: str | None = None):
    if ruta_de(nombre).exists() and not forzar:
        modelos, segundos, _ = cargar(nombre)
        return modelos, segundos, "cargado de models/"
    t0 = time.perf_counter()
    modelos = entrenar()
    segundos = time.perf_counter() - t0
    guardar(nombre, cfg, modelos, segundos, entrenados_sobre, notas)
    return modelos, segundos, "entrenado ahora"


def inventario() -> str:
    if not MODELOS.is_dir():
        return "models/ todavía no existe."
    ficheros = sorted(MODELOS.rglob("*.pt"))
    if not ficheros:
        return "models/ está vacío."
    lineas = [f"{len(ficheros)} conjuntos en models/"]
    for f in ficheros:
        r = torch.load(f, map_location="cpu", weights_only=False)
        seg = f"{r['segundos']:6.0f} s" if r.get("segundos") else "     n/d"
        lineas.append(f"  {r['nombre']:34s} {len(r['pesos']):2d} modelos  "
                      f"{r['config']['architecture']:20s} {seg}  "
                      f"{f.stat().st_size / 1e6:5.1f} MB")
    return "\n".join(lineas)
