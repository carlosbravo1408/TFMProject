from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ALL_SPECIMENS, empirical_growth_rates, load_curves

RESULTS = Path(__file__).resolve().parent / "results"
FIGDIR = RESULTS / "figures"

ALGO_LABEL = {"sa": "Recocido simulado", "acor": "ACO$_\\mathbb{R}$",
              "vns": "VNS", "de": "Evolución diferencial", "pso": "PSO",
              "ga": "Algoritmo genético"}


def _save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIGDIR / (name + '.png')}")


def fig_growth_curves() -> None:
    curves = load_curves(ALL_SPECIMENS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, c in curves.items():
        style = "--o" if name in ("T7", "T8") else "-o"
        ax.plot(c.cycles / 1000, c.crack_mm, style, ms=4, lw=1.4, label=name)
    ax.set_xlabel("Ciclos de fatiga (miles)")
    ax.set_ylabel("Longitud de grieta $a$ [mm]")
    ax.set_title("Curvas de crecimiento medidas (PHM Data Challenge 2019)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "curvas_crecimiento")


def fig_loglog() -> None:
    path = RESULTS / "stage_e_loglog.json"
    if not path.is_file():
        return
    e = json.loads(path.read_text())
    df = pd.DataFrame(e["points"])
    p = e["pooled"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for spec, g in df.groupby("specimen"):
        ax.loglog(g["delta_k_MPa_sqrt_m"], g["da_dn_m_per_cycle"], "o", ms=6, label=spec)
    xs = np.linspace(df["delta_k_MPa_sqrt_m"].min(), df["delta_k_MPa_sqrt_m"].max(), 50)
    ys = p["C"] * xs ** p["m"]
    ax.loglog(xs, ys, "k-", lw=2,
              label=f"Ajuste global: $m$ = {p['m']:.2f} ± {p['m_ci95']:.2f}")
    band = 10 ** (1.96 * p["residual_sd_dex"])
    ax.fill_between(xs, ys / band, ys * band, color="k", alpha=0.10,
                    label="Banda de dispersión 95 %")
    ax.set_xlabel(r"$\Delta K$ [MPa$\sqrt{\mathrm{m}}$]")
    ax.set_ylabel(r"$da/dN$ [m/ciclo]")
    ax.set_title("Regresión de Paris libre de modelo (diferencias finitas)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    _save(fig, "regresion_loglog")


def fig_algorithm_benchmark() -> None:
    path = RESULTS / "stage_a_algorithm_benchmark.csv"
    if not path.is_file():
        return
    a = pd.read_csv(path)
    a["gap_mm"] = a["objective_rmse_mm"] - a.groupby(["law", "specimen"])[
        "objective_rmse_mm"].transform("min")
    order = a.groupby("algorithm")["gap_mm"].median().sort_values().index
    labels = [ALGO_LABEL[k] for k in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].boxplot([a.loc[a.algorithm == k, "gap_mm"] for k in order],
                    tick_labels=labels, showfliers=False)
    axes[0].set_ylabel("Brecha al mejor óptimo conocido [mm]")
    axes[0].set_title("Calidad de la solución", fontsize=10)
    axes[0].grid(alpha=0.3, axis="y")
    sec = a.groupby("algorithm")["seconds"].mean().reindex(order)
    axes[1].bar(labels, sec.values, color="0.6")
    axes[1].set_ylabel("Segundos por ejecución")
    axes[1].set_title("Coste (mismo presupuesto de evaluaciones)", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    for ax in axes:
        ax.set_xticks(range(1 if ax is axes[0] else 0,
                            len(labels) + (1 if ax is axes[0] else 0)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    fig.suptitle("Comparativa de metaheurísticos en la identificación de parámetros")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "comparativa_metaheuristicos")


def fig_profile_m() -> None:
    path = RESULTS / "stage_c_profile_m.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text())
    prof = pd.DataFrame(data["profile"])
    best = prof["objective"].min()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(prof["value"], prof["objective"], "o-", color="C0")
    ax.axhline(1.05 * best, ls="--", color="C3",
               label="Umbral +5 % sobre el óptimo")
    inside = prof.loc[prof["objective"] <= 1.05 * best, "value"]
    if len(inside):
        ax.axvspan(inside.min(), inside.max(), color="C3", alpha=0.12,
                   label=f"$m \\in$ [{inside.min():.1f}, {inside.max():.1f}] indistinguibles")
    ax.set_xlabel("Exponente de Paris $m$ (fijado)")
    ax.set_ylabel("RMSE media agrupada [mm]")
    ax.set_title(f"Perfil de verosimilitud sobre $m$ (ley {data['law']})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "perfil_exponente_m")


def fig_transfer() -> None:
    from .models import LAWS
    from .pooled import fit_coefficient_only

    path = RESULTS / "stage_b_pooled_best.json"
    if not path.is_file():
        return
    shared = np.asarray(
        json.loads(path.read_text())["paris"]["shared_vector"], dtype=float
    )
    curves = load_curves(ALL_SPECIMENS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, spec in zip(axes, ("T7", "T8")):
        c = curves[spec]
        ax.plot(c.cycles / 1000, c.crack_mm, "ko-", lw=2.2, ms=7, zorder=5,
                label="Verdad de campo")
        ax.axvline(c.cycles[1] / 1000, color="0.5", ls=":", lw=1.5,
                   label="Fin de señales PZT (2 puntos)")
        for k, n_anchor in enumerate((2, 3, 4)):
            _, pred = fit_coefficient_only(LAWS["paris"], shared, c, n_points=n_anchor)
            ax.plot(c.cycles / 1000, pred, "--o", ms=3.5, lw=1.5, color=f"C{k}",
                    label=f"{n_anchor} anclajes")
        ax.set_xlabel("Ciclos de fatiga (miles)")
        ax.set_title(f"Espécimen {spec}")
        ax.set_ylim(0, 2.2 * c.final_crack_mm)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Longitud de grieta $a$ [mm]")
    fig.suptitle("Prognosis puramente física (ley de Paris, exponente congelado de T1/T3/T4/T6):\n"
                 "con 2 puntos de anclaje el integrador diverge en T8")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "transferencia_prognosis")


def main() -> None:
    print("Generando figuras...")
    fig_growth_curves()
    fig_loglog()
    fig_algorithm_benchmark()
    fig_profile_m()
    fig_transfer()


if __name__ == "__main__":
    main()
