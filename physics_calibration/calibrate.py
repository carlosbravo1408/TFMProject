from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    ALL_SPECIMENS, CALIBRATION_SPECIMENS, empirical_growth_rates, load_curves,
)
from .metaheuristics import ALGORITHMS, polish
from .models import LAWS
from .objectives import OBJECTIVES, evaluate_all, phm_penalty
from .pooled import ProfiledPooledProblem, fit_coefficient_only, profile_exponent

RESULTS = Path(__file__).resolve().parent / "results"

STUDY_LAWS = ("paris", "walker", "walker_closure", "forman", "paris_hole", "walker_hole", "forman_hole")

# Stage A runs every law x specimen x algorithm x seed at full budget; these
# three cover the 2, 3 and 4-parameter cases.
BENCHMARK_LAWS = ("paris", "walker", "paris_hole")

# T2 and T5 have only two non-zero measurements each.
FITTABLE = ("T1", "T3", "T4", "T6", "T7", "T8")

WINNER_SCORES = {"1st (Youn, SVR+trans-fitting)": 7.36,
                 "2nd (Kong, RF+PF / Walker+MC)": 7.63,
                 "3rd (Rao, linear ens.+Paris)": 16.14}


def stage_a_algorithm_benchmark(curves, seeds, max_evals, laws=BENCHMARK_LAWS) -> pd.DataFrame:
    rows = []
    for law_name in laws:
        law = LAWS[law_name]
        for spec in FITTABLE:
            curve = curves[spec]
            f = OBJECTIVES["rmse"](law, curve)
            for algo_name, algo in ALGORITHMS.items():
                for seed in seeds:
                    t0 = time.perf_counter()
                    res = algo(f, law.bounds, seed=seed, max_evals=max_evals)
                    dt = time.perf_counter() - t0
                    metrics = evaluate_all(law, res.x, curve)
                    rows.append(
                        {
                            "law": law_name, "specimen": spec, "algorithm": algo_name,
                            "seed": seed, "objective_rmse_mm": res.fun,
                            "phm_penalty": metrics["phm_penalty"],
                            "max_abs_err_mm": metrics["max_abs_err_mm"],
                            "n_evals": res.n_evals, "seconds": dt,
                            **{f"x{i}": v for i, v in enumerate(res.x)},
                        }
                    )
    return pd.DataFrame(rows)


def stage_b_pooled(curves, seeds, max_evals, laws=STUDY_LAWS, loss="rmse"):
    train = tuple(curves[s] for s in CALIBRATION_SPECIMENS)
    runs, best = [], {}
    for law_name in laws:
        problem = ProfiledPooledProblem(LAWS[law_name], train, loss=loss)
        f = problem.objective()
        best_res = None
        for algo_name, algo in ALGORITHMS.items():
            for seed in seeds:
                t0 = time.perf_counter()
                res = algo(f, problem.bounds, seed=seed, max_evals=max_evals)
                runs.append(
                    {
                        "law": law_name, "algorithm": algo_name, "seed": seed,
                        "objective_mean_rmse_mm": res.fun,
                        "seconds": time.perf_counter() - t0,
                    }
                )
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
        x_best, f_best = polish(f, best_res.x, problem.bounds)
        info = problem.describe(x_best)
        log_c = np.array(list(info["log10_coefficients"].values()))
        best[law_name] = {
            "shared_vector": np.asarray(x_best).tolist(),
            "algorithm": best_res.algorithm, "seed": best_res.seed,
            "objective_mean_rmse_mm": float(f_best),
            "objective_before_polish": float(best_res.fun),
            "log10_coefficient_mean": float(log_c.mean()),
            "log10_coefficient_std": float(log_c.std(ddof=1)),
            **info,
        }
    return pd.DataFrame(runs), best


def stage_c_profile(curves, law_name, max_evals):
    train = tuple(curves[s] for s in CALIBRATION_SPECIMENS)
    problem = ProfiledPooledProblem(LAWS[law_name], train, loss="rmse")
    m_lo, m_hi = problem.bounds[0]
    values = np.linspace(m_lo, min(m_hi, 7.0), 13)
    return profile_exponent(
        problem, ALGORITHMS["de"], index=0, values=values, seed=0, max_evals=max_evals
    )


def stage_d_transfer(curves, best_b, laws=STUDY_LAWS):
    rows = []
    for law_name in laws:
        law = LAWS[law_name]
        shared = np.asarray(best_b[law_name]["shared_vector"], dtype=float)
        for spec in FITTABLE:
            curve = curves[spec]
            n_anchor = min(curve.n_observed, 2)
            log_c, pred = fit_coefficient_only(law, shared, curve, n_points=n_anchor)
            out = slice(n_anchor, None)
            true = curve.crack_mm
            rows.append(
                {
                    "law": law_name, "specimen": spec, "n_anchor": n_anchor,
                    "log10_coefficient": log_c,
                    "rmse_all_mm": float(np.sqrt(np.mean((pred - true) ** 2))),
                    "rmse_holdout_mm": float(np.sqrt(np.mean((pred[out] - true[out]) ** 2))),
                    "phm_penalty_all": float(phm_penalty(pred[None, :], true, curve.final_crack_mm)[0]),
                    "phm_penalty_holdout": float(
                        phm_penalty(pred[None, out], true[out], curve.final_crack_mm)[0]
                    ),
                    "runaway": bool(pred[-1] > 2.0 * curve.final_crack_mm),
                    "pred_mm": np.round(pred, 3).tolist(), "true_mm": true.tolist(),
                }
            )
    return pd.DataFrame(rows)


def stage_e_loglog(curves) -> dict:
    df = pd.concat(
        [empirical_growth_rates(c) for c in curves.values() if len(c.crack_mm) > 2]
    )
    df = df[df["da_dn_m_per_cycle"] > 0]
    x = np.log10(df["delta_k_MPa_sqrt_m"].to_numpy())
    y = np.log10(df["da_dn_m_per_cycle"].to_numpy())
    m, b = np.polyfit(x, y, 1)
    resid = y - (m * x + b)
    n = len(x)
    s = resid.std(ddof=2)
    se_m = s / np.sqrt(np.sum((x - x.mean()) ** 2))
    per_spec = {}
    for spec, g in df.groupby("specimen"):
        if len(g) >= 3:
            mi, bi = np.polyfit(
                np.log10(g["delta_k_MPa_sqrt_m"]), np.log10(g["da_dn_m_per_cycle"]), 1
            )
            per_spec[spec] = {"m": float(mi), "C": float(10 ** bi), "n": int(len(g))}
    return {
        "pooled": {
            "m": float(m), "m_ci95": float(1.96 * se_m), "C": float(10 ** b),
            "r2": float(1 - resid.var() / y.var()), "n": int(n),
            "residual_sd_dex": float(s),
            "delta_k_min": float(df["delta_k_MPa_sqrt_m"].min()),
            "delta_k_max": float(df["delta_k_MPa_sqrt_m"].max()),
        },
        "per_specimen": per_spec,
        "points": df.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small budget smoke run")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--evals", type=int, default=20000)
    parser.add_argument("--pooled-evals", type=int, default=1200)
    parser.add_argument("--skip-a", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.evals, args.pooled_evals = 3, 3000, 400

    RESULTS.mkdir(exist_ok=True)
    seeds = list(range(args.seeds))
    curves = load_curves(ALL_SPECIMENS)

    if not args.skip_a:
        print(f"== Stage A: per-specimen algorithm benchmark "
              f"({args.seeds} seeds x {args.evals} evals) ==")
        t0 = time.perf_counter()
        a = stage_a_algorithm_benchmark(curves, seeds, args.evals)
        a.to_csv(RESULTS / "stage_a_algorithm_benchmark.csv", index=False)
        # Absolute RMSE mixes specimens of different difficulty.
        a["gap_mm"] = a["objective_rmse_mm"] - a.groupby(["law", "specimen"])[
            "objective_rmse_mm"].transform("min")
        print(
            a.groupby("algorithm")
            .agg(wins=("gap_mm", lambda v: int((v < 1e-6).sum())),
                 median_gap=("gap_mm", "median"), q90_gap=("gap_mm", lambda v: v.quantile(0.9)),
                 worst_gap=("gap_mm", "max"), sec_per_run=("seconds", "mean"))
            .round(5).to_string()
        )
        print(f"[{time.perf_counter() - t0:.1f}s]\n")

    print(f"== Stage B: pooled identification on {', '.join(CALIBRATION_SPECIMENS)} "
          f"({args.seeds} seeds x {args.pooled_evals} evals) ==")
    t0 = time.perf_counter()
    b_runs, b_best = stage_b_pooled(curves, seeds, args.pooled_evals)
    b_runs.to_csv(RESULTS / "stage_b_pooled_runs.csv", index=False)
    (RESULTS / "stage_b_pooled_best.json").write_text(json.dumps(b_best, indent=2))
    print(
        b_runs.groupby("algorithm")
        .agg(median_obj=("objective_mean_rmse_mm", "median"),
             worst_obj=("objective_mean_rmse_mm", "max"), sec=("seconds", "mean"))
        .round(4).to_string()
    )
    print()
    for law_name, info in sorted(b_best.items(), key=lambda kv: kv[1]["objective_mean_rmse_mm"]):
        shared = {k: round(v, 4) for k, v in info["shared"].items()}
        print(f"  {law_name:16s} mean RMSE {info['objective_mean_rmse_mm']:.4f} mm  "
              f"shared={shared}  log10C = {info['log10_coefficient_mean']:.3f} "
              f"+/- {info['log10_coefficient_std']:.3f}  ({info['algorithm']}, seed {info['seed']})")
    print(f"[{time.perf_counter() - t0:.1f}s]\n")

    winner = min(b_best, key=lambda k: b_best[k]["objective_mean_rmse_mm"])
    print(f"== Stage C: profile likelihood over m (best pooled law: {winner}) ==")
    t0 = time.perf_counter()
    c = stage_c_profile(curves, winner, max_evals=max(400, args.pooled_evals))
    (RESULTS / "stage_c_profile_m.json").write_text(
        json.dumps({"law": winner, "profile": c}, indent=2)
    )
    best_obj = min(row["objective"] for row in c)
    for row in c:
        flag = " <-- within 5%" if row["objective"] <= 1.05 * best_obj else ""
        print(f"  m = {row['value']:.2f}  ->  mean RMSE {row['objective']:.4f} mm{flag}")
    print(f"[{time.perf_counter() - t0:.1f}s]\n")

    print("== Stage D: frozen-exponent transfer (3 anchor points, rest predicted) ==")
    d = stage_d_transfer(curves, b_best)
    d.drop(columns=["pred_mm", "true_mm"]).to_csv(RESULTS / "stage_d_transfer.csv", index=False)
    (RESULTS / "stage_d_transfer_curves.json").write_text(
        d[["law", "specimen", "pred_mm", "true_mm"]].to_json(orient="records", indent=2)
    )
    print(d.drop(columns=["pred_mm", "true_mm"]).round(4).to_string(index=False))
    val = d[d.specimen.isin(("T7", "T8"))].groupby("law")["phm_penalty_all"].sum()
    print("\n  Sum of T7+T8 official penalty, physics-only:")
    for law_name, v in val.sort_values().items():
        print(f"    {law_name:16s} {v:7.2f}")
    for label, v in WINNER_SCORES.items():
        print(f"    {label:16s} {v:7.2f}  (published)")
    print()

    print("== Stage E: model-free log-log regression ==")
    e = stage_e_loglog(curves)
    (RESULTS / "stage_e_loglog.json").write_text(json.dumps(e, indent=2))
    p = e["pooled"]
    print(f"  pooled: m = {p['m']:.3f} +/- {p['m_ci95']:.3f} (95% CI), C = {p['C']:.3e}, "
          f"R2 = {p['r2']:.3f}, n = {p['n']}, residual sd = {p['residual_sd_dex']:.3f} dex")
    print(f"  dK spans {p['delta_k_min']:.2f}-{p['delta_k_max']:.2f} MPa*sqrt(m) "
          f"({np.log10(p['delta_k_max'] / p['delta_k_min']):.2f} decades)")
    for spec, v in e["per_specimen"].items():
        print(f"    {spec}: m = {v['m']:6.2f}  C = {v['C']:.3e}  (n = {v['n']})")


if __name__ == "__main__":
    main()
