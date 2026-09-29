from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimizeResult:
    x: np.ndarray
    fun: float
    n_evals: int
    algorithm: str
    seed: int
    history: list[float] = field(default_factory=list)


def _prepare(bounds):
    b = np.asarray(bounds, dtype=float)
    return b[:, 0], b[:, 1]


class _Budget:
    def __init__(self, f, max_evals: int):
        self.f = f
        self.max_evals = max_evals
        self.n = 0
        self.best_x = None
        self.best_f = np.inf
        self.history: list[float] = []

    @property
    def exhausted(self) -> bool:
        return self.n >= self.max_evals

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if self.exhausted:
            return np.full(len(X), np.inf)
        room = self.max_evals - self.n
        if len(X) > room:
            X = X[:room]
        vals = np.asarray(self.f(X), dtype=float)
        vals = np.where(np.isfinite(vals), vals, np.inf)
        self.n += len(X)
        i = int(np.argmin(vals))
        if vals[i] < self.best_f:
            self.best_f, self.best_x = float(vals[i]), X[i].copy()
        self.history.append(self.best_f)
        return vals


def _result(budget: _Budget, algorithm: str, seed: int) -> OptimizeResult:
    return OptimizeResult(
        x=budget.best_x, fun=budget.best_f, n_evals=budget.n,
        algorithm=algorithm, seed=seed, history=budget.history,
    )


def _pad(vals: np.ndarray, n: int) -> np.ndarray:
    if len(vals) == n:
        return vals
    out = np.full(n, np.inf)
    out[: len(vals)] = vals
    return out


def simulated_annealing(
    f, bounds, seed: int = 0, max_evals: int = 20000,
    n_chains: int = 12, t0: float = 1.0, t_end: float = 1e-4, step0: float = 0.35,
) -> OptimizeResult:
    rng = np.random.default_rng(seed)
    lo, hi = _prepare(bounds)
    d = len(lo)
    span = hi - lo
    budget = _Budget(f, max_evals)

    x = lo + rng.random((n_chains, d)) * span
    fx = _pad(budget(x), n_chains)

    n_iter = max(1, (max_evals - n_chains) // n_chains)
    cooling = (t_end / t0) ** (1.0 / n_iter)
    temp = t0
    for _ in range(n_iter):
        if budget.exhausted:
            break
        scale = step0 * (temp / t0) ** 0.5
        cand = np.clip(x + rng.normal(0.0, scale, (n_chains, d)) * span, lo, hi)
        fc = _pad(budget(cand), n_chains)
        # Normalising by the incumbent makes the schedule problem-independent.
        scale_f = max(abs(budget.best_f), 1e-12)
        delta = (fc - fx) / scale_f
        accept = (delta <= 0) | (rng.random(n_chains) < np.exp(-np.clip(delta, 0, 50) / temp))
        x = np.where(accept[:, None], cand, x)
        fx = np.where(accept, fc, fx)
        temp *= cooling
    return _result(budget, "sa", seed)


def aco_r(
    f, bounds, seed: int = 0, max_evals: int = 20000,
    archive_size: int = 20, n_ants: int = 20, q: float = 0.1, xi: float = 0.85,
) -> OptimizeResult:
    rng = np.random.default_rng(seed)
    lo, hi = _prepare(bounds)
    d = len(lo)
    budget = _Budget(f, max_evals)

    archive = lo + rng.random((archive_size, d)) * (hi - lo)
    arch_f = _pad(budget(archive), archive_size)
    order = np.argsort(arch_f)
    archive, arch_f = archive[order], arch_f[order]

    ranks = np.arange(1, archive_size + 1)
    w = np.exp(-((ranks - 1) ** 2) / (2 * (q * archive_size) ** 2)) / (q * archive_size * np.sqrt(2 * np.pi))
    p = w / w.sum()

    while not budget.exhausted:
        chosen = rng.choice(archive_size, size=n_ants, p=p)
        mu = archive[chosen]
        # Socha & Dorigo, Eq. 10.
        sigma = xi * np.abs(archive[None, :, :] - archive[chosen][:, None, :]).sum(axis=1) / (archive_size - 1)
        sigma = np.maximum(sigma, 1e-9 * (hi - lo))
        ants = np.clip(rng.normal(mu, sigma), lo, hi)
        ants_f = _pad(budget(ants), n_ants)

        merged = np.vstack([archive, ants])
        merged_f = np.concatenate([arch_f, ants_f])
        keep = np.argsort(merged_f)[:archive_size]
        archive, arch_f = merged[keep], merged_f[keep]
    return _result(budget, "acor", seed)


def vns(
    f, bounds, seed: int = 0, max_evals: int = 20000,
    k_max: int = 6, n_local: int = 12, radii=(0.02, 0.05, 0.1, 0.2, 0.35, 0.6),
) -> OptimizeResult:
    rng = np.random.default_rng(seed)
    lo, hi = _prepare(bounds)
    d = len(lo)
    span = hi - lo
    budget = _Budget(f, max_evals)
    radii = np.asarray(radii, dtype=float)[:k_max]

    x = lo + rng.random(d) * span
    fx = float(_pad(budget(x[None, :]), 1)[0])

    while not budget.exhausted:
        k = 0
        while k < len(radii) and not budget.exhausted:
            shaken = np.clip(x + rng.normal(0.0, radii[k], d) * span, lo, hi)
            y, fy = shaken, float(_pad(budget(shaken[None, :]), 1)[0])
            step = radii[0]
            stall = 0
            while stall < 4 and not budget.exhausted:
                cand = np.clip(y + rng.normal(0.0, step, (n_local, d)) * span, lo, hi)
                fc = _pad(budget(cand), n_local)
                j = int(np.argmin(fc))
                if fc[j] < fy:
                    y, fy = cand[j], float(fc[j])
                    stall = 0
                else:
                    step *= 0.5
                    stall += 1
            if fy < fx:
                x, fx, k = y, fy, 0
            else:
                k += 1
    return _result(budget, "vns", seed)


def differential_evolution(
    f, bounds, seed: int = 0, max_evals: int = 20000,
    pop_size: int = 40, F: float = 0.7, CR: float = 0.9,
) -> OptimizeResult:
    rng = np.random.default_rng(seed)
    lo, hi = _prepare(bounds)
    d = len(lo)
    budget = _Budget(f, max_evals)

    pop = lo + rng.random((pop_size, d)) * (hi - lo)
    fit = _pad(budget(pop), pop_size)

    while not budget.exhausted:
        idx = np.array([rng.choice(pop_size, 3, replace=False) for _ in range(pop_size)])
        a, b, c = pop[idx[:, 0]], pop[idx[:, 1]], pop[idx[:, 2]]
        mutant = np.clip(a + F * (b - c), lo, hi)
        cross = rng.random((pop_size, d)) < CR
        cross[np.arange(pop_size), rng.integers(0, d, pop_size)] = True
        trial = np.where(cross, mutant, pop)
        f_trial = _pad(budget(trial), pop_size)
        better = f_trial < fit
        pop = np.where(better[:, None], trial, pop)
        fit = np.where(better, f_trial, fit)
    return _result(budget, "de", seed)


def pso(
    f, bounds, seed: int = 0, max_evals: int = 20000,
    swarm_size: int = 40, w0: float = 0.9, w_end: float = 0.4, c1: float = 1.5, c2: float = 1.5,
) -> OptimizeResult:
    rng = np.random.default_rng(seed)
    lo, hi = _prepare(bounds)
    d = len(lo)
    span = hi - lo
    budget = _Budget(f, max_evals)

    x = lo + rng.random((swarm_size, d)) * span
    v = rng.uniform(-0.1, 0.1, (swarm_size, d)) * span
    fx = _pad(budget(x), swarm_size)
    p_best, p_best_f = x.copy(), fx.copy()
    g = int(np.argmin(p_best_f))
    g_best = p_best[g].copy()

    n_iter = max(1, (max_evals - swarm_size) // swarm_size)
    for it in range(n_iter):
        if budget.exhausted:
            break
        w = w0 + (w_end - w0) * it / n_iter
        r1, r2 = rng.random((swarm_size, d)), rng.random((swarm_size, d))
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        v = np.clip(v, -0.3 * span, 0.3 * span)
        x = np.clip(x + v, lo, hi)
        fx = _pad(budget(x), swarm_size)
        better = fx < p_best_f
        p_best = np.where(better[:, None], x, p_best)
        p_best_f = np.where(better, fx, p_best_f)
        g_best = p_best[int(np.argmin(p_best_f))].copy()
    return _result(budget, "pso", seed)


def genetic_algorithm(
    f, bounds, seed: int = 0, max_evals: int = 20000,
    pop_size: int = 40, n_elite: int = 2, tournament: int = 3,
    p_crossover: float = 0.9, eta_c: float = 15.0,
    p_mutation: float | None = None, eta_m: float = 20.0,
) -> OptimizeResult:
    rng = np.random.default_rng(seed)
    lo, hi = _prepare(bounds)
    d = len(lo)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    if p_mutation is None:
        p_mutation = 1.0 / d
    budget = _Budget(f, max_evals)

    pop = lo + rng.random((pop_size, d)) * (hi - lo)
    fit = _pad(budget(pop), pop_size)

    def tournament_select(n):
        picks = rng.integers(0, pop_size, (n, tournament))
        winners = picks[np.arange(n), np.argmin(fit[picks], axis=1)]
        return pop[winners]

    while not budget.exhausted:
        elite_idx = np.argsort(fit)[:n_elite]
        elite, elite_fit = pop[elite_idx].copy(), fit[elite_idx].copy()

        n_children = pop_size - n_elite
        p1 = tournament_select(n_children)
        p2 = tournament_select(n_children)

        u = rng.random((n_children, d))
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)),
                        (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
        do_cx = (rng.random((n_children, 1)) < p_crossover)
        children = np.where(do_cx, 0.5 * ((1 + beta) * p1 + (1 - beta) * p2), p1)

        u = rng.random((n_children, d))
        delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1.0,
                         1.0 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
        do_mut = rng.random((n_children, d)) < p_mutation
        children = np.where(do_mut, children + delta * span, children)

        children = np.clip(children, lo, hi)
        child_fit = _pad(budget(children), n_children)

        pop = np.vstack([elite, children])
        fit = np.concatenate([elite_fit, child_fit])

    return _result(budget, "ga", seed)


ALGORITHMS = {
    "sa": simulated_annealing,
    "acor": aco_r,
    "vns": vns,
    "de": differential_evolution,
    "pso": pso,
    "ga": genetic_algorithm,
}


def polish(f, x0, bounds, max_iter: int = 4000) -> tuple[np.ndarray, float]:
    from scipy.optimize import minimize

    lo, hi = _prepare(bounds)

    def scalar(x):
        v = float(np.asarray(f(np.atleast_2d(np.clip(x, lo, hi))), dtype=float)[0])
        return v if np.isfinite(v) else 1e12

    res = minimize(
        scalar, np.clip(np.asarray(x0, float), lo, hi), method="Nelder-Mead",
        bounds=list(zip(lo, hi)), options={"maxiter": max_iter, "xatol": 1e-10, "fatol": 1e-12},
    )
    x_best = np.clip(res.x, lo, hi)
    f_best = scalar(x_best)
    f0 = scalar(x0)
    return (x_best, f_best) if f_best <= f0 else (np.asarray(x0, float), f0)
