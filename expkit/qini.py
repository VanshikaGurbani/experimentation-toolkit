"""Qini curve and uplift evaluation for randomized experiments.

Rank users by a model score (highest predicted uplift first). For the top fraction x of users,

    Q(x) = n(x) * (outcome rate among treated - outcome rate among control)

is the estimated number of extra outcomes if only those users were treated. We divide by the
total population, so Q(1) equals the overall average treatment effect (ATE) and a random
ranking follows the straight line x * ATE. The Qini coefficient is the area between the model
curve and that line, in units of "outcomes per user".

The curve is evaluated on `n_bins` equal-sized slices of the ranking. Uncertainty comes from a
Poisson bootstrap over users, which re-weights every user by an independent Poisson(1) draw.
All models in one bootstrap share the same weights, so differences between models are paired.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QiniCurve:
    x: np.ndarray  # fraction of population targeted, 0..1
    q: np.ndarray  # incremental outcomes per user in the population
    ate: float
    coefficient: float

    def capture(self, fraction: float) -> float:
        """Share of the total incremental outcome captured by targeting the top `fraction`."""
        return float(np.interp(fraction, self.x, self.q) / self.ate)


def _bin_edges(n: int, n_bins: int) -> np.ndarray:
    return np.linspace(0, n, n_bins + 1).astype(np.int64)


def _curve_from_bin_sums(
    n_t: np.ndarray, n_c: np.ndarray, y_t: np.ndarray, y_c: np.ndarray, sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build (x, q, ate, coefficient) from per-bin sums (not yet cumulative)."""
    cn_t, cn_c = np.cumsum(n_t, dtype=np.float64), np.cumsum(n_c, dtype=np.float64)
    cy_t, cy_c = np.cumsum(y_t, dtype=np.float64), np.cumsum(y_c, dtype=np.float64)
    cn = np.cumsum(sizes, dtype=np.float64)
    total = cn[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(cn_t > 0, cy_t / cn_t, 0.0) - np.where(cn_c > 0, cy_c / cn_c, 0.0)
    q = np.concatenate([[0.0], cn * lift / total])
    x = np.concatenate([[0.0], cn / total])
    ate = float(q[-1])
    coefficient = float(np.trapezoid(q, x) - 0.5 * ate)
    return x, q, ate, coefficient


def _prepare(y: np.ndarray, t: np.ndarray, score: np.ndarray):
    order = np.argsort(-np.asarray(score), kind="stable")
    t_s = np.asarray(t, dtype=np.float32)[order]
    y_s = np.asarray(y, dtype=np.float32)[order]
    return order, t_s, y_s


def qini_curve(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    n_bins: int = 100,
    weights: np.ndarray | None = None,
) -> QiniCurve:
    """Qini curve of `score` against outcome `y` and treatment indicator `t`.

    `weights` are optional per-user weights, for example inverse propensity weights when the
    arms are not perfectly balanced. The ranking is cut into bins of equal user counts.
    """
    n = len(y)
    if not (len(t) == len(score) == n) or (weights is not None and len(weights) != n):
        raise ValueError("y, t, score and weights must have the same length")
    order, t_s, y_s = _prepare(y, t, score)
    edges = _bin_edges(n, n_bins)
    starts = edges[:-1]
    if weights is None:
        sizes = np.diff(edges).astype(np.float64)
        n_t = np.add.reduceat(t_s, starts)
        y_t = np.add.reduceat(t_s * y_s, starts)
        n_c = sizes - n_t
        y_c = np.add.reduceat(y_s, starts) - y_t
    else:
        w = np.asarray(weights, dtype=np.float64)[order]
        sizes = np.add.reduceat(w, starts)
        n_t = np.add.reduceat(w * t_s, starts)
        n_c = sizes - n_t
        y_t = np.add.reduceat(w * t_s * y_s, starts)
        y_c = np.add.reduceat(w * y_s, starts) - y_t
    x, q, ate, coef = _curve_from_bin_sums(n_t, n_c, y_t, y_c, sizes)
    return QiniCurve(x, q, ate, coef)


def qini_bootstrap(
    y: np.ndarray,
    t: np.ndarray,
    scores: dict[str, np.ndarray],
    *,
    n_boot: int = 200,
    n_bins: int = 100,
    capture_fractions: tuple[float, ...] = (0.1, 0.3),
    seed: int = 0,
) -> dict[str, dict[str, np.ndarray]]:
    """Poisson-bootstrap distributions of the Qini coefficient and capture shares.

    Returns {model: {"coefficient": array, "capture@0.1": array, ...}} plus the key "_ate".
    Arrays are aligned across models, so model differences can be taken replicate by replicate.
    """
    n = len(y)
    edges = _bin_edges(n, n_bins)
    starts = edges[:-1]
    prepared = {}
    for name, s in scores.items():
        order, t_s, y_s = _prepare(y, t, s)
        prepared[name] = (order, t_s, 1.0 - t_s, t_s * y_s, (1.0 - t_s) * y_s)
    idx = {f: int(round(f * n_bins)) for f in capture_fractions}
    out: dict[str, dict[str, np.ndarray]] = {
        name: {"coefficient": np.empty(n_boot), **{f"capture@{f}": np.empty(n_boot) for f in idx}}
        for name in scores
    }
    out["_ate"] = {"ate": np.empty(n_boot)}
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        w = rng.poisson(1.0, n).astype(np.float32)
        for name, (order, t_s, c_s, ty_s, cy_s) in prepared.items():
            ws = w[order]
            n_t = np.add.reduceat(ws * t_s, starts)
            n_c = np.add.reduceat(ws * c_s, starts)
            y_t = np.add.reduceat(ws * ty_s, starts)
            y_c = np.add.reduceat(ws * cy_s, starts)
            sizes = np.add.reduceat(ws, starts)
            x, q, ate, coef = _curve_from_bin_sums(n_t, n_c, y_t, y_c, sizes)
            out[name]["coefficient"][b] = coef
            for f, i in idx.items():
                out[name][f"capture@{f}"][b] = q[i] / ate if ate != 0 else np.nan
        out["_ate"]["ate"][b] = ate
    return out


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo = (1 - level) / 2
    return float(np.nanquantile(values, lo)), float(np.nanquantile(values, 1 - lo))
