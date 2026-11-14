"""Confidence intervals for a treatment effect on a mean or a rate."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class EffectEstimate:
    n_treatment: int
    n_control: int
    mean_treatment: float
    mean_control: float
    diff: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    rel_lift: float | None
    rel_ci_low: float | None
    rel_ci_high: float | None
    alpha: float

    def as_dict(self) -> dict:
        return asdict(self)


def compare_means(
    mean_t: float,
    var_t: float,
    n_t: int,
    mean_c: float,
    var_c: float,
    n_c: int,
    alpha: float = 0.05,
) -> EffectEstimate:
    """Welch-style normal interval for the difference in means from summary statistics.

    The relative lift interval uses the delta method on log(mean_t / mean_c) and is only
    reported when both means are clearly positive (see below).
    """
    if n_t < 2 or n_c < 2:
        raise ValueError("need at least 2 observations per arm")
    z = float(norm.ppf(1 - alpha / 2))
    diff = mean_t - mean_c
    se = math.sqrt(var_t / n_t + var_c / n_c)
    p = float(2 * (1 - norm.cdf(abs(diff) / se))) if se > 0 else float("nan")
    rel = rel_lo = rel_hi = None
    # The delta method on log(mean ratio) breaks down when a mean is not clearly above zero,
    # so the relative interval is only reported when both means exceed 3 standard errors.
    clearly_positive = mean_t > 3 * math.sqrt(var_t / n_t) and mean_c > 3 * math.sqrt(var_c / n_c)
    if clearly_positive:
        rel = mean_t / mean_c - 1
        se_log = math.sqrt(var_t / (n_t * mean_t**2) + var_c / (n_c * mean_c**2))
        log_ratio = math.log(mean_t / mean_c)
        rel_lo = math.exp(log_ratio - z * se_log) - 1
        rel_hi = math.exp(log_ratio + z * se_log) - 1
    return EffectEstimate(
        n_t,
        n_c,
        mean_t,
        mean_c,
        diff,
        se,
        diff - z * se,
        diff + z * se,
        p,
        rel,
        rel_lo,
        rel_hi,
        alpha,
    )


def compare_proportions(
    x_t: int, n_t: int, x_c: int, n_c: int, alpha: float = 0.05
) -> EffectEstimate:
    """Unpooled (Wald) interval for the difference in two proportions from counts."""
    p_t, p_c = x_t / n_t, x_c / n_c
    return compare_means(p_t, p_t * (1 - p_t), n_t, p_c, p_c * (1 - p_c), n_c, alpha)


def compare_samples(y_t: np.ndarray, y_c: np.ndarray, alpha: float = 0.05) -> EffectEstimate:
    """Difference in means from two raw samples (any numeric outcome)."""
    y_t = np.asarray(y_t, dtype=float)
    y_c = np.asarray(y_c, dtype=float)
    return compare_means(
        float(y_t.mean()),
        float(y_t.var(ddof=1)),
        len(y_t),
        float(y_c.mean()),
        float(y_c.var(ddof=1)),
        len(y_c),
        alpha,
    )
