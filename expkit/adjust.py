"""Covariate-adjusted treatment effect (augmented inverse propensity weighting, AIPW).

When the arms are balanced this agrees with the plain difference in means. When they are not,
it corrects for the imbalance as long as either the outcome models or the propensity model is
right (the "doubly robust" property). Fit the nuisance models on data that is NOT used here
(sample splitting), then pass their predictions on the evaluation sample.

    psi1 = mu1(x) + t * (y - mu1(x)) / e(x)
    psi0 = mu0(x) + (1 - t) * (y - mu0(x)) / (1 - e(x))

The effect is the mean of psi1 - psi0, and its standard error is the standard deviation of
(psi1 - psi0) divided by sqrt(n) (the influence-function interval).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from .inference import EffectEstimate


def aipw_effect(
    y: np.ndarray,
    t: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: np.ndarray,
    alpha: float = 0.05,
    clip: tuple[float, float] = (0.01, 0.99),
) -> EffectEstimate:
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    e = np.clip(np.asarray(propensity, dtype=np.float64), *clip)
    mu0 = np.asarray(mu0, dtype=np.float64)
    mu1 = np.asarray(mu1, dtype=np.float64)
    psi1 = mu1 + t * (y - mu1) / e
    psi0 = mu0 + (1.0 - t) * (y - mu0) / (1.0 - e)
    n = len(y)
    m1, m0 = float(psi1.mean()), float(psi0.mean())
    diff_i = psi1 - psi0
    diff = float(diff_i.mean())
    se = float(diff_i.std(ddof=1) / math.sqrt(n))
    z = float(norm.ppf(1 - alpha / 2))
    p = float(2 * (1 - norm.cdf(abs(diff) / se))) if se > 0 else float("nan")
    rel = rel_lo = rel_hi = None
    if m1 > 0 and m0 > 0:
        # delta method for log(m1 / m0) using the influence functions of the two means
        se_log = float((psi1 / m1 - psi0 / m0).std(ddof=1) / math.sqrt(n))
        lr = math.log(m1 / m0)
        rel, rel_lo, rel_hi = (
            m1 / m0 - 1,
            math.exp(lr - z * se_log) - 1,
            math.exp(lr + z * se_log) - 1,
        )
    return EffectEstimate(
        n_treatment=int(t.sum()),
        n_control=int((1 - t).sum()),
        mean_treatment=m1,
        mean_control=m0,
        diff=diff,
        se=se,
        ci_low=diff - z * se,
        ci_high=diff + z * se,
        p_value=p,
        rel_lift=rel,
        rel_ci_low=rel_lo,
        rel_ci_high=rel_hi,
        alpha=alpha,
    )
