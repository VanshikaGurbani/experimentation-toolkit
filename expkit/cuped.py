"""CUPED variance reduction (Deng et al., 2013).

Subtract the part of the outcome that a pre-experiment covariate predicts. The covariate must
not be affected by the treatment. Because the adjustment has mean zero in every arm, the effect
estimate stays unbiased while its variance shrinks by roughly the squared correlation between
the outcome and the covariate.

With a single pre-period copy of the metric this is classic CUPED. Passing several covariates
gives the regression-adjusted version. Passing a model prediction of the outcome (fitted on
data outside the experiment sample) is the CUPAC variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inference import EffectEstimate, compare_samples


@dataclass(frozen=True)
class CupedResult:
    theta: np.ndarray
    y_adjusted: np.ndarray
    correlation_r2: float  # 1 - var(adjusted) / var(raw) on the pooled sample


@dataclass(frozen=True)
class CupedComparison:
    raw: EffectEstimate
    adjusted: EffectEstimate
    theta: np.ndarray
    variance_reduction: float  # 1 - (se_adjusted / se_raw) ** 2
    sample_size_ratio: float  # n needed with CUPED relative to n without it


def _as_matrix(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != n:
        raise ValueError("covariate and outcome must have the same number of rows")
    return x


def fit_theta(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of y on the (centered) covariates."""
    y = np.asarray(y, dtype=np.float64)
    xm = _as_matrix(x, len(y))
    xc = xm - xm.mean(axis=0)
    gram = xc.T @ xc
    return np.linalg.solve(gram + 1e-12 * np.eye(gram.shape[0]), xc.T @ (y - y.mean()))


def cuped_adjust(y: np.ndarray, x: np.ndarray, theta: np.ndarray | None = None) -> CupedResult:
    """Return y minus the covariate-explained part, pooling both arms.

    Pass `theta` fitted elsewhere (for example on a held-out split) to avoid using the
    experiment sample to fit the adjustment. Never include the treatment indicator in `x`.
    """
    y = np.asarray(y, dtype=np.float64)
    xm = _as_matrix(x, len(y))
    if theta is None:
        theta = fit_theta(y, xm)
    xc = xm - xm.mean(axis=0)
    y_adj = y - xc @ np.asarray(theta, dtype=np.float64)
    r2 = 1.0 - float(y_adj.var()) / float(y.var())
    return CupedResult(np.asarray(theta), y_adj, r2)


def cuped_compare(
    y: np.ndarray,
    x: np.ndarray,
    treated: np.ndarray,
    *,
    theta: np.ndarray | None = None,
    alpha: float = 0.05,
) -> CupedComparison:
    """Estimate the treatment effect with and without CUPED on the same rows."""
    y = np.asarray(y, dtype=np.float64)
    tmask = np.asarray(treated).astype(bool)
    res = cuped_adjust(y, x, theta)
    raw = compare_samples(y[tmask], y[~tmask], alpha)
    adj = compare_samples(res.y_adjusted[tmask], res.y_adjusted[~tmask], alpha)
    ratio = (adj.se / raw.se) ** 2
    return CupedComparison(raw, adj, res.theta, 1.0 - ratio, ratio)
