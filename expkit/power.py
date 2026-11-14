"""Power and sample-size calculations for two-arm experiments.

Binary outcomes use the Fleiss formula (pooled variance under the null, unpooled under the
alternative, no continuity correction). Continuous outcomes use the normal approximation.
`ratio` is the treatment-to-control allocation, so ratio=1 is a 50/50 split and ratio=85/15
reproduces the Criteo design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class SampleSize:
    n_control: int
    n_treatment: int
    baseline: float
    treated_rate: float
    absolute_effect: float
    relative_effect: float
    alpha: float
    power: float
    ratio: float

    @property
    def total(self) -> int:
        return self.n_control + self.n_treatment


def _z_alpha(alpha: float, two_sided: bool) -> float:
    return float(norm.ppf(1 - alpha / 2)) if two_sided else float(norm.ppf(1 - alpha))


def _check_rate(p: float, name: str) -> None:
    if not 0 < p < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {p}")


def _check_common(alpha: float, power: float, ratio: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if ratio <= 0:
        raise ValueError("ratio must be positive")


def sample_size_proportions(
    baseline: float,
    mde: float,
    *,
    relative: bool = False,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> SampleSize:
    """Sample size for a two-sample test of proportions.

    `mde` is the minimum detectable effect, absolute (0.01 means +1 percentage point) unless
    `relative=True` (0.10 means +10% of the baseline).
    """
    _check_rate(baseline, "baseline")
    _check_common(alpha, power, ratio)
    if mde == 0:
        raise ValueError("mde must be non-zero")
    delta = baseline * mde if relative else mde
    p1 = baseline
    p2 = baseline + delta
    _check_rate(p2, "baseline + effect")
    za = _z_alpha(alpha, two_sided)
    zb = float(norm.ppf(power))
    pbar = (p1 + ratio * p2) / (1 + ratio)
    term0 = za * math.sqrt((1 + 1 / ratio) * pbar * (1 - pbar))
    term1 = zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
    n_c = math.ceil((term0 + term1) ** 2 / delta**2)
    n_t = math.ceil(ratio * n_c)
    return SampleSize(n_c, n_t, p1, p2, delta, delta / p1, alpha, power, ratio)


def power_proportions(
    baseline: float,
    effect: float,
    n_control: int,
    n_treatment: int | None = None,
    *,
    relative: bool = False,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Power of the two-proportion z-test at the given group sizes."""
    _check_rate(baseline, "baseline")
    if n_control <= 0:
        raise ValueError("n_control must be positive")
    n_t = n_control if n_treatment is None else n_treatment
    delta = baseline * effect if relative else effect
    p1, p2 = baseline, baseline + delta
    _check_rate(p2, "baseline + effect")
    pbar = (p1 * n_control + p2 * n_t) / (n_control + n_t)
    se0 = math.sqrt(pbar * (1 - pbar) * (1 / n_control + 1 / n_t))
    se1 = math.sqrt(p1 * (1 - p1) / n_control + p2 * (1 - p2) / n_t)
    za = _z_alpha(alpha, two_sided)
    upper = 1 - norm.cdf((za * se0 - abs(delta)) / se1)
    lower = norm.cdf((-za * se0 - abs(delta)) / se1) if two_sided else 0.0
    return float(upper + lower)


def mde_for_n(
    baseline: float,
    n_control: int,
    n_treatment: int | None = None,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """Smallest positive absolute lift detectable at the given sample sizes and power."""
    _check_rate(baseline, "baseline")
    _check_common(alpha, power, 1.0)

    def gap(effect: float) -> float:
        return (
            power_proportions(
                baseline, effect, n_control, n_treatment, alpha=alpha, two_sided=two_sided
            )
            - power
        )

    hi = (1 - baseline) - 1e-9
    if gap(hi) < 0:
        raise ValueError("no detectable effect below 100% at these sample sizes")
    return float(brentq(gap, 1e-9, hi, xtol=1e-12))


def sample_size_means(
    sd: float,
    mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> tuple[int, int]:
    """Sample sizes (control, treatment) for a difference in means with a common SD."""
    _check_common(alpha, power, ratio)
    if sd <= 0 or mde == 0:
        raise ValueError("sd must be positive and mde non-zero")
    za = _z_alpha(alpha, two_sided)
    zb = float(norm.ppf(power))
    n_c = math.ceil((za + zb) ** 2 * sd**2 * (1 + 1 / ratio) / mde**2)
    return n_c, math.ceil(ratio * n_c)


def power_curve(
    baseline: float, mde: float, n_values: list[int], *, relative: bool = False, alpha: float = 0.05
) -> list[float]:
    """Power at each per-arm sample size in `n_values` (equal arms)."""
    return [power_proportions(baseline, mde, n, relative=relative, alpha=alpha) for n in n_values]
