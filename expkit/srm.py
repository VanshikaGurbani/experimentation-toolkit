"""Sample ratio mismatch (SRM) check.

If the observed split between arms differs from the designed split by more than chance allows,
the assignment or logging is broken and the experiment result cannot be trusted. The usual
alarm threshold is p < 0.001 because the check runs on every experiment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import chisquare

DEFAULT_THRESHOLD = 0.001


@dataclass(frozen=True)
class SRMResult:
    observed: tuple[int, ...]
    expected: tuple[float, ...]
    chi2: float
    dof: int
    p_value: float
    threshold: float
    flagged: bool

    @property
    def observed_shares(self) -> tuple[float, ...]:
        total = sum(self.observed)
        return tuple(o / total for o in self.observed)


def srm_test(
    observed: Sequence[int],
    expected_ratios: Sequence[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> SRMResult:
    """Chi-square goodness-of-fit of arm counts against the designed allocation.

    `expected_ratios` need not sum to one (for example [85, 15] or [1, 1, 1]).
    """
    obs = np.asarray(observed, dtype=float)
    ratios = np.asarray(expected_ratios, dtype=float)
    if obs.ndim != 1 or obs.shape != ratios.shape or len(obs) < 2:
        raise ValueError("observed and expected_ratios must be 1-D with matching length >= 2")
    if (ratios <= 0).any() or (obs < 0).any():
        raise ValueError("ratios must be positive and counts non-negative")
    expected = ratios / ratios.sum() * obs.sum()
    chi2, p = chisquare(obs, expected)
    return SRMResult(
        observed=tuple(int(o) for o in obs),
        expected=tuple(float(e) for e in expected),
        chi2=float(chi2),
        dof=len(obs) - 1,
        p_value=float(p),
        threshold=threshold,
        flagged=bool(p < threshold),
    )
