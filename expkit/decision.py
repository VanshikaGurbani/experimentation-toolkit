"""Ship / don't ship / run longer decision rule.

The rule works on the confidence interval, not on a p-value alone, and it compares the interval
to a practical threshold: the smallest lift that would justify shipping. That threshold is a
business input and must be set before looking at results.

- INVALID: a sample ratio mismatch was flagged, so the data cannot be trusted.
- DONT_SHIP: the whole interval is below the threshold, or a guardrail metric is confidently
  worse than its tolerance.
- SHIP: the whole interval is at or above the threshold and no guardrail is at risk.
- RUN_LONGER: the interval straddles the threshold, or a guardrail cannot be ruled out. When the
  interval straddles, `sample_scale` is the factor by which both arms must grow for the interval
  to clear the threshold if the current point estimate holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .inference import EffectEstimate
from .srm import SRMResult

SHIP = "SHIP"
DONT_SHIP = "DONT_SHIP"
RUN_LONGER = "RUN_LONGER"
INVALID = "INVALID"


@dataclass(frozen=True)
class Guardrail:
    """A metric where higher is better and a drop larger than `tolerance` is unacceptable."""

    name: str
    estimate: EffectEstimate
    tolerance: float


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    sample_scale: float | None = None
    notes: list[str] = field(default_factory=list)


def required_scale(est: EffectEstimate, threshold: float) -> float | None:
    """Factor by which both arms must grow for the interval to clear the threshold.

    Assumes the point estimate stays where it is. Returns None when the estimate sits exactly
    on the threshold (no amount of data resolves it).
    """
    gap = abs(est.diff - threshold)
    half_width = (est.ci_high - est.ci_low) / 2
    if gap <= 0:
        return None
    return (half_width / gap) ** 2


def decide(
    est: EffectEstimate,
    practical_threshold: float,
    *,
    srm: SRMResult | None = None,
    guardrails: list[Guardrail] | None = None,
) -> Decision:
    if srm is not None and srm.flagged:
        return Decision(
            INVALID,
            f"Sample ratio mismatch (p = {srm.p_value:.2e}); fix assignment or logging first.",
        )
    notes: list[str] = []
    for g in guardrails or []:
        if g.estimate.ci_high < -g.tolerance:
            return Decision(
                DONT_SHIP,
                f"Guardrail '{g.name}' is confidently worse than the allowed drop "
                f"of {g.tolerance}.",
            )
        if g.estimate.ci_low < -g.tolerance:
            notes.append(f"Guardrail '{g.name}' cannot rule out a drop larger than {g.tolerance}.")
    if est.ci_high < practical_threshold:
        return Decision(
            DONT_SHIP, "Interval is entirely below the practical threshold.", None, notes
        )
    if est.ci_low >= practical_threshold:
        if notes:
            return Decision(
                RUN_LONGER, "Primary metric clears the threshold but a guardrail is unresolved.",
                None, notes,
            )  # fmt: skip
        return Decision(SHIP, "Interval is entirely at or above the practical threshold.")
    return Decision(
        RUN_LONGER,
        "Interval straddles the practical threshold.",
        required_scale(est, practical_threshold),
        notes,
    )
