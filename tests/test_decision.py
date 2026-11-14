from expkit.decision import (
    DONT_SHIP,
    INVALID,
    RUN_LONGER,
    SHIP,
    Guardrail,
    decide,
    required_scale,
)
from expkit.inference import compare_proportions
from expkit.srm import srm_test


def est(xt, xc, n=100_000):
    return compare_proportions(xt, n, xc, n)


def test_ship_when_interval_above_threshold():
    assert decide(est(5500, 4000), 0.005).action == SHIP


def test_dont_ship_when_interval_below_threshold():
    assert decide(est(4050, 4000), 0.005).action == DONT_SHIP


def test_negative_effect_is_dont_ship():
    assert decide(est(3500, 4000), 0.0).action == DONT_SHIP


def test_run_longer_when_straddling_and_scale_resolves_it():
    e = est(4700, 4000, n=20_000)
    d = decide(e, 0.03)
    assert d.action == RUN_LONGER and d.sample_scale and d.sample_scale > 1
    # growing both arms by the scale (same rates) should make the interval clear the threshold
    n2 = round(20_000 * d.sample_scale * 1.05)
    again = compare_proportions(round(0.235 * n2), n2, round(0.2 * n2), n2)
    assert decide(again, 0.03).action == SHIP


def test_required_scale_formula():
    e = est(4700, 4000, n=20_000)
    half = (e.ci_high - e.ci_low) / 2
    assert required_scale(e, 0.0) == (half / abs(e.diff)) ** 2


def test_estimate_exactly_on_threshold_has_no_finite_scale():
    e = est(4500, 4000, n=20_000)
    assert required_scale(e, e.diff) is None


def test_srm_overrides_everything():
    bad = srm_test([5500, 4500], [1, 1])
    assert decide(est(6000, 4000), 0.0, srm=bad).action == INVALID
    good = srm_test([5000, 5000], [1, 1])
    assert decide(est(6000, 4000), 0.0, srm=good).action == SHIP


def test_guardrail_harm_blocks_shipping():
    g = Guardrail("conversion", est(3000, 4000), tolerance=0.001)
    assert decide(est(6000, 4000), 0.0, guardrails=[g]).action == DONT_SHIP


def test_guardrail_unresolved_means_run_longer():
    g = Guardrail("conversion", compare_proportions(41, 1000, 40, 1000), tolerance=0.001)
    d = decide(est(6000, 4000), 0.0, guardrails=[g])
    assert d.action == RUN_LONGER and d.notes
