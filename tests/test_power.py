import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from expkit.power import (
    mde_for_n,
    power_curve,
    power_proportions,
    sample_size_means,
    sample_size_proportions,
)


def test_textbook_value():
    # 10% to 15%, alpha 5% two-sided, power 80%: about 686 per arm without continuity correction.
    s = sample_size_proportions(0.10, 0.05)
    assert s.n_control == s.n_treatment
    assert 680 <= s.n_control <= 692


def test_close_to_statsmodels_cohens_h():
    for p1, delta in [(0.05, 0.01), (0.2, 0.03), (0.5, 0.05)]:
        ours = sample_size_proportions(p1, delta).n_control
        h = proportion_effectsize(p1 + delta, p1)
        ref = NormalIndPower().solve_power(h, alpha=0.05, power=0.8, ratio=1.0)
        assert abs(ours - ref) / ref < 0.03


def test_power_round_trip():
    for baseline, mde, ratio in [(0.04, 0.01, 1), (0.04, 0.005, 85 / 15), (0.3, 0.02, 0.5)]:
        s = sample_size_proportions(baseline, mde, ratio=ratio)
        got = power_proportions(baseline, mde, s.n_control, s.n_treatment)
        assert 0.80 <= got < 0.805


def test_relative_equals_absolute():
    a = sample_size_proportions(0.04, 0.10, relative=True)
    b = sample_size_proportions(0.04, 0.004)
    assert a.n_control == b.n_control


def test_unequal_split_needs_more_total_users():
    equal = sample_size_proportions(0.04, 0.01)
    skew = sample_size_proportions(0.04, 0.01, ratio=85 / 15)
    assert skew.total > equal.total


def test_one_sided_needs_fewer():
    two = sample_size_proportions(0.1, 0.02).n_control
    one = sample_size_proportions(0.1, 0.02, two_sided=False).n_control
    assert one < two


def test_negative_effect_has_comparable_power():
    up = power_proportions(0.1, 0.02, 2000)
    down = power_proportions(0.1, -0.02, 2000)
    assert abs(up - down) < 0.08 and down > 0.5


def test_mde_is_inverse_of_power():
    n = 5000
    mde = mde_for_n(0.1, n)
    assert power_proportions(0.1, mde, n) == pytest.approx(0.80, abs=1e-4)


def test_power_increases_with_n():
    curve = power_curve(0.04, 0.01, [500, 2000, 8000, 32000])
    assert curve == sorted(curve) and curve[-1] > 0.99


def test_monte_carlo_matches_formula(rng):
    s = sample_size_proportions(0.08, 0.02)
    reps, hits = 3000, 0
    for _ in range(reps):
        xc = rng.binomial(s.n_control, 0.08)
        xt = rng.binomial(s.n_treatment, 0.10)
        pc, pt = xc / s.n_control, xt / s.n_treatment
        pbar = (xc + xt) / (s.n_control + s.n_treatment)
        se = np.sqrt(pbar * (1 - pbar) * (2 / s.n_control))
        hits += (pt - pc) / se > 1.96
    assert abs(hits / reps - 0.80) < 0.03


def test_means_formula():
    n_c, n_t = sample_size_means(sd=10, mde=2)
    # 2 * (1.96 + 0.8416)^2 * 100 / 4 = 392.4
    assert n_c == n_t and 391 <= n_c <= 394


@pytest.mark.parametrize(
    "call",
    [
        lambda: sample_size_proportions(0, 0.01),
        lambda: sample_size_proportions(0.5, 0.0),
        lambda: sample_size_proportions(0.99, 0.05),
        lambda: sample_size_proportions(0.1, 0.01, alpha=1.5),
        lambda: sample_size_proportions(0.1, 0.01, ratio=0),
        lambda: power_proportions(0.1, 0.01, 0),
        lambda: sample_size_means(0, 1),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
