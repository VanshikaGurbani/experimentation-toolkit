import numpy as np
import pytest
from scipy import stats

from expkit.inference import compare_means, compare_proportions, compare_samples


def test_proportion_interval_by_hand():
    e = compare_proportions(120, 1000, 100, 1000)
    se = np.sqrt(0.12 * 0.88 / 1000 + 0.10 * 0.90 / 1000)
    assert e.diff == pytest.approx(0.02)
    assert e.se == pytest.approx(se)
    assert e.ci_low == pytest.approx(0.02 - 1.959964 * se, abs=1e-6)
    assert e.rel_lift == pytest.approx(0.2)
    assert e.rel_ci_low < 0.2 < e.rel_ci_high


def test_matches_scipy_welch(rng):
    a = rng.normal(1.0, 2.0, 400)
    b = rng.normal(0.8, 3.0, 500)
    e = compare_samples(a, b)
    t = stats.ttest_ind(a, b, equal_var=False)
    assert e.p_value == pytest.approx(t.pvalue, rel=0.02)


def test_interval_coverage(rng):
    covered, reps = 0, 3000
    for _ in range(reps):
        xt, xc = rng.binomial(3000, 0.12), rng.binomial(3000, 0.10)
        e = compare_proportions(xt, 3000, xc, 3000)
        covered += e.ci_low <= 0.02 <= e.ci_high
    assert abs(covered / reps - 0.95) < 0.02


def test_relative_lift_coverage(rng):
    covered, reps = 0, 3000
    for _ in range(reps):
        xt, xc = rng.binomial(5000, 0.06), rng.binomial(5000, 0.05)
        e = compare_proportions(xt, 5000, xc, 5000)
        covered += e.rel_ci_low <= 0.2 <= e.rel_ci_high
    assert abs(covered / reps - 0.95) < 0.025


def test_no_relative_lift_for_nonpositive_mean():
    e = compare_means(-1.0, 1.0, 100, 2.0, 1.0, 100)
    assert e.rel_lift is None and e.rel_ci_low is None


def test_no_relative_lift_when_mean_is_noise_around_zero(rng):
    a = rng.normal(0.001, 1.0, 500)
    b = rng.normal(0.0, 1.0, 500)
    e = compare_samples(a, b)
    assert e.rel_lift is None and e.rel_ci_high is None


def test_alpha_changes_width():
    wide = compare_proportions(120, 1000, 100, 1000, alpha=0.01)
    narrow = compare_proportions(120, 1000, 100, 1000, alpha=0.20)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


def test_too_few_rows():
    with pytest.raises(ValueError):
        compare_means(0.1, 0.1, 1, 0.1, 0.1, 10)
