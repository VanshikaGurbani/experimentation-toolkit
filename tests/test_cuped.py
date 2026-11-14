import numpy as np
import pytest

from expkit.cuped import cuped_adjust, cuped_compare, fit_theta


def simulate(rng, n=40_000, rho=0.7, effect=0.5):
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    y = rho * x + np.sqrt(1 - rho**2) * noise + effect * t
    return y, x, t


def test_variance_reduction_matches_rho_squared(rng):
    y, x, t = simulate(rng, rho=0.7)
    comp = cuped_compare(y, x, t)
    assert comp.variance_reduction == pytest.approx(0.49, abs=0.04)
    assert comp.sample_size_ratio == pytest.approx(0.51, abs=0.04)


def test_theta_is_covariance_ratio(rng):
    y, x, _ = simulate(rng, rho=0.6)
    theta = fit_theta(y, x)
    assert theta[0] == pytest.approx(np.cov(x, y)[0, 1] / x.var(ddof=1), rel=1e-6)


def test_no_relationship_means_no_gain(rng):
    y, x, t = simulate(rng, rho=0.0)
    comp = cuped_compare(y, x, t)
    assert abs(comp.variance_reduction) < 0.01


def test_estimate_stays_unbiased(rng):
    diffs_raw, diffs_adj = [], []
    for _ in range(300):
        y, x, t = simulate(rng, n=2000, rho=0.8, effect=0.3)
        c = cuped_compare(y, x, t)
        diffs_raw.append(c.raw.diff)
        diffs_adj.append(c.adjusted.diff)
    assert np.mean(diffs_adj) == pytest.approx(0.3, abs=0.01)
    assert np.var(diffs_adj) < 0.6 * np.var(diffs_raw)


def test_interval_coverage_after_adjustment(rng):
    covered = 0
    reps = 600
    for _ in range(reps):
        y, x, t = simulate(rng, n=3000, rho=0.6, effect=0.25)
        c = cuped_compare(y, x, t)
        covered += c.adjusted.ci_low <= 0.25 <= c.adjusted.ci_high
    assert abs(covered / reps - 0.95) < 0.03


def test_multiple_covariates_and_external_theta(rng):
    n = 30_000
    X = rng.normal(size=(n, 3))
    t = rng.integers(0, 2, n)
    y = X @ np.array([0.5, -0.3, 0.2]) + rng.normal(size=n) * 0.5 + 0.2 * t
    train = slice(0, 10_000)
    theta = fit_theta(y[train], X[train])
    comp = cuped_compare(y[10_000:], X[10_000:], t[10_000:], theta=theta)
    assert comp.variance_reduction > 0.4
    assert comp.adjusted.diff == pytest.approx(0.2, abs=0.03)


def test_adjusted_outcome_has_same_mean_as_raw(rng):
    y, x, _ = simulate(rng)
    res = cuped_adjust(y, x)
    assert res.y_adjusted.mean() == pytest.approx(y.mean(), abs=1e-9)


def test_treatment_must_not_be_leaked_into_covariates(rng):
    # If the covariate IS the treatment, the adjustment would erase the effect. The function
    # cannot know, which is why the docstring forbids it; this test documents the failure.
    y, _, t = simulate(rng, effect=0.5)
    comp = cuped_compare(y, t.astype(float), t)
    assert abs(comp.adjusted.diff) < 0.05 < comp.raw.diff


def test_shape_mismatch_raises(rng):
    with pytest.raises(ValueError):
        cuped_adjust(np.zeros(10), np.zeros(9))
