import numpy as np
import pytest

from expkit.adjust import aipw_effect
from expkit.balance import assignment_auc, auc_null_se, fit_propensity


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def confounded(rng, n=40_000, effect=0.10):
    """Users with high x are both more likely to be treated and more likely to convert."""
    x = rng.normal(size=n)
    e = sigmoid(1.2 * x)
    t = (rng.random(n) < e).astype(int)
    mu0 = sigmoid(-1.0 + 0.9 * x)
    mu1 = np.clip(mu0 + effect, 0, 1)
    y = (rng.random(n) < np.where(t == 1, mu1, mu0)).astype(int)
    return x, e, t, y, mu0, mu1


def test_aipw_recovers_truth_where_raw_difference_is_biased(rng):
    x, e, t, y, mu0, mu1 = confounded(rng)
    raw = y[t == 1].mean() - y[t == 0].mean()
    est = aipw_effect(y, t, mu0, mu1, e)
    assert raw - 0.10 > 0.05
    assert est.diff == pytest.approx(0.10, abs=0.015)
    assert est.ci_low < 0.10 < est.ci_high


def test_aipw_matches_raw_difference_in_clean_rct(rng):
    n = 60_000
    t = (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < 0.10 + 0.03 * t).astype(int)
    est = aipw_effect(y, t, np.full(n, 0.1), np.full(n, 0.13), np.full(n, 0.5))
    raw = y[t == 1].mean() - y[t == 0].mean()
    assert est.diff == pytest.approx(raw, abs=0.01)


def test_aipw_interval_coverage(rng):
    covered, reps = 0, 300
    for _ in range(reps):
        x, e, t, y, mu0, mu1 = confounded(rng, n=4000, effect=0.1)
        est = aipw_effect(y, t, mu0, mu1, e)
        covered += est.ci_low <= 0.1 <= est.ci_high
    assert abs(covered / reps - 0.95) < 0.04


def test_aipw_survives_wrong_outcome_model_if_propensity_is_right(rng):
    x, e, t, y, mu0, mu1 = confounded(rng)
    wrong = np.full(len(y), y.mean())
    est = aipw_effect(y, t, wrong, wrong, e)
    assert est.diff == pytest.approx(0.10, abs=0.03)


def test_propensity_is_clipped(rng):
    n = 1000
    t = (rng.random(n) < 0.5).astype(int)
    y = rng.integers(0, 2, n)
    est = aipw_effect(y, t, np.full(n, 0.5), np.full(n, 0.5), np.zeros(n))
    assert np.isfinite(est.diff) and np.isfinite(est.se)


def test_balance_passes_when_assignment_is_random(rng):
    n = 30_000
    X = rng.normal(size=(n, 5)).astype(np.float32)
    t = (rng.random(n) < 0.85).astype(int)
    half = n // 2
    m = fit_propensity(X[:half], t[:half], seed=1)
    check = assignment_auc(t[half:], m.predict_proba(X[half:])[:, 1])
    assert not check.flagged and abs(check.auc - 0.5) < 0.02


def test_balance_flags_when_assignment_depends_on_features(rng):
    n = 30_000
    X = rng.normal(size=(n, 5)).astype(np.float32)
    t = (rng.random(n) < sigmoid(0.3 * X[:, 0] + 1.5)).astype(int)
    half = n // 2
    m = fit_propensity(X[:half], t[:half], seed=1)
    check = assignment_auc(t[half:], m.predict_proba(X[half:])[:, 1])
    assert check.flagged and check.auc > 0.53


def test_null_standard_error_matches_simulation(rng):
    from sklearn.metrics import roc_auc_score

    t = np.array([1] * 300 + [0] * 700)
    aucs = [roc_auc_score(t, rng.random(1000)) for _ in range(600)]
    assert np.std(aucs) == pytest.approx(auc_null_se(300, 700), rel=0.12)
