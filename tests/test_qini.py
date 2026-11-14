import numpy as np
import pytest

from expkit.qini import percentile_ci, qini_bootstrap, qini_curve


def brute_force_curve(y, t, score, n_bins):
    """Slow reference implementation with explicit loops."""
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    n = len(y)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    xs, qs = [0.0], [0.0]
    for e in edges[1:]:
        yt, yc = y[:e][t[:e] == 1], y[:e][t[:e] == 0]
        lift = yt.mean() - yc.mean()
        xs.append(e / n)
        qs.append(e * lift / n)
    return np.array(xs), np.array(qs)


def make(rng, n=20_000, hetero=True):
    x = rng.normal(size=n)
    t = (rng.random(n) < 0.5).astype(int)
    tau = 0.15 * (x > 0) if hetero else np.full(n, 0.05)
    y = (rng.random(n) < 0.2 + tau * t).astype(int)
    return x, t, y, tau


def test_matches_brute_force(rng):
    x, t, y, _ = make(rng, n=5000)
    c = qini_curve(y, t, x, n_bins=20)
    xs, qs = brute_force_curve(y, t, x, 20)
    assert np.allclose(c.x, xs)
    assert np.allclose(c.q, qs, atol=1e-6)


def test_curve_ends_at_ate(rng):
    x, t, y, _ = make(rng)
    c = qini_curve(y, t, x)
    ate = y[t == 1].mean() - y[t == 0].mean()
    assert c.q[-1] == pytest.approx(ate, abs=1e-6)
    assert c.ate == pytest.approx(ate, abs=1e-6)


def test_oracle_beats_random_and_inverted(rng):
    x, t, y, tau = make(rng, n=60_000)
    oracle = qini_curve(y, t, tau + 1e-6 * rng.normal(size=len(y)))
    random = qini_curve(y, t, rng.normal(size=len(y)))
    inverted = qini_curve(y, t, -tau + 1e-6 * rng.normal(size=len(y)))
    assert oracle.coefficient > 0.005
    assert abs(random.coefficient) < 0.002
    assert inverted.coefficient < -0.005


def test_capture_of_oracle_exceeds_share_targeted(rng):
    x, t, y, tau = make(rng, n=60_000)
    c = qini_curve(y, t, tau + 1e-6 * rng.normal(size=len(y)))
    # half the users carry all of the effect, so the top 50% captures about all of it
    assert c.capture(0.5) == pytest.approx(1.0, abs=0.1)
    assert c.capture(0.1) == pytest.approx(0.2, abs=0.05)


def test_no_effect_gives_zero_coefficient_for_any_score(rng):
    n = 40_000
    x = rng.normal(size=n)
    t = (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < 0.2).astype(int)
    assert abs(qini_curve(y, t, x).coefficient) < 0.002


def test_shuffling_outcomes_destroys_signal(rng):
    x, t, y, tau = make(rng, n=40_000)
    real = qini_curve(y, t, tau + 1e-6 * rng.normal(size=len(y))).coefficient
    shuffled = qini_curve(rng.permutation(y), t, tau).coefficient
    assert real > 0.005 > abs(shuffled)


def test_bootstrap_shapes_and_coverage(rng):
    x, t, y, tau = make(rng, n=10_000)
    scores = {"oracle": tau + 1e-6 * rng.normal(size=len(y)), "random": rng.normal(size=len(y))}
    boot = qini_bootstrap(y, t, scores, n_boot=60, seed=1)
    assert boot["oracle"]["coefficient"].shape == (60,)
    lo, hi = percentile_ci(boot["oracle"]["coefficient"])
    point = qini_curve(y, t, scores["oracle"]).coefficient
    assert lo < point < hi
    # an informative score is clearly above a random one, and its interval excludes zero
    assert lo > 0.0
    assert point > qini_curve(y, t, scores["random"]).coefficient


def test_bootstrap_is_reproducible(rng):
    x, t, y, tau = make(rng, n=4000)
    a = qini_bootstrap(y, t, {"m": x}, n_boot=5, seed=3)["m"]["coefficient"]
    b = qini_bootstrap(y, t, {"m": x}, n_boot=5, seed=3)["m"]["coefficient"]
    assert np.array_equal(a, b)


def test_length_mismatch():
    with pytest.raises(ValueError):
        qini_curve(np.zeros(5), np.zeros(4), np.zeros(5))
