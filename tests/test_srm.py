import pytest

from expkit.srm import srm_test


def test_exact_split_passes():
    r = srm_test([5000, 5000], [1, 1])
    assert r.p_value == pytest.approx(1.0) and not r.flagged


def test_clear_mismatch_is_flagged():
    r = srm_test([5200, 4800], [1, 1])
    assert r.flagged and r.p_value < 1e-3


def test_uneven_design_ratio():
    r = srm_test([1500, 8500], [15, 85])
    assert not r.flagged
    assert r.observed_shares[1] == pytest.approx(0.85)


def test_three_arms():
    r = srm_test([3300, 3350, 3350], [1, 1, 1])
    assert r.dof == 2 and not r.flagged
    assert srm_test([3000, 3500, 3500], [1, 1, 1]).flagged


def test_false_alarm_rate_is_near_threshold(rng):
    reps, flagged = 4000, 0
    for _ in range(reps):
        x = rng.binomial(2000, 0.5)
        flagged += srm_test([x, 2000 - x], [1, 1], threshold=0.05).flagged
    assert abs(flagged / reps - 0.05) < 0.015


def test_detects_one_percent_loss_at_scale():
    n_t = round(0.85e6 * 0.99)
    assert srm_test([150_000, n_t], [15, 85]).flagged


@pytest.mark.parametrize(
    "obs,ratios",
    [([1, 2, 3], [1, 1]), ([5], [1]), ([5, 5], [1, 0]), ([-1, 5], [1, 1])],
)
def test_bad_input(obs, ratios):
    with pytest.raises(ValueError):
        srm_test(obs, ratios)
