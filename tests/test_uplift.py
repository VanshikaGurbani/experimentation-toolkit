import numpy as np
import pytest

from expkit import data
from expkit.qini import qini_curve
from expkit.uplift import CausalForest, ResponseModel, SLearner, TLearner

FAST = {"n_estimators": 80, "min_child_samples": 50, "learning_rate": 0.1}


def split(df, frac=0.5):
    X = df[data.FEATURES].to_numpy(np.float32)
    t = df["treatment"].to_numpy()
    y = df["visit"].to_numpy()
    k = int(len(df) * frac)
    return (X[:k], t[:k], y[:k]), (X[k:], t[k:], y[k:])


def test_t_learner_finds_planted_heterogeneity(criteo_df):
    (X, t, y), (Xe, te, ye) = split(criteo_df)
    m = TLearner(FAST).fit(X, t, y)
    c = qini_curve(ye, te, m.score(Xe))
    assert c.coefficient > 0.002
    # the planted effect grows with f2, so the score should correlate with it
    assert np.corrcoef(m.score(Xe), Xe[:, 2])[0, 1] > 0.3


def test_uplift_models_beat_response_model(criteo_df):
    (X, t, y), (Xe, te, ye) = split(criteo_df)
    resp = qini_curve(ye, te, ResponseModel(FAST).fit(X, t, y).score(Xe)).coefficient
    tl = qini_curve(ye, te, TLearner(FAST).fit(X, t, y).score(Xe)).coefficient
    sl = qini_curve(ye, te, SLearner(FAST).fit(X, t, y).score(Xe)).coefficient
    assert tl > resp and sl > resp


def test_shuffled_treatment_labels_destroy_uplift_signal(criteo_df):
    (X, t, y), (Xe, te, ye) = split(criteo_df)
    real = qini_curve(ye, te, TLearner(FAST).fit(X, t, y).score(Xe)).coefficient
    rng = np.random.default_rng(0)
    shuffled = TLearner(FAST).fit(X, rng.permutation(t), y)
    assert abs(qini_curve(ye, te, shuffled.score(Xe)).coefficient) < 0.4 * real


def test_early_stopping_with_validation_set(criteo_df):
    (X, t, y), (Xe, te, ye) = split(criteo_df)
    m = TLearner({**FAST, "n_estimators": 400}).fit(X, t, y, val=(Xe, te, ye))
    assert m.models[1].best_iteration_ < 400
    assert m.score(Xe).shape == (len(Xe),)


def test_exposure_is_never_a_feature():
    assert "exposure" not in data.FEATURES
    assert all(f.startswith("f") for f in data.FEATURES) and len(data.FEATURES) == 12


def test_causal_forest_runs_on_small_data(criteo_df):
    pytest.importorskip("econml")
    (X, t, y), (Xe, te, ye) = split(criteo_df.iloc[:12_000])
    m = CausalForest(n_estimators=40, min_samples_leaf=50).fit(X, t, y)
    s = m.score(Xe)
    assert s.shape == (len(Xe),) and np.isfinite(s).all()
