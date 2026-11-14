"""Uplift (heterogeneous treatment effect) models for randomized experiments.

Every model exposes fit(X, t, y, val=None) and score(X). Higher score means a larger predicted
gain from treatment. Treatment must have been randomly assigned (possibly with unequal
probabilities), which is what makes these estimates causal rather than predictive.

- TLearner: one outcome model per arm; uplift is the difference of predicted probabilities.
- SLearner: one outcome model with the treatment as a feature; uplift is the difference
  between predictions at t=1 and t=0.
- ResponseModel: predicts the outcome and ignores treatment. This is the common mistake
  ("target the people most likely to convert") and serves as a baseline, not an uplift model.
- CausalForest: EconML's CausalForestDML (optional dependency), the same family of method used
  in the growth-mindset-intervention project.
"""

from __future__ import annotations

import os
from typing import Any

import lightgbm as lgb
import numpy as np

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "verbose": -1,
}


def _classifier(params: dict[str, Any] | None, seed: int) -> lgb.LGBMClassifier:
    p = {**DEFAULT_PARAMS, **(params or {})}
    return lgb.LGBMClassifier(random_state=seed, n_jobs=os.cpu_count() or 1, **p)


def _fit_binary(model, X, y, X_val=None, y_val=None, patience: int = 30):
    if X_val is not None and len(X_val) > 0 and len(np.unique(y_val)) > 1:
        model.fit(
            X,
            y,
            eval_X=X_val,
            eval_y=y_val,
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(patience, verbose=False)],
        )
    else:
        model.fit(X, y)
    return model


class TLearner:
    name = "t_learner"

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0):
        self.params, self.seed = params, seed
        self.models: dict[int, lgb.LGBMClassifier] = {}

    def fit(self, X, t, y, val=None):
        t = np.asarray(t).astype(int)
        for arm in (0, 1):
            m = _classifier(self.params, self.seed + arm)
            Xv = yv = None
            if val is not None:
                Xv_all, tv, yv_all = val
                keep = np.asarray(tv).astype(int) == arm
                Xv, yv = Xv_all[keep], yv_all[keep]
            self.models[arm] = _fit_binary(m, X[t == arm], y[t == arm], Xv, yv)
        return self

    def score(self, X) -> np.ndarray:
        return self.models[1].predict_proba(X)[:, 1] - self.models[0].predict_proba(X)[:, 1]


class SLearner:
    name = "s_learner"

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0):
        self.params, self.seed = params, seed
        self.model: lgb.LGBMClassifier | None = None

    @staticmethod
    def _stack(X, t) -> np.ndarray:
        return np.column_stack([X, np.asarray(t, dtype=np.float32)])

    def fit(self, X, t, y, val=None):
        m = _classifier(self.params, self.seed)
        Xv = yv = None
        if val is not None:
            Xv = self._stack(val[0], val[1])
            yv = val[2]
        self.model = _fit_binary(m, self._stack(X, t), y, Xv, yv)
        return self

    def score(self, X) -> np.ndarray:
        assert self.model is not None
        ones = np.ones(len(X), dtype=np.float32)
        return (
            self.model.predict_proba(self._stack(X, ones))[:, 1]
            - self.model.predict_proba(self._stack(X, 0 * ones))[:, 1]
        )


class ResponseModel:
    """Predicts P(outcome) ignoring treatment. Not an uplift model; a baseline."""

    name = "response_model"

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0):
        self.params, self.seed = params, seed
        self.model: lgb.LGBMClassifier | None = None

    def fit(self, X, t, y, val=None):
        Xv = yv = None
        if val is not None:
            Xv, yv = val[0], val[2]
        self.model = _fit_binary(_classifier(self.params, self.seed), X, y, Xv, yv)
        return self

    def score(self, X) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(X)[:, 1]


class CausalForest:
    """EconML CausalForestDML with the known (constant) assignment probability.

    Needs the optional `causal` extra. Forest fitting and prediction are slow, so callers
    normally fit on a subsample.
    """

    name = "causal_forest"

    def __init__(self, n_estimators: int = 200, min_samples_leaf: int = 100, seed: int = 0):
        self.n_estimators, self.min_samples_leaf, self.seed = n_estimators, min_samples_leaf, seed
        self.model = None

    def fit(self, X, t, y, val=None):
        from econml.dml import CausalForestDML
        from sklearn.dummy import DummyClassifier

        self.model = CausalForestDML(
            model_y=lgb.LGBMRegressor(
                n_estimators=150, learning_rate=0.05, num_leaves=15, verbose=-1,
                random_state=self.seed, n_jobs=os.cpu_count() or 1,
            ),
            model_t=DummyClassifier(strategy="prior"),
            discrete_treatment=True,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            cv=2,
            random_state=self.seed,
            n_jobs=os.cpu_count() or 1,
        )  # fmt: skip
        self.model.fit(np.asarray(y, dtype=float), np.asarray(t).astype(int), X=X)
        return self

    def score(self, X) -> np.ndarray:
        assert self.model is not None
        return np.asarray(self.model.effect(X)).ravel()
