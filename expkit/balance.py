"""Randomization checks beyond the sample ratio: can the features predict who got treated?

In a clean randomized experiment, treatment assignment is independent of pre-treatment features,
so no model can predict it better than chance (AUC 50%). A model that does better means the arms
differ in who they contain, and a plain difference in means mixes the treatment effect with that
difference. With millions of users even a tiny imbalance is statistically unmistakable, so the
size of the AUC matters as much as its p-value.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

PROPENSITY_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 2000,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
    "verbose": -1,
}


@dataclass(frozen=True)
class BalanceCheck:
    auc: float
    null_se: float
    z: float
    p_value: float
    threshold: float
    flagged: bool
    n_test: int


def fit_propensity(X, t, val=None, seed: int = 0) -> lgb.LGBMClassifier:
    """Model of P(treated | features), with early stopping on a validation split if given."""
    m = lgb.LGBMClassifier(random_state=seed, n_jobs=os.cpu_count() or 1, **PROPENSITY_PARAMS)
    if val is not None:
        m.fit(
            X,
            t,
            eval_X=val[0],
            eval_y=val[1],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
    else:
        m.fit(X, t)
    return m


def auc_null_se(n_pos: int, n_neg: int) -> float:
    """Standard error of the AUC when the scores carry no information (ties aside)."""
    return math.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg))


def assignment_auc(t: np.ndarray, propensity: np.ndarray, threshold: float = 0.001) -> BalanceCheck:
    """AUC of held-out predicted propensity for the true assignment, tested against 50%."""
    t = np.asarray(t).astype(int)
    n_pos, n_neg = int(t.sum()), int((1 - t).sum())
    auc = float(roc_auc_score(t, propensity))
    se = auc_null_se(n_pos, n_neg)
    z = (auc - 0.5) / se
    p = float(2 * (1 - norm.cdf(abs(z))))
    return BalanceCheck(auc, se, float(z), p, threshold, bool(p < threshold), len(t))
