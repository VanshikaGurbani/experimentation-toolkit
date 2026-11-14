"""Synthetic datasets with the same columns as the real ones, for tests and CI.

The Criteo-like generator has a known heterogeneous treatment effect (the lift grows with f2),
so uplift models can be checked against a ground truth. Nothing here is used for reported results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_criteo_like(
    n: int = 60_000, seed: int = 0, treat_share: float = 0.85, strong_effect: bool = True
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12))
    t = (rng.random(n) < treat_share).astype(int)
    base = -3.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1]
    lift = 0.15 + (0.8 * X[:, 2] if strong_effect else 0.0)
    visit = (rng.random(n) < _sigmoid(base + t * lift)).astype(int)
    conversion = visit * (rng.random(n) < 0.08).astype(int)
    exposure = t * (rng.random(n) < 0.03).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    df["treatment"], df["conversion"], df["visit"], df["exposure"] = t, conversion, visit, exposure
    return df


def make_hillstrom_like(n: int = 6_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    segment = rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], size=n)
    history = rng.gamma(2.0, 120.0, size=n).round(2)
    treated = segment != "No E-Mail"
    p_visit = _sigmoid(-2.2 + 0.004 * history + 0.5 * treated)
    visit = (rng.random(n) < p_visit).astype(int)
    conversion = visit * (rng.random(n) < 0.12).astype(int)
    spend = conversion * rng.gamma(2.0, 40.0, size=n).round(2)
    return pd.DataFrame(
        {
            "recency": rng.integers(1, 13, size=n),
            "history_segment": "1) $0 - $100",
            "history": history,
            "mens": rng.integers(0, 2, size=n),
            "womens": rng.integers(0, 2, size=n),
            "zip_code": "Urban",
            "newbie": rng.integers(0, 2, size=n),
            "channel": "Web",
            "segment": segment,
            "visit": visit,
            "conversion": conversion,
            "spend": spend,
        }
    )


def write_fixture_files(directory: str | Path, n: int = 60_000, seed: int = 0) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    criteo = directory / "criteo_fixture.csv"
    hill = directory / "hillstrom_fixture.csv"
    make_criteo_like(n, seed).to_csv(criteo, index=False)
    make_hillstrom_like(max(n // 10, 3000), seed).to_csv(hill, index=False)
    return criteo, hill
