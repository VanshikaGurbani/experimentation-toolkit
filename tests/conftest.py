import numpy as np
import pytest

from expkit import data
from expkit.synth import make_criteo_like, make_hillstrom_like


@pytest.fixture(scope="session")
def criteo_df():
    return make_criteo_like(60_000, seed=1)


@pytest.fixture(scope="session")
def fixture_db(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx")
    csv = d / "criteo.csv"
    make_criteo_like(40_000, seed=2).to_csv(csv, index=False)
    con = data.connect()
    data.build_criteo(con, csv)
    yield con
    con.close()


@pytest.fixture()
def rng():
    return np.random.default_rng(123)


@pytest.fixture(scope="session")
def hillstrom_df():
    return make_hillstrom_like(4000, seed=3)
