import numpy as np
import pandas as pd

from expkit import data
from expkit.synth import make_criteo_like, write_fixture_files


def test_arm_summary_matches_pandas(fixture_db):
    df = make_criteo_like(40_000, seed=2)
    sql = data.run_sql(fixture_db, "03_arm_summary.sql").set_index("treatment")
    for arm in (0, 1):
        sub = df[df["treatment"] == arm]
        assert sql.loc[arm, "n"] == len(sub)
        assert sql.loc[arm, "visits"] == sub["visit"].sum()
        assert sql.loc[arm, "conversions"] == sub["conversion"].sum()
        assert sql.loc[arm, "exposed"] == sub["exposure"].sum()


def test_split_is_exclusive_complete_and_sized(fixture_db):
    counts = (
        fixture_db.execute("SELECT split, count(*) n FROM criteo_split GROUP BY split")
        .df()
        .set_index("split")["n"]
    )
    assert counts.sum() == 40_000
    assert abs(counts["train"] / 40_000 - 0.40) < 0.02
    assert abs(counts["val"] / 40_000 - 0.10) < 0.02
    assert abs(counts["test"] / 40_000 - 0.50) < 0.02
    dup = fixture_db.execute(
        "SELECT count(*) FROM (SELECT row_id FROM criteo_split GROUP BY row_id HAVING count(*) > 1)"
    ).fetchone()[0]
    assert dup == 0


def test_split_is_deterministic_and_balanced_by_arm(fixture_db):
    a = data.fetch_split(fixture_db, "test")["row_id"]
    b = data.fetch_split(fixture_db, "test")["row_id"]
    assert np.array_equal(a, b)
    s = data.run_sql(fixture_db, "05_split_summary.sql")
    shares = s.pivot(index="split", columns="treatment", values="n")
    ratio = shares[1] / (shares[0] + shares[1])
    assert (abs(ratio - 0.85) < 0.02).all()


def test_fetch_split_shapes_and_dtypes(fixture_db):
    d = data.fetch_split(fixture_db, "train")
    n = len(d["t"])
    assert d["X"].shape == (n, 12) and d["X"].dtype == np.float32
    assert set(np.unique(d["t"])) <= {0, 1}
    assert set(np.unique(d["visit"])) <= {0, 1}


def test_splits_do_not_share_rows(fixture_db):
    ids = [set(data.fetch_split(fixture_db, s)["row_id"]) for s in ("train", "val", "test")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_feature_balance_query(fixture_db):
    bal = data.run_sql(fixture_db, "04_feature_balance.sql").set_index("treatment")
    smd = [
        (bal.loc[1, f"m{i}"] - bal.loc[0, f"m{i}"])
        / np.sqrt((bal.loc[0, f"s{i}"] ** 2 + bal.loc[1, f"s{i}"] ** 2) / 2)
        for i in range(12)
    ]
    assert max(abs(x) for x in smd) < 0.05


def test_decile_query_orders_by_score(fixture_db):
    rng = np.random.default_rng(0)
    df = make_criteo_like(30_000, seed=5)
    score = df["f2"].to_numpy() + rng.normal(0, 0.01, len(df))
    scores = pd.DataFrame({"score": score, "treatment": df["treatment"], "visit": df["visit"]})
    fixture_db.register("scores", scores)
    out = data.run_sql(fixture_db, "06_uplift_deciles.sql")
    fixture_db.unregister("scores")
    assert list(out["decile"]) == list(range(1, 11))
    assert out["n"].sum() == 30_000
    assert out["uplift"].iloc[0] > out["uplift"].iloc[-1]


def test_hillstrom_sql_and_arms(tmp_path, hillstrom_df):
    csv = tmp_path / "h.csv"
    hillstrom_df.to_csv(csv, index=False)
    con = data.connect()
    data.build_hillstrom(con, csv)
    summ = data.run_sql(con, "08_hillstrom_summary.sql")
    assert summ["n"].sum() == len(hillstrom_df)
    assert set(summ["segment"]) == {"Mens E-Mail", "Womens E-Mail", "No E-Mail"}
    con.close()


def test_read_only_connections_can_share_a_file_and_create_temp_objects(tmp_path):
    criteo, hill = write_fixture_files(tmp_path, n=2000)
    db = tmp_path / "shared.duckdb"
    con = data.connect(db)
    data.build_criteo(con, criteo)
    con.close()
    first, second = data.connect(db, read_only=True), data.connect(db, read_only=True)
    data.build_hillstrom(first, hill, temp=True)
    assert data.has_table(first, "hillstrom") and not data.has_table(second, "hillstrom")
    n = "SELECT count(*) FROM criteo_split"
    assert first.execute(n).fetchone() == second.execute(n).fetchone()
