import json
import re
from pathlib import Path

import pytest

from expkit import data
from expkit.experiment import Settings, run_all
from expkit.report import write_report
from expkit.synth import write_fixture_files

EM, EN = chr(0x2014), chr(0x2013)
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("smoke")
    criteo, hill = write_fixture_files(d / "raw", n=40_000, seed=7)
    st = Settings.smoke()
    st.run_causal_forest = False
    res = run_all(
        criteo, hill, None, d / "docs" / "results" / "results.json", st, dataset_label="fx"
    )
    readme = d / "README.md"
    readme.write_text("intro\n<!-- HEADLINE:START -->\nold\n<!-- HEADLINE:END -->\nrest\n")
    write_report(d / "docs" / "results" / "results.json", d / "docs", readme)
    return res, d, readme


def test_results_have_every_section(smoke_run):
    res, d, _ = smoke_run
    assert set(res) >= {"meta", "design", "effect", "power", "validation", "cuped", "uplift"}
    on_disk = json.loads((d / "docs" / "results" / "results.json").read_text())
    assert on_disk["effect"]["visit"]["diff"] == res["effect"]["visit"]["diff"]


def test_srm_and_balance_pass_on_randomized_fixture(smoke_run):
    res, _, _ = smoke_run
    assert not res["design"]["srm"]["flagged"]
    assert res["design"]["max_abs_smd"] < 0.1


def test_split_covers_all_rows(smoke_run):
    res, _, _ = smoke_run
    u = res["uplift"]
    total = res["design"]["arms"]["control"]["n"] + res["design"]["arms"]["treatment"]["n"]
    assert u["n_train"] + u["n_val"] + u["n_test"] == total


def test_uplift_models_have_signal_and_response_model_does_not(smoke_run):
    res, _, _ = smoke_run
    t = res["uplift"]["test"]
    assert t["t_learner"]["qini_coefficient"] > t["response_model"]["qini_coefficient"]
    assert res["uplift"]["selected_on_validation"] in ("s_learner", "t_learner")


def test_report_files_written_and_contain_json_numbers(smoke_run):
    res, d, readme = smoke_run
    results_md = (d / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    memo = (d / "docs" / "MEMO.md").read_text(encoding="utf-8")
    lift = res["effect"]["visit"]["diff"] * 100
    assert f"{lift:+.2f} pp" in memo
    assert f"{res['design']['srm']['chi2']:.3f}" in results_md
    assert "Recommendation:" in memo
    for fig in ("qini.png", "deciles.png", "power.png"):
        assert (d / "docs" / "results" / fig).stat().st_size > 1000
    head = readme.read_text(encoding="utf-8")
    assert "STALE_TEXT" not in head and head.startswith("intro") and head.rstrip().endswith("rest")
    assert "HEADLINE:START" in head


def test_generated_text_has_no_dashes_or_nan(smoke_run):
    _, d, readme = smoke_run
    for p in (d / "docs" / "RESULTS.md", d / "docs" / "MEMO.md", readme):
        text = p.read_text(encoding="utf-8")
        assert EM not in text and EN not in text
        assert not re.search(r"\bnan\b", text, re.I)


def test_hash_split_matches_documented_sizes():
    sql = (ROOT / "sql" / "02_split.sql").read_text()
    assert "< 40" in sql and "< 50" in sql


def test_rebuild_is_idempotent(tmp_path):
    criteo, _ = write_fixture_files(tmp_path, n=5000)
    con = data.connect()
    data.build_criteo(con, criteo)
    data.build_criteo(con, criteo)
    assert con.execute("SELECT count(*) FROM criteo").fetchone()[0] == 5000
