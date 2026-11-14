"""Static checks on the committed notebooks.

Running them end to end is a separate CI step (`expkit run-notebooks --fixture`), because it
needs a Jupyter kernel and takes a minute or two. These tests check that what is committed is a
finished, honest run: executed on the real data, no errors, every recomputed number matching the
pipeline, the required sections present, and the files a reasonable size.
"""

import re
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
EXPECTED = [
    "01_eda_criteo.ipynb",
    "02_experiment_validity_and_effect.ipynb",
    "03_cuped_power_and_aa.ipynb",
    "04_uplift_and_targeting.ipynb",
    "05_hillstrom_sanity.ipynb",
]
MAX_NOTEBOOK_BYTES = 1_500_000
MAX_TOTAL_BYTES = 5_000_000


def load(name: str):
    return nbformat.read(NB_DIR / name, as_version=4)


def output_text(nb) -> str:
    """All text and markdown outputs of the code cells, joined."""
    parts = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.output_type == "stream":
                parts.append(out.text)
            elif out.output_type in ("display_data", "execute_result"):
                data = out.get("data", {})
                parts.append(data.get("text/markdown") or data.get("text/plain", ""))
    return "\n".join(parts)


def test_all_five_notebooks_exist_in_order():
    found = sorted(p.name for p in NB_DIR.glob("*.ipynb"))
    assert found == EXPECTED


@pytest.mark.parametrize("name", EXPECTED)
def test_notebook_is_valid_and_fully_executed(name):
    nb = load(name)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code, "no code cells"
    assert all(c.execution_count is not None for c in code), "a code cell was not executed"
    errors = [o for c in code for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, f"error output saved in {name}"


@pytest.mark.parametrize("name", EXPECTED)
def test_notebook_has_the_required_structure(name):
    nb = load(name)
    first_md = next(c for c in nb.cells if c.cell_type == "markdown").source
    assert "The question this notebook answers" in first_md
    text = output_text(nb)
    assert "What this means" in text
    assert "What is weak" in text


@pytest.mark.parametrize("name", EXPECTED)
def test_committed_notebooks_were_run_on_the_real_data(name):
    text = output_text(load(name))
    assert "Synthetic fixture run" not in text, "committed notebook holds fixture output"
    assert "Criteo Uplift v2.1" in text


@pytest.mark.parametrize("name", EXPECTED)
def test_recomputed_numbers_match_the_pipeline(name):
    text = output_text(load(name))
    assert "DIFFERS" not in text
    for line in re.findall(r"- Check: .*", text):
        assert "(matches)" in line, line


@pytest.mark.parametrize("name", EXPECTED)
def test_figures_are_embedded(name):
    nb = load(name)
    images = [
        o
        for c in nb.cells
        if c.cell_type == "code"
        for o in c.get("outputs", [])
        if o.output_type == "display_data" and "image/png" in o.get("data", {})
    ]
    assert images, f"{name} has no saved charts"


def test_notebook_sizes_are_reasonable():
    sizes = {n: (NB_DIR / n).stat().st_size for n in EXPECTED}
    assert all(s < MAX_NOTEBOOK_BYTES for s in sizes.values()), sizes
    assert sum(sizes.values()) < MAX_TOTAL_BYTES, sizes


def test_saved_figures_exist_and_are_linked_from_the_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    linked = re.findall(r"docs/figures/[\w\-]+\.png", readme)
    assert 3 <= len(set(linked)) <= 4
    for rel in set(linked):
        assert (ROOT / rel).exists(), rel


def test_notebook_dependencies_are_declared():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for pkg in ("nbformat", "nbclient", "ipykernel"):
        assert pkg in pyproject


def test_mode_switch_rejects_unknown_values(monkeypatch):
    from expkit import nb as nbmod

    monkeypatch.setenv(nbmod.MODE_ENV, "bogus")
    with pytest.raises(SystemExit):
        nbmod.mode_from_env()
    monkeypatch.setenv(nbmod.MODE_ENV, "fixture")
    assert nbmod.mode_from_env() == "fixture"


def test_rate_ci_matches_the_normal_formula():
    import numpy as np

    from expkit.nb import rate_ci

    p, hw = rate_ci(np.array([50.0]), np.array([1000.0]))
    assert p[0] == pytest.approx(0.05)
    assert hw[0] == pytest.approx(1.96 * np.sqrt(0.05 * 0.95 / 1000))
