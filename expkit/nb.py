"""Helpers shared by the notebooks in `notebooks/`.

Two modes, chosen by the environment variable EXPKIT_NOTEBOOK_MODE:

- `real` (default): the Criteo and Hillstrom data and `docs/results/results.json` written by
  `expkit run-all`. Figures are saved to `docs/figures/`.
- `fixture`: a small synthetic dataset and a smoke run of the pipeline, cached in the system
  temp folder. This is what CI uses to prove the notebooks run end to end. Nothing it produces
  is a result, and figures go to the temp folder, not to the repository.

Numbers shown in the notebooks come from SQL aggregates, from code, or from results.json. They
are never typed into the text.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

from . import data
from .experiment import Settings, run_all
from .synth import write_fixture_files

MODE_ENV = "EXPKIT_NOTEBOOK_MODE"
FIXTURE_DIR = Path(tempfile.gettempdir()) / "expkit_notebook_fixture_v1"

# Okabe-Ito colour-blind-safe palette.
BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"
PINK = "#CC79A7"
YELLOW = "#F0E442"
BLACK = "#000000"
GREY = "#6B6B6B"
TREATED, CONTROL = BLUE, ORANGE
ACTION_COLORS = {"SHIP": GREEN, "RUN_LONGER": ORANGE, "DONT_SHIP": VERMILLION, "INVALID": GREY}
MODEL_COLORS = {
    "response_model": GREY,
    "s_learner": BLUE,
    "t_learner": VERMILLION,
    "causal_forest": GREEN,
}


def apply_style() -> None:
    try:  # inside a notebook kernel, draw figures inline even if an import chose Agg
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("matplotlib", "inline")
            if "inline" not in plt.get_backend():
                plt.switch_backend("module://matplotlib_inline.backend_inline")
    except Exception:  # pragma: no cover - plain Python session
        pass
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 130,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E3E3E3",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


@dataclass
class Context:
    mode: str
    results: dict
    con: duckdb.DuckDBPyConnection
    fig_dir: Path
    settings: Settings

    @property
    def is_fixture(self) -> bool:
        return self.mode == "fixture"

    def savefig(self, fig, name: str) -> None:
        """Save a figure as PNG (real mode: docs/figures, fixture mode: temp folder) and show it."""
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.fig_dir / f"{name}.png", bbox_inches="tight")
        plt.show()

    def banner(self) -> None:
        meta = self.results["meta"]
        if self.is_fixture:
            md(
                "> **Synthetic fixture run.** This notebook is running on generated data so that "
                "CI can check that it executes. Every number and every chart here is **not a "
                "result**. The committed copy of this notebook was run on the real data."
            )
        else:
            commit = (meta.get("git_commit") or "unknown")[:7]
            md(
                f"> Data: **{meta['dataset']}**. Pipeline numbers are read from "
                f"`docs/results/results.json` (written by `expkit run-all` at code commit "
                f"`{commit}`) and match `docs/RESULTS.md`. Seed {meta['seed']}."
            )

    def check(self, label: str, recomputed: float, reported: float, rtol: float = 1e-6) -> None:
        """Show a recomputed value next to the pipeline value; fail loudly if they disagree."""
        denom = max(abs(reported), 1e-12)
        rel = abs(recomputed - reported) / denom
        status = "matches" if rel <= rtol else "DIFFERS"
        md(f"- Check: {label}: recomputed `{recomputed:.6g}`, pipeline `{reported:.6g}` ({status})")
        if rel > rtol:
            raise AssertionError(f"{label}: {recomputed} vs {reported} (rel diff {rel:.2e})")


def md(text: str) -> None:
    from IPython.display import Markdown, display

    display(Markdown(text))


def notes(title: str, items: list[str]) -> None:
    md(f"### {title}\n\n" + "\n".join(f"- {x}" for x in items))


def mode_from_env() -> str:
    mode = os.environ.get(MODE_ENV, "real").strip().lower()
    if mode not in ("real", "fixture"):
        raise SystemExit(f"{MODE_ENV} must be 'real' or 'fixture', got {mode!r}")
    return mode


def load(mode: str | None = None) -> Context:
    apply_style()
    mode = mode or mode_from_env()
    if mode == "fixture":
        return _load_fixture()
    db = data.ROOT / "data" / "criteo.duckdb"
    res = data.ROOT / "docs" / "results" / "results.json"
    hill = data.ROOT / "data" / "raw" / "hillstrom.csv"
    for p in (db, res):
        if not p.exists():
            raise SystemExit(
                f"{p} not found. Run `expkit fetch-data` then `expkit run-all`, "
                f"or set {MODE_ENV}=fixture."
            )
    # Read-only, so several notebooks (kernels) can be open on the same file at once. Anything
    # missing from the file is created as a temporary object in this connection only.
    con = data.connect(db, read_only=True)
    if not data.has_table(con, "criteo_split"):
        split_sql = data.sql_text("02_split.sql").replace("OR REPLACE VIEW", "OR REPLACE TEMP VIEW")
        con.execute(split_sql)
    if not data.has_table(con, "hillstrom"):
        data.build_hillstrom(con, hill, temp=True)
    results = json.loads(res.read_text(encoding="utf-8"))
    return Context(
        "real",
        results,
        con,
        data.ROOT / "docs" / "figures",
        Settings(**results["meta"]["settings"]),
    )


def _load_fixture() -> Context:
    results_path = FIXTURE_DIR / "results.json"
    criteo, hill = FIXTURE_DIR / "criteo_fixture.csv", FIXTURE_DIR / "hillstrom_fixture.csv"
    if not (results_path.exists() and criteo.exists() and hill.exists()):
        write_fixture_files(FIXTURE_DIR, n=60_000)
        run_all(
            criteo, hill, None, results_path, Settings.smoke(),
            dataset_label="synthetic fixture (not a result)",
        )  # fmt: skip
    con = data.connect()
    data.build_criteo(con, criteo)
    data.build_hillstrom(con, hill)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return Context("fixture", results, con, FIXTURE_DIR / "figures", Settings.smoke())


# ---------------------------------------------------------------- small plotting helpers


def forest(
    ax, rows: list[dict], *, scale: float = 100.0, xlabel: str = "", ref: float | None = None
):
    """Horizontal confidence-interval plot. rows: label, est, lo, hi, optional color."""
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        color = r.get("color", BLUE)
        ax.plot(
            [r["lo"] * scale, r["hi"] * scale], [y, y], color=color, lw=2.5, solid_capstyle="butt"
        )
        ax.plot(r["est"] * scale, y, "o", color=color, ms=7, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows][::-1])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    if ref is not None:
        ax.axvline(ref * scale, color=BLACK, lw=1, ls="--")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def rate_ci(successes: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Normal-approximation 95% half-width and the rate, for plotting binned rates."""
    p = successes / np.maximum(n, 1)
    return p, z * np.sqrt(p * (1 - p) / np.maximum(n, 1))
