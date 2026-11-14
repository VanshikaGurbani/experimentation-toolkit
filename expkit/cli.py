"""Command line: fetch data, run the analysis, write the report.

expkit fetch-data          download the two public datasets into data/raw
expkit run-all             run every stage on the real data, then write the report
expkit run-all --fixture   same pipeline on a small synthetic dataset (CI smoke test)
expkit report              rebuild docs/RESULTS.md and docs/MEMO.md from results.json
expkit run-notebooks       execute the notebooks in notebooks/ (add --fixture --out DIR for CI)
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

from . import data
from .experiment import Settings, run_all
from .report import write_report
from .synth import write_fixture_files

CRITEO_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
RAW = data.ROOT / "data" / "raw"


def fetch_data() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for url, name in (
        (HILLSTROM_URL, "hillstrom.csv"),
        (CRITEO_URL, "criteo-research-uplift-v2.1.csv.gz"),
    ):
        dest = RAW / name
        if dest.exists():
            print(f"{dest} already present")
            continue
        print(f"downloading {url} (Criteo is about 311 MB)")
        urllib.request.urlretrieve(url, dest)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="expkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-data")
    run = sub.add_parser("run-all")
    run.add_argument("--fixture", action="store_true", help="use a small synthetic dataset")
    run.add_argument("--rebuild", action="store_true", help="rebuild the DuckDB tables")
    run.add_argument("--no-causal-forest", action="store_true")
    run.add_argument("--out", type=Path, default=data.ROOT / "docs")
    sub.add_parser("report").add_argument("--out", type=Path, default=data.ROOT / "docs")
    nbp = sub.add_parser("run-notebooks", help="execute the notebooks in notebooks/")
    nbp.add_argument("--fixture", action="store_true", help="synthetic data, for CI")
    nbp.add_argument("--out", type=Path, help="where to write the executed copies")
    args = ap.parse_args(argv)

    if args.cmd == "fetch-data":
        fetch_data()
        return
    if args.cmd == "run-notebooks":
        from .nb_run import run_notebooks

        run_notebooks(fixture=args.fixture, out_dir=args.out)
        return
    if args.cmd == "report":
        write_report(args.out / "results" / "results.json", args.out, data.ROOT / "README.md")
        return

    out: Path = args.out
    if args.fixture:
        tmp = Path(tempfile.mkdtemp(prefix="expkit_fixture_"))
        criteo, hill = write_fixture_files(tmp, n=60_000)
        st, db, label = Settings.smoke(), None, "synthetic fixture (not a result)"
        readme = None
    else:
        criteo = RAW / "criteo-research-uplift-v2.1.csv.gz"
        hill = RAW / "hillstrom.csv"
        for p in (criteo, hill):
            if not p.exists():
                raise SystemExit(f"{p} not found; run `expkit fetch-data` first")
        st, db, label = Settings(), data.ROOT / "data" / "criteo.duckdb", "Criteo Uplift v2.1"
        readme = data.ROOT / "README.md"
    if args.no_causal_forest:
        st.run_causal_forest = False
    res_json = out / "results" / "results.json"
    run_all(criteo, hill, db, res_json, st, rebuild=args.rebuild, dataset_label=label)
    write_report(res_json, out, readme)


if __name__ == "__main__":
    main()
