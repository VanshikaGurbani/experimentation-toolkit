"""Execute the notebooks in `notebooks/` from the command line.

    expkit run-notebooks --fixture --out DIR   run on the synthetic fixture (what CI does); the
                                               executed copies go to DIR and the committed
                                               notebooks are left untouched
    expkit run-notebooks                       run on the real data and save the outputs in place

Needs the optional `notebooks` extra (nbformat, nbclient, ipykernel).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from . import data
from .nb import MODE_ENV

NOTEBOOK_DIR = data.ROOT / "notebooks"


def notebook_paths(directory: Path = NOTEBOOK_DIR) -> list[Path]:
    return sorted(directory.glob("[0-9][0-9]_*.ipynb"))


def run_notebooks(
    directory: Path = NOTEBOOK_DIR,
    *,
    fixture: bool = False,
    out_dir: Path | None = None,
    timeout: int = 3600,
) -> list[Path]:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise SystemExit(
            "Running notebooks needs the 'notebooks' extra: pip install -e '.[notebooks]'"
        ) from exc

    if fixture and out_dir is None:
        raise SystemExit("--fixture needs --out so the committed notebooks are not overwritten")
    paths = notebook_paths(directory)
    if not paths:
        raise SystemExit(f"no notebooks found in {directory}")
    os.environ[MODE_ENV] = "fixture" if fixture else "real"
    written = []
    for path in paths:
        started = time.time()
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            record_timing=False,
            resources={"metadata": {"path": str(directory)}},
        )
        client.execute()
        dest_dir = out_dir if out_dir is not None else directory
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / path.name
        nbformat.write(nb, dest)
        written.append(dest)
        print(f"{path.name}: ok in {time.time() - started:.0f}s -> {dest}", flush=True)
    return written
