"""Data access: DuckDB database built from the SQL files in the repo's `sql/` folder."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT / "sql"
FEATURES = [f"f{i}" for i in range(12)]


def sql_text(name: str, **params: str) -> str:
    """Read a SQL file and substitute {{KEY}} placeholders."""
    text = (SQL_DIR / name).read_text(encoding="utf-8")
    for key, value in params.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def connect(
    db_path: str | Path | None = None, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB file (or an in-memory database). Read-only connections can be shared by
    several processes at once, a read-write connection is exclusive."""
    con = duckdb.connect(str(db_path) if db_path else ":memory:", read_only=read_only)
    con.execute("SET preserve_insertion_order = true")
    return con


def run_sql(con: duckdb.DuckDBPyConnection, name: str, **params: str) -> pd.DataFrame:
    return con.execute(sql_text(name, **params)).df()


def build_criteo(con: duckdb.DuckDBPyConnection, csv_path: str | Path) -> None:
    """Create the `criteo` table and the `criteo_split` view from the raw file."""
    path = str(Path(csv_path).resolve()).replace("\\", "/").replace("'", "''")
    con.execute(sql_text("01_criteo_load.sql", CSV=path))
    con.execute(sql_text("02_split.sql"))


def build_hillstrom(
    con: duckdb.DuckDBPyConnection, csv_path: str | Path, temp: bool = False
) -> None:
    path = str(Path(csv_path).resolve()).replace("\\", "/").replace("'", "''")
    sql = sql_text("07_hillstrom_load.sql", CSV=path)
    if temp:  # for read-only connections: a temporary table that lives in this session only
        sql = sql.replace("CREATE OR REPLACE TABLE", "CREATE OR REPLACE TEMP TABLE")
    con.execute(sql)


def has_table(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(row and row[0] > 0)


def fetch_split(con: duckdb.DuckDBPyConnection, split: str | None = None) -> dict[str, np.ndarray]:
    """Features (float32 matrix `X`) and outcome vectors for one split, ordered by row id."""
    where = "" if split is None else f"WHERE split = '{split}'"
    cols = ", ".join(FEATURES + ["treatment", "visit", "conversion", "exposure", "row_id"])
    raw = con.execute(f"SELECT {cols} FROM criteo_split {where} ORDER BY row_id").fetchnumpy()
    X = np.column_stack([raw[f].astype(np.float32) for f in FEATURES])
    return {
        "X": X,
        "t": raw["treatment"].astype(np.int8),
        "visit": raw["visit"].astype(np.int8),
        "conversion": raw["conversion"].astype(np.int8),
        "exposure": raw["exposure"].astype(np.int8),
        "row_id": raw["row_id"],
    }
