"""Copy one BIRD SQLite database into its own PostgreSQL database, faithfully and fast.

Why not trust the declared column types: BIRD's SQLite files are real-world data, so a
column declared INTEGER sometimes holds '' or text, and dates are stored as text. Each
column's PostgreSQL type is therefore chosen from what the column *actually holds*
(SQLite's `typeof()`), so every value survives the copy unchanged and the gold answers,
computed on the original SQLite file, stay comparable with answers computed here.

* integer-only columns → bigint (numeric if a value exceeds bigint)
* integer/real columns → double precision
* anything holding text → text (dates included, exactly as the SQLite file stores them)
* blobs → bytea
* names are kept exactly (`"CDSCode"`, `"Free Meal Count (K-12)"`)
* data goes in with COPY, so a million-row table takes seconds, not minutes
* primary and foreign keys are added afterwards, and skipped where BIRD's own data
  violates them
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import psycopg


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _column_type(src: sqlite3.Connection, table: str, column: str) -> str:
    kinds = {k for (k,) in src.execute(
        f"SELECT DISTINCT typeof({quote(column)}) FROM {quote(table)}")} - {"null"}
    if not kinds:
        return "text"
    if kinds == {"integer"}:
        lo, hi = src.execute(f"SELECT min({quote(column)}), max({quote(column)}) "
                             f"FROM {quote(table)}").fetchone()
        return "bigint" if -2**63 < (lo or 0) and (hi or 0) < 2**63 - 1 else "numeric"
    if kinds <= {"integer", "real"}:
        return "double precision"
    if kinds == {"blob"}:
        return "bytea"
    return "text"


def _clean(value, pg_type: str):
    if value is None:
        return None
    if pg_type == "text":
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return str(value).replace("\x00", "")
    return value


def convert(sqlite_path: Path, conninfo_admin: str, conninfo_for, target: str,
            log=print) -> dict:
    """Create database `target` from the SQLite file. Returns a small report."""
    t0 = time.time()
    src = sqlite3.connect(str(sqlite_path))
    src.text_factory = lambda b: b.decode("utf-8", errors="replace")
    tables = [r[0] for r in src.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")]

    with psycopg.connect(conninfo_admin, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {quote(target)} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {quote(target)} ENCODING 'UTF8' TEMPLATE template0")

    report = {"database": target, "tables": 0, "rows": 0, "primary_keys": 0,
              "foreign_keys": 0}
    with psycopg.connect(conninfo_for(target)) as dst:
        for t in tables:
            info = list(src.execute(f"PRAGMA table_info({quote(t)})"))
            cols = [(c[1], _column_type(src, t, c[1])) for c in info]
            dst.execute(f"CREATE TABLE {quote(t)} ("
                        + ", ".join(f"{quote(n)} {ty}" for n, ty in cols) + ")")
            col_list = ", ".join(quote(n) for n, _ in cols)
            n_rows = 0
            with dst.cursor().copy(f"COPY {quote(t)} ({col_list}) FROM STDIN") as cp:
                for row in src.execute(f"SELECT {col_list} FROM {quote(t)}"):
                    cp.write_row([_clean(v, ty) for v, (_, ty) in zip(row, cols)])
                    n_rows += 1
            dst.commit()
            report["tables"] += 1
            report["rows"] += n_rows
            log(f"    {t}: {n_rows:,} rows")

            pk = [c[1] for c in sorted(info, key=lambda c: c[5]) if c[5]]
            if pk:
                try:
                    dst.execute(f"ALTER TABLE {quote(t)} ADD PRIMARY KEY "
                                f"({', '.join(quote(c) for c in pk)})")
                    dst.commit()
                    report["primary_keys"] += 1
                except psycopg.Error:
                    dst.rollback()  # BIRD data has duplicate/NULL keys in places

        for t in tables:
            for fk in src.execute(f"PRAGMA foreign_key_list({quote(t)})"):
                ref_table, col, ref_col = fk[2], fk[3], fk[4]
                if not ref_col:
                    continue
                try:
                    dst.execute(f"ALTER TABLE {quote(t)} ADD FOREIGN KEY ({quote(col)}) "
                                f"REFERENCES {quote(ref_table)} ({quote(ref_col)})")
                    dst.commit()
                    report["foreign_keys"] += 1
                except psycopg.Error:
                    dst.rollback()
        dst.execute("ANALYZE")
        dst.commit()
    src.close()
    report["seconds"] = round(time.time() - t0, 1)
    return report
