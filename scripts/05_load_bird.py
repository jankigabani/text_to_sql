"""Bring the BIRD benchmark into PostgreSQL.

BIRD's files are released by the authors behind a consent page, so they cannot be fetched
unattended. This script does two things:

1. with no arguments, it prints exactly what to download and where to put it;
2. with `--convert <db_id>`, it converts one of BIRD's SQLite databases into a PostgreSQL
   schema of the same name, so the pipeline can answer BIRD questions against PostgreSQL
   rather than SQLite — which matters, because L4 uses `EXPLAIN (FORMAT JSON)` and the
   read-only role is a PostgreSQL feature.

    python scripts/05_load_bird.py                      # instructions
    python scripts/05_load_bird.py --list               # what is downloaded already
    python scripts/05_load_bird.py --convert california_schools
    python scripts/05_load_bird.py --convert-all
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg    

from app.core.config import get_settings    
from eval.datasets.bird import bird_dir    

INSTRUCTIONS = """
BIRD download steps (once)
──────────────────────────
1. Open  https://bird-bench.github.io/
2. Download either
     · "Mini-Dev" (500 questions, includes a PostgreSQL version)  — recommended, small
     · or the full dev set (1534 questions, SQLite databases)
3. Unzip it so that you end up with:

     {folder}/
       mini_dev_postgresql.json        (or dev.json)
       dev_databases/<db_id>/<db_id>.sqlite

4. Convert a database into PostgreSQL:
     python scripts/05_load_bird.py --convert california_schools

5. Point .env at it and rebuild the index:
     PGDATABASE=bird_california_schools
     INCLUDE_SCHEMAS=public
     python scripts/04_build_schema_index.py

6. Run the suite:
     python -m eval.runner --suite bird --arm ours

Note for the write-up: the Mini-Dev PostgreSQL release is the version to cite, because the
published SQLite numbers are not directly comparable across dialects.
"""

TYPE_MAP = {
    "INTEGER": "bigint", "INT": "bigint", "BIGINT": "bigint", "SMALLINT": "smallint",
    "REAL": "double precision", "FLOAT": "double precision", "DOUBLE": "double precision",
    "NUMERIC": "numeric", "DECIMAL": "numeric", "BOOLEAN": "boolean",
    "DATE": "date", "DATETIME": "timestamp", "TIMESTAMP": "timestamp",
    "TEXT": "text", "VARCHAR": "text", "CHAR": "text", "BLOB": "bytea",
}


def pg_type(sqlite_type: str) -> str:
    base = re.sub(r"\(.*\)", "", (sqlite_type or "TEXT")).strip().upper()
    for key, val in TYPE_MAP.items():
        if base.startswith(key):
            return val
    return "text"


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def convert(db_id: str, target_db: str | None = None) -> None:
    s = get_settings()
    sqlite_path = bird_dir() / "dev_databases" / db_id / f"{db_id}.sqlite"
    if not sqlite_path.exists():
        raise SystemExit(f"not found: {sqlite_path}\nRun this script with no arguments for help.")

    target = target_db or f"bird_{db_id}".lower()
    print(f"converting {sqlite_path.name} -> PostgreSQL database '{target}'")

    with psycopg.connect(f"host={s.pghost} port={s.pgport} dbname=postgres "
                         f"user={s.pguser} password={s.pgpassword}",
                         autocommit=True) as admin, admin.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS {quote(target)}')
        cur.execute(f'CREATE DATABASE {quote(target)}')

    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    tables = [r[0] for r in src.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    print(f"  {len(tables)} tables")

    dsn = (f"host={s.pghost} port={s.pgport} dbname={target} "
           f"user={s.pguser} password={s.pgpassword}")
    with psycopg.connect(dsn) as dst:
        with dst.cursor() as cur:

            for t in tables:
                cols = list(src.execute(f'PRAGMA table_info("{t}")'))
                defs = [f"{quote(c['name'])} {pg_type(c['type'])}" for c in cols]
                pks = [quote(c["name"]) for c in cols if c["pk"]]
                if pks:
                    defs.append(f"PRIMARY KEY ({', '.join(pks)})")
                cur.execute(f'CREATE TABLE {quote(t)} ({", ".join(defs)})')
            dst.commit()

            for t in tables:
                cols = [c["name"] for c in src.execute(f'PRAGMA table_info("{t}")')]
                rows = list(src.execute(f'SELECT * FROM "{t}"'))
                if not rows:
                    continue
                placeholders = ",".join(["%s"] * len(cols))
                stmt = (f'INSERT INTO {quote(t)} ({", ".join(quote(c) for c in cols)}) '
                        f"VALUES ({placeholders})")
                cur.executemany(stmt, [tuple(r) for r in rows])
                print(f"    {t}: {len(rows):,} rows")
            dst.commit()

            added = 0
            for t in tables:
                for fk in src.execute(f'PRAGMA foreign_key_list("{t}")'):
                    try:
                        cur.execute(
                            f'ALTER TABLE {quote(t)} ADD FOREIGN KEY ({quote(fk["from"])}) '
                            f'REFERENCES {quote(fk["table"])} ({quote(fk["to"])})'
                        )
                        added += 1
                        dst.commit()
                    except Exception:
                        dst.rollback() 
            print(f"  foreign keys added: {added}")
            cur.execute("ANALYZE")
            dst.commit()

    src.close()
    print(f"\ndone. Set PGDATABASE={target} and INCLUDE_SCHEMAS=public in .env, then run:")
    print("  python scripts/04_build_schema_index.py")
    print("  python scripts/03_create_readonly_role.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", metavar="DB_ID")
    ap.add_argument("--convert-all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--target-db", default=None)
    a = ap.parse_args()

    folder = bird_dir()
    if a.list:
        print(f"BIRD_DIR = {folder.resolve()}")
        for p in sorted(folder.glob("*.json")):
            print(f"  question file: {p.name}")
        dbs = sorted((folder / "dev_databases").glob("*/*.sqlite")) \
            if (folder / "dev_databases").exists() else []
        for p in dbs:
            print(f"  database:      {p.parent.name}")
        if not dbs:
            print("  (no databases found)")
        return 0

    if a.convert_all:
        dbs = sorted((folder / "dev_databases").glob("*/*.sqlite"))
        if not dbs:
            print(INSTRUCTIONS.format(folder=folder))
            return 1
        for p in dbs:
            convert(p.parent.name)
        return 0

    if a.convert:
        convert(a.convert, a.target_db)
        return 0

    print(INSTRUCTIONS.format(folder=folder.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
