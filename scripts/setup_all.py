"""One command: load the whole dataset into PostgreSQL so it shows up in pgAdmin.

    python scripts\\setup_all.py                      # AdventureWorks + ~10M warehouse rows
    python scripts\\setup_all.py --rows 1000000       # smaller, faster
    python scripts\\setup_all.py --reload             # drop and load everything again

Steps (each one is skipped if it is already done, so it is safe to rerun):

  1  check the server and the password in .env          scripts/00_check_postgres.py
  2  load AdventureWorks: 68 tables, 5 schemas            scripts/01_load_adventureworks.py
  3  add the `warehouse` schema with millions of rows     scripts/02_amplify.py
  4  create the read-only role                            scripts/03_create_readonly_role.py
  5  build the schema index (v1 endpoints only)           scripts/04_build_schema_index.py

Afterwards, in pgAdmin: right-click Servers ▸ PostgreSQL 18 ▸ Databases ▸ Refresh, then open
adventureworks ▸ Schemas. Right-click any table ▸ View/Edit Data ▸ First 100 Rows.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:  # box-drawing characters must not crash a cp1252 console or a redirected log
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
sys.path.insert(0, str(ROOT))

import psycopg   

from app.core.config import get_settings   
from scripts._psql import find_psql   


def step(n: int, title: str) -> None:
    print(f"\n{'═' * 78}\n {n}/5  {title}\n{'═' * 78}", flush=True)


def run(script: str, *args: str) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    t0 = time.time()
    code = subprocess.call(cmd, cwd=ROOT)
    print(f"   ({script} finished in {time.time() - t0:.0f}s, exit code {code})")
    return code


def db_state(s) -> dict:
    """What is already loaded, so finished steps can be skipped."""
    out = {"exists": False, "tables": 0, "fact_rows": 0}
    try:
        with psycopg.connect(s.dsn, connect_timeout=8) as conn, conn.cursor() as cur:
            out["exists"] = True
            cur.execute("""SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                           WHERE c.relkind = 'r' AND n.nspname NOT IN ('pg_catalog','information_schema')""")
            out["tables"] = cur.fetchone()[0]
            cur.execute("SELECT to_regclass('warehouse.fact_sales') IS NOT NULL")
            if cur.fetchone()[0]:
                cur.execute("SELECT reltuples::bigint FROM pg_class WHERE oid = 'warehouse.fact_sales'::regclass")
                out["fact_rows"] = max(0, cur.fetchone()[0])
    except psycopg.OperationalError:
        pass
    return out


def summary(s) -> None:
    with psycopg.connect(s.dsn, connect_timeout=8) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT n.nspname, count(*) AS tables, sum(greatest(c.reltuples, 0))::bigint AS rows
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r' AND n.nspname NOT IN ('pg_catalog','information_schema','t2s')
            GROUP BY 1 ORDER BY 1""")
        rows = cur.fetchall()
        cur.execute("""SELECT count(*) FROM pg_constraint con JOIN pg_namespace n ON n.oid = con.connamespace
                       WHERE con.contype = 'f' AND n.nspname NOT IN ('pg_catalog','information_schema')""")
        n_fk = cur.fetchone()[0]
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        size = cur.fetchone()[0]
    print(f"\n  database '{s.pgdatabase}' — {size}, {n_fk} foreign keys\n")
    print(f"  {'schema':<18}{'tables':>8}{'rows (approx.)':>18}")
    for name, n, r in rows:
        print(f"  {name:<18}{n:>8}{r:>18,}")
    print(f"  {'total':<18}{sum(r[1] for r in rows):>8}{sum(r[2] for r in rows):>18,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=10_000_000, help="target rows for the warehouse schema")
    ap.add_argument("--reload", action="store_true", help="drop and reload everything")
    ap.add_argument("--skip-amplify", action="store_true", help="AdventureWorks only, no big tables")
    ap.add_argument("--psql", default=None, help="path to psql.exe (found automatically)")
    a = ap.parse_args()
    s = get_settings()
    t_start = time.time()

    step(1, "checking PostgreSQL and .env")
    if run("00_check_postgres.py") != 0:
        print("\nFix the [FAIL] lines above (usually PGPASSWORD in .env), then rerun.")
        return 1
    psql = find_psql(a.psql)
    if not psql:
        print("[FAIL] psql.exe not found. Rerun with  --psql \"D:\\Janki\\PostgreSQL\\18\\bin\\psql.exe\"")
        return 1
    print(f"   psql: {psql}")

    st = db_state(s)
    step(2, f"loading AdventureWorks into '{s.pgdatabase}'")
    if st["tables"] >= 60 and not a.reload:
        print(f"   already loaded ({st['tables']} tables) — skipping. Use --reload to start again.")
    elif run("01_load_adventureworks.py", "--psql", psql) != 0:
        print("\nThe loader failed — the psql errors above say why.")
        return 1

    step(3, f"adding the warehouse schema (~{a.rows:,} rows)")
    st = db_state(s)
    if a.skip_amplify:
        print("   skipped (--skip-amplify)")
    elif st["fact_rows"] >= a.rows * 0.5 and not a.reload:
        print(f"   already there (~{st['fact_rows']:,} rows in warehouse.fact_sales) — skipping.")
    else:
        args = ["--target-rows", str(a.rows)]
        if st["fact_rows"] or a.reload:
            args.append("--drop-first")
        if run("02_amplify.py", *args) != 0:
            print("\nAmplification failed; AdventureWorks itself is loaded and usable.")
            return 1

    step(4, "creating the read-only role")
    if run("03_create_readonly_role.py") != 0:
        print("   [warn] read-only role step failed. The web UI does not need it (every session it "
              "opens is read-only anyway); the v1 endpoints do.")

    step(5, "building the schema index for the v1 endpoints")
    if run("04_build_schema_index.py") != 0:
        print("   [warn] index build failed (is Ollama running with nomic-embed-text?). "
              "The web UI does not use it, so you can carry on.")

    print(f"\n{'═' * 78}\n DONE in {(time.time() - t_start) / 60:.1f} min\n{'═' * 78}")
    summary(s)
    print("""
  See it in pgAdmin:
    Servers ▸ PostgreSQL 18 ▸ Databases  →  right-click ▸ Refresh
    adventureworks ▸ Schemas ▸ sales ▸ Tables  →  right-click a table ▸ View/Edit Data

  Start the app:
    uvicorn app.main:app --reload
    then open  http://127.0.0.1:8000/
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
