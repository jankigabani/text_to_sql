"""Pre-flight check: can we reach PostgreSQL, and is it set up the way the pipeline needs?

Run this first. It prints exactly what to fix, and nothing else in the project will work
until it passes.

    python scripts/00_check_postgres.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg   

from app.core.config import get_settings   

OK, BAD, WARN = "[ ok ]", "[FAIL]", "[warn]"


def main() -> int:
    s = get_settings()
    print(f"\nConnecting to {s.pguser}@{s.pghost}:{s.pgport}/{s.pgdatabase}\n")
    failures = 0

    try:
        conn = psycopg.connect(s.dsn, connect_timeout=8)
    except Exception as exc:
        message = str(exc)

        # The target database not existing yet is the normal state before the loader has
        # run, and it is not an authentication problem - say so, and carry on checking the
        # server itself by connecting to the built-in `postgres` database.
        if "does not exist" in message and s.pgdatabase in message:
            print(f"{WARN} database '{s.pgdatabase}' does not exist yet "
                  "(expected before step 3 - the loader creates it)")
            print(f"{OK} the server accepted your credentials, so PGPASSWORD is correct")
            try:
                conn = psycopg.connect(
                    f"host={s.pghost} port={s.pgport} dbname=postgres "
                    f"user={s.pguser} password={s.pgpassword}",
                    connect_timeout=8,
                )
                print(f"{OK} connected to the 'postgres' database instead\n")
            except Exception as inner:
                print(f"{BAD} could not connect to 'postgres' either: {inner}")
                return 1
        else:
            print(f"{BAD} cannot connect: {exc}")
            print("\n  Check that PostgreSQL is running, and that PGPASSWORD in .env is correct.")
            print("  On Windows: Get-Service *postgres*  should show it Running.")
            print("  Forgot the password? See SETUP_WINDOWS.md for the reset steps.")
            return 1

    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            print(f"{OK} connected: {version.split(',')[0]}")

            major = int(version.split()[1].split(".")[0])
            if major < 13:
                print(f"{WARN} PostgreSQL {major} is old; 14+ recommended (EXPLAIN FORMAT JSON "
                      "behaviour used by L4 is stable from 12 onwards)")

            cur.execute("SELECT current_setting('server_encoding')")
            print(f"{OK} encoding: {cur.fetchone()[0]}")

            cur.execute("""SELECT count(*) FROM pg_class c JOIN pg_namespace n
                           ON n.oid = c.relnamespace
                           WHERE c.relkind IN ('r','p') AND n.nspname = ANY(%s)""",
                        (s.schema_allowlist,))
            n_tables = cur.fetchone()[0]
            if n_tables == 0:
                print(f"{WARN} no tables found in schemas {s.schema_allowlist}")
                print("       run: python scripts/01_load_adventureworks.py")
            else:
                print(f"{OK} tables visible: {n_tables}")

            cur.execute("""SELECT count(*) FROM pg_constraint con JOIN pg_class c
                           ON c.oid = con.conrelid JOIN pg_namespace n ON n.oid = c.relnamespace
                           WHERE con.contype = 'f' AND n.nspname = ANY(%s)""",
                        (s.schema_allowlist,))
            print(f"{OK} foreign keys: {cur.fetchone()[0]}")

            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (s.pg_readonly_user,))
            if cur.fetchone():
                print(f"{OK} read-only role '{s.pg_readonly_user}' exists")
            else:
                print(f"{WARN} read-only role '{s.pg_readonly_user}' missing")
                print("       run: python scripts/03_create_readonly_role.py")

            cur.execute("SELECT count(*) FROM pg_stats WHERE schemaname = ANY(%s)",
                        (s.schema_allowlist,))
            if cur.fetchone()[0] == 0 and n_tables:
                print(f"{WARN} no planner statistics — L4 cost estimates will be meaningless")
                print("       run: ANALYZE;")
            elif n_tables:
                print(f"{OK} planner statistics present (L4 cost check will work)")

    # The read-only role must actually be able to connect.
    if s.pg_readonly_password:
        try:
            with psycopg.connect(s.readonly_dsn, connect_timeout=8) as ro:
                with ro.cursor() as cur:
                    cur.execute("SELECT current_user")
                    print(f"{OK} read-only connection works as {cur.fetchone()[0]}")
                    try:
                        cur.execute("CREATE TEMP TABLE should_not_work(x int)")
                        print(f"{BAD} the read-only role can create tables — it is not read-only")
                        failures += 1
                    except Exception:
                        ro.rollback()
                        print(f"{OK} the read-only role cannot write (as intended)")
        except Exception as exc:
            print(f"{WARN} read-only connection failed: {exc}")

    print("\nAll good." if failures == 0 else f"\n{failures} problem(s) to fix.")
    print("\nNext step: python scripts/01_load_adventureworks.py")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
