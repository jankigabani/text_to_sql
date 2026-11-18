"""Create the SELECT-only role that stage 5 executes as.

The L3 policy check rejects anything that is not a read. This role is the second,
independent guard: even if a write slipped past the parser, the database would refuse it.
Saying that in the paper is much stronger than "we filter the SQL".

    python scripts/03_create_readonly_role.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg   
from psycopg import sql   

from app.core.config import get_settings   


def main() -> int:
    s = get_settings()
    if not s.pg_readonly_password:
        print("[FAIL] PG_READONLY_PASSWORD is empty in .env — set it first.")
        return 1

    role = sql.Identifier(s.pg_readonly_user)
    with psycopg.connect(s.dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (s.pg_readonly_user,))
        exists = cur.fetchone() is not None

        if exists:
            # CREATE/ALTER ROLE cannot take a bind parameter, so the password is
            # quoted as a literal by psycopg rather than interpolated by hand.
            cur.execute(sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(
                role, sql.Literal(s.pg_readonly_password)))
            print(f"[ ok ] role {s.pg_readonly_user} updated")
        else:
            cur.execute(sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD {}").format(
                role, sql.Literal(s.pg_readonly_password)))
            print(f"[ ok ] role {s.pg_readonly_user} created")

        cur.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
            sql.Identifier(s.pgdatabase), role))

        cur.execute("""SELECT nspname FROM pg_namespace
                        WHERE nspname NOT LIKE 'pg\\_%' AND nspname <> 'information_schema'""")
        schemas = [r[0] for r in cur.fetchall()]

        for ns in schemas:
            ident = sql.Identifier(ns)
            cur.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(ident, role))
            cur.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(
                ident, role))
            cur.execute(sql.SQL("GRANT SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}").format(
                ident, role))
            # Anything created later is readable too, without re-running this script.
            cur.execute(sql.SQL(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA {} GRANT SELECT ON TABLES TO {}"
            ).format(ident, role))
        print(f"[ ok ] SELECT granted on {len(schemas)} schemas: {', '.join(schemas)}")

        # Explicitly take away the ability to write anywhere.
        cur.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(role))
        cur.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(role))
        cur.execute(sql.SQL("ALTER ROLE {} SET statement_timeout = {}").format(
            role, sql.Literal(str(s.gate_statement_timeout_ms))))
        print("[ ok ] role is read-only by default, with a statement timeout")

    # Prove it.
    with psycopg.connect(s.readonly_dsn, autocommit=True) as ro, ro.cursor() as cur:
        cur.execute("SELECT current_user, current_setting('default_transaction_read_only')")
        user, ro_flag = cur.fetchone()
        print(f"[ ok ] connected as {user}, read_only={ro_flag}")
        try:
            cur.execute("CREATE TABLE t2s_should_fail(x int)")
            print("[FAIL] the role could create a table — it is NOT read-only")
            return 1
        except Exception as exc:
            print(f"[ ok ] write refused as expected: {str(exc).splitlines()[0]}")

    print("\nNext: python scripts/04_build_schema_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
