"""Connection pools for every database the UI has talked to, plus a connection test.

Pools are cached by target fingerprint (least-recently-used, small), so switching between
two databases in the UI does not reconnect every time, and a stale database's pool is
closed once it falls out of the cache.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

import psycopg
from psycopg_pool import ConnectionPool

from app.core.config import get_settings
from app.engine.target import DBTarget

log = logging.getLogger(__name__)

_pools: "OrderedDict[str, ConnectionPool]" = OrderedDict()
_lock = threading.Lock()


class ConnectionFailed(Exception):
    """A connection error with a message written for a person, not a stack trace."""


def explain_connection_error(exc: Exception, target: DBTarget) -> str:
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    low = msg.lower()
    if "password authentication failed" in low:
        return f"Wrong password for user '{target.user}'."
    if "does not exist" in low and "database" in low:
        return f"Database '{target.database}' does not exist on {target.host}:{target.port}."
    if "role" in low and "does not exist" in low:
        return f"User '{target.user}' does not exist on this server."
    if "connection refused" in low or "could not connect" in low or "timeout" in low:
        return (f"Could not reach PostgreSQL at {target.host}:{target.port}. "
                "Is the server running, and is the port right?")
    return msg


def test_connection(target: DBTarget) -> dict:
    """Open one connection directly (not through a pool) so failures are fast and clear."""
    t0 = time.perf_counter()
    try:
        with psycopg.connect(target.conninfo()) as conn, conn.cursor() as cur:
            latency = (time.perf_counter() - t0) * 1000
            cur.execute("SELECT version(), current_database(), current_user, "
                        "current_setting('server_version')")
            version_full, db, user, version = cur.fetchone()
            cur.execute("""SELECT datname FROM pg_database
                            WHERE datallowconn AND NOT datistemplate ORDER BY 1""")
            databases = [r[0] for r in cur.fetchall()]
            cur.execute("""SELECT count(*) FILTER (WHERE c.relkind IN ('r','p')),
                                  count(DISTINCT n.nspname)
                             FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                            WHERE n.nspname NOT IN ('pg_catalog','information_schema')
                              AND n.nspname NOT LIKE 'pg\\_%'""")
            n_tables, n_schemas = cur.fetchone()
    except Exception as exc:
        raise ConnectionFailed(explain_connection_error(exc, target)) from exc

    return {
        "ok": True,
        "server_version": version,
        "server": version_full.split(",")[0],
        "database": db,
        "user": user,
        "latency_ms": round(latency, 1),
        "databases": databases,
        "tables": n_tables,
        "schemas": n_schemas,
    }


def pool_for(target: DBTarget) -> ConnectionPool:
    s = get_settings()
    key = target.fingerprint()
    with _lock:
        pool = _pools.get(key)
        if pool is not None:
            _pools.move_to_end(key)
            return pool
        pool = ConnectionPool(target.conninfo(), min_size=0, max_size=4, open=True,
                              timeout=15, name=f"t2s-{key[:6]}")
        _pools[key] = pool
        while len(_pools) > s.connection_cache_size:
            old_key, old = _pools.popitem(last=False)
            log.info("closing idle pool %s", old_key[:6])
            try:
                old.close()
            except Exception:
                pass
        log.info("pool opened for %s@%s:%s/%s", target.user, target.host, target.port,
                 target.database)
        return pool


def close_all() -> None:
    with _lock:
        for p in _pools.values():
            try:
                p.close()
            except Exception:
                pass
        _pools.clear()
