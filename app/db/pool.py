"""Two connection pools, deliberately separated.

- `owner_pool`  : reads pg_catalog, writes the run log. Normal privileges.
- `readonly_pool`: stage 4 (L4/L5) and stage 5 run here, as a SELECT-only role with
  a statement timeout and a read-only transaction. Two independent guards: the SQL is
  checked by L3, *and* the connection itself cannot write.
"""
from __future__ import annotations

import atexit
import logging

from psycopg_pool import ConnectionPool

from app.core.config import get_settings

log = logging.getLogger(__name__)

_owner: ConnectionPool | None = None
_readonly: ConnectionPool | None = None


def owner_pool() -> ConnectionPool:
    global _owner
    if _owner is None:
        s = get_settings()
        _owner = ConnectionPool(s.dsn, min_size=1, max_size=4, open=True, timeout=10)
        log.info("owner pool open -> %s:%s/%s", s.pghost, s.pgport, s.pgdatabase)
    return _owner


def readonly_pool() -> ConnectionPool:
    global _readonly
    if _readonly is None:
        s = get_settings()
        _readonly = ConnectionPool(
            s.readonly_dsn,
            min_size=1,
            max_size=8,
            open=True,
            timeout=10,
            kwargs={"options": f"-c statement_timeout={s.gate_statement_timeout_ms}"},
        )
        log.info("readonly pool open (user=%s)", s.pg_readonly_user)
    return _readonly


def close_pools() -> None:
    global _owner, _readonly
    for p in (_owner, _readonly):
        if p is not None:
            p.close()
    _owner = _readonly = None


atexit.register(close_pools)
