"""Stage 5 — execution, with every guard the gate cannot provide.

The connection itself is the last line of defence: a SELECT-only role, a read-only
transaction, a statement timeout, and a hard row limit wrapped around the query.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from psycopg.rows import dict_row

from app.core.config import get_settings
from app.db.pool import readonly_pool

log = logging.getLogger(__name__)


@dataclass
class ExecResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    runtime_ms: float = 0.0
    error: str | None = None


def run_query(sql: str, row_limit: int | None = None, pool=None) -> ExecResult:
    s = get_settings()
    limit = row_limit or s.gate_row_limit
    t0 = time.perf_counter()
    try:
        with (pool or readonly_pool()).connection() as conn:
            conn.read_only = True
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(f"SET LOCAL statement_timeout = {s.gate_statement_timeout_ms}")
                cur.execute(f"SELECT * FROM ({sql}) AS t2s_result LIMIT {limit + 1}")
                fetched = [dict(r) for r in cur.fetchall()]
            conn.rollback()
    except Exception as exc:
        msg = str(exc).strip().splitlines()[0]
        log.warning("execution failed: %s", msg)
        return ExecResult(error=msg, runtime_ms=(time.perf_counter() - t0) * 1000)

    truncated = len(fetched) > limit
    rows = fetched[:limit]
    return ExecResult(
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        runtime_ms=(time.perf_counter() - t0) * 1000,
    )
