"""L5 — run it, under limits, and look at the result before anybody else does.

A query can parse, bind, be read-only and plan cheaply, and still be wrong. Three result
shapes are suspicious often enough to be worth flagging:

* **empty** — for a question phrased as "which/what/how many", an empty result usually
  means the filter was wrong, not that the answer is "none";
* **single NULL** — an aggregate over zero matching rows (`SUM` of nothing);
* **whole-table count** — the answer equals the row count of the biggest table involved,
  i.e. the filter did nothing.

These are *warnings*, not hard failures: they lower confidence and can trigger one repair,
but an empty result can be the truth. Treating them as fatal would make the system refuse
legitimate questions, which the abstention metric would punish.
"""
from __future__ import annotations

import time
from typing import Any

from psycopg.rows import dict_row

from app.core.config import get_settings
from app.core.models import CheckResult, Layer
from app.db.pool import readonly_pool

QUESTION_EXPECTS_ROWS = ("which", "what", "who", "how many", "how much", "list", "top", "name")


def check(sql: str, question: str = "", row_limit: int | None = None, pool=None) -> CheckResult:
    s = get_settings()
    limit = row_limit or s.gate_row_limit
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []

    def done(passed: bool, message: str = "", hint: str = "", warning: bool = False,
             **details) -> CheckResult:
        return CheckResult(
            layer=Layer.L5_SNIFF, passed=passed, message=message, repair_hint=hint,
            warning=warning, duration_ms=(time.perf_counter() - t0) * 1000,
            details={"rows": rows, **details},
        )

    try:
        with (pool or readonly_pool()).connection() as conn:
            conn.read_only = True
            # row_factory on the cursor, not the connection: a pooled connection is
            # reused by other layers, and L4 expects tuples.
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(f"SET LOCAL statement_timeout = {s.gate_statement_timeout_ms}")
                cur.execute(f"SELECT * FROM ({sql}) AS t2s_probe LIMIT {limit + 1}")
                rows = [dict(r) for r in cur.fetchall()]
            conn.rollback()
    except Exception as exc:
        msg = str(exc).strip().splitlines()[0]
        return done(False, f"the query failed when executed: {msg}",
                    f"Your query failed at execution with: {msg}. Fix it.", execution_error=msg)

    truncated = len(rows) > limit
    rows = rows[:limit]
    low = question.lower()

    if not rows:
        expects = any(w in low for w in QUESTION_EXPECTS_ROWS)
        return done(
            True,
            "the query returned no rows, which is suspicious for this question" if expects
            else "the query returned no rows",
            "Your query returned no rows. Check the filter values against the example values "
            "in the schema (e.g. a status column may store 'A' rather than 'active'), and check "
            "that the date range exists in the data." if expects else "",
            warning=expects, row_count=0, truncated=False, shape="empty",
        )

    if len(rows) == 1 and all(v is None for v in rows[0].values()):
        return done(True, "the query returned a single NULL, so nothing matched the filter",
                    "Your query returned NULL, which means the aggregate matched zero rows. "
                    "Re-check the WHERE clause.", warning=True, row_count=1, shape="single_null")

    return done(True, f"returned {len(rows)}{'+' if truncated else ''} rows",
                row_count=len(rows), truncated=truncated, shape="rows")
