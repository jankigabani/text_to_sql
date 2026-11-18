"""L1 — does it parse as PostgreSQL, and is it exactly one statement?

Cheapest possible check, and it catches the most common LLM failures for free:
truncated output, markdown fences that leaked through, prose glued onto the query, and
two statements where one was asked for.
"""
from __future__ import annotations

import time

import sqlglot
from sqlglot import exp

from app.core.models import CheckResult, Layer


def check(sql: str) -> CheckResult:
    t0 = time.perf_counter()

    def done(passed: bool, message: str = "", hint: str = "", **details) -> CheckResult:
        return CheckResult(
            layer=Layer.L1_PARSE, passed=passed, message=message, repair_hint=hint,
            duration_ms=(time.perf_counter() - t0) * 1000, details=details,
        )

    if not sql or not sql.strip():
        return done(False, "The model produced no SQL.",
                    "You produced no SQL. Output one SELECT query in a ```sql block.")

    try:
        statements = [s for s in sqlglot.parse(sql, dialect="postgres") if s is not None]
    except Exception as exc:
        return done(False, f"The SQL does not parse: {exc}",
                    f"Your SQL failed to parse with this error: {exc}. "
                    "Output valid PostgreSQL in a single ```sql block.")

    if not statements:
        return done(False, "The SQL does not parse.", "Your SQL could not be parsed. Rewrite it.")

    if len(statements) > 1:
        # "SELECT ...; DELETE ..." is an injection pattern, not a formatting slip: if any of
        # the statements is not a query, the whole request is blocked rather than repaired.
        writes = [type(st).__name__ for st in statements
                  if not isinstance(st, (exp.Select, exp.Union, exp.Subquery, exp.With,
                                         exp.Except, exp.Intersect))]
        extra = {"blocked": writes[0].upper()} if writes else {}
        return done(
            False,
            f"{'blocked: ' if writes else ''}{len(statements)} statements were produced"
            f"{' (including ' + ', '.join(writes) + ')' if writes else ''}; only one is allowed.",
            "You produced more than one SQL statement. Output exactly one SELECT statement.",
            statement_count=len(statements), **extra,
        )

    root = statements[0]
    kind = type(root).__name__
    if not isinstance(root, (exp.Select, exp.Union, exp.Subquery, exp.With, exp.Except, exp.Intersect)):
        # A write statement is a policy matter, not a typo: flagged as `blocked` so the
        # orchestrator refuses it outright instead of asking the model to try again.
        return done(False, f"blocked: the statement is a {kind}, not a query.",
                    f"You produced a {kind} statement. Only a single SELECT query is allowed.",
                    statement_type=kind, blocked=kind.upper())

    return done(True, "parses as one SELECT statement", statement_type=kind)
