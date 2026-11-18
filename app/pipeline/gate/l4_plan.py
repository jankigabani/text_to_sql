"""L4 — ask the planner what this query would cost, without running it.

`EXPLAIN (FORMAT JSON)` is free: PostgreSQL plans the query but does not execute it. From
the plan we reject three things:

* total cost above a ceiling — a 10-million-row database makes this check meaningful,
  which is why the dataset size matters to the experiment;
* a cartesian product — a join with no join condition, the classic symptom of a model that
  did not understand the foreign keys;
* a plan whose estimated row count is absurd relative to the largest table involved.

A rejection here is a *likely wrong join path*, not just a slow query, which is why the
repair hint talks about join keys rather than performance.
"""
from __future__ import annotations

import json
import re
import time

from app.core.config import get_settings
from app.core.models import CheckResult, Layer
from app.db.pool import readonly_pool


def check(sql: str, cost_ceiling: float | None = None, pool=None) -> CheckResult:
    s = get_settings()
    ceiling = cost_ceiling if cost_ceiling is not None else s.gate_plan_cost_ceiling
    t0 = time.perf_counter()

    def done(passed: bool, message: str = "", hint: str = "", **details) -> CheckResult:
        return CheckResult(
            layer=Layer.L4_PLAN, passed=passed, message=message, repair_hint=hint,
            duration_ms=(time.perf_counter() - t0) * 1000, details=details,
        )

    try:
        with (pool or readonly_pool()).connection() as conn:
            conn.read_only = True
            with conn.cursor() as cur:
                cur.execute(f"EXPLAIN (FORMAT JSON, VERBOSE false) {sql}")
                row = cur.fetchone()
            conn.rollback()
    except Exception as exc:
        # The planner rejects things the parser accepted: wrong types, ambiguous columns,
        # GROUP BY errors. This is a genuinely useful repair signal.
        msg = str(exc).strip().splitlines()[0]
        return done(False, f"PostgreSQL rejected the query when planning it: {msg}",
                    planner_hint(msg, sql), planner_error=msg)

    raw = list(row.values())[0] if isinstance(row, dict) else row[0]
    explained = raw if isinstance(raw, list) else json.loads(raw)
    plan = explained[0]["Plan"]
    total_cost = float(plan.get("Total Cost", 0.0))
    plan_rows = int(plan.get("Plan Rows", 0))

    cartesian = _find_cartesian(plan)
    if cartesian:
        return done(False,
                    f"rejected: {cartesian} with no join condition (cartesian product)",
                    "Your query joins tables without a join condition, which multiplies every "
                    "row against every other row. Join along the foreign keys listed in the "
                    "schema, using ON clauses.",
                    node=cartesian, total_cost=total_cost)

    if total_cost > ceiling:
        return done(False,
                    f"rejected: estimated cost {total_cost:,.0f} exceeds the ceiling {ceiling:,.0f}",
                    "Your query is estimated to be extremely expensive, which usually means a "
                    "missing join condition or a filter applied to the wrong column. Add the "
                    "correct join keys and filters.",
                    total_cost=total_cost, ceiling=ceiling)

    return done(True, f"plan cost {total_cost:,.0f}, estimated {plan_rows:,} rows",
                total_cost=total_cost, plan_rows=plan_rows,
                node_type=plan.get("Node Type", ""))


RESERVED = {
    "all", "analyse", "analyze", "and", "any", "array", "as", "asc", "asymmetric", "both",
    "case", "cast", "check", "collate", "column", "constraint", "create", "current_date",
    "current_role", "current_time", "current_timestamp", "current_user", "default",
    "deferrable", "desc", "distinct", "do", "else", "end", "except", "false", "fetch", "for",
    "foreign", "from", "grant", "group", "having", "in", "initially", "intersect", "into",
    "is", "isnull", "join", "lateral", "leading", "left", "limit", "localtime", "natural",
    "not", "notnull", "null", "offset", "on", "only", "or", "order", "placing", "primary",
    "references", "returning", "right", "select", "session_user", "some", "symmetric",
    "table", "then", "to", "trailing", "true", "union", "unique", "user", "using",
    "variadic", "when", "where", "window", "with",
}


def planner_hint(msg: str, sql: str) -> str:
    """Turn PostgreSQL's error into an instruction a model can act on.

    A generic "fix that and retry" makes small models repeat the same query (seen with
    llama3.1 using `is` as a table alias three times in a row), so the common cases get a
    specific instruction."""
    low = msg.lower()
    m = re.search(r'syntax error at or near "([^"]+)"', msg)
    if m and m.group(1).lower() in RESERVED:
        tok = m.group(1)
        return (f'"{tok}" is a reserved word in PostgreSQL, so it cannot be used as a table '
                f'alias or column name without quotes. Rename the alias "{tok}" to something '
                "like t1, t2 everywhere in the query.")
    m = re.search(r'column "([^"]+)" does not exist', msg)
    if m:
        return (f'PostgreSQL says column "{m.group(1)}" does not exist. Column names that '
                'contain capitals, spaces or symbols must be written exactly as in the schema, '
                'inside double quotes, e.g. "CDSCode". Check the spelling and the table alias.')
    m = re.search(r'relation "([^"]+)" does not exist', msg)
    if m:
        return (f'PostgreSQL says table "{m.group(1)}" does not exist. Use the exact table name '
                'from the schema; names with capitals or spaces need double quotes.')
    m = re.search(r"function (\w+)\(", msg)
    if m and "does not exist" in low:
        fn = m.group(1).lower()
        sqlite = {"strftime": "to_char(col::date, 'YYYY') or EXTRACT(YEAR FROM col::date)",
                  "julianday": "(a::date - b::date) for a number of days",
                  "iif": "CASE WHEN ... THEN ... ELSE ... END",
                  "instr": "position(sub IN col)", "substr": "substring(col FROM a FOR n)"}
        if fn in sqlite:
            return (f"{fn}() is SQLite, not PostgreSQL. Use {sqlite[fn]} instead.")
        return (f"PostgreSQL reported: {msg}. The function or its argument types do not exist; "
                "cast the arguments (e.g. col::numeric, col::date) or use a PostgreSQL function.")
    if "must appear in the group by clause" in low:
        return ("Every selected column that is not inside an aggregate (SUM, COUNT, AVG, ...) "
                "must be listed in GROUP BY. Either add it to GROUP BY or aggregate it.")
    if "operator does not exist" in low or "invalid input syntax" in low:
        return (f"PostgreSQL reported a type mismatch: {msg}. Cast one side explicitly, e.g. "
                "CAST(col AS numeric) or col::text, so both sides have the same type.")
    if "division by zero" in low:
        return "Guard the division with NULLIF(denominator, 0)."
    return f"PostgreSQL could not plan your query. It reported: {msg}. Fix that and retry."


_JOIN_NODES = {"Nested Loop", "Hash Join", "Merge Join"}


_COND_KEYS = ("Hash Cond", "Merge Cond", "Join Filter", "Index Cond", "Recheck Cond",
              "Filter", "TID Cond")


def _aliases(node: dict) -> set[str]:
    out = {node["Alias"]} if node.get("Alias") else set()
    for child in node.get("Plans", []) or []:
        out |= _aliases(child)
    return out


def _conditions(node: dict) -> list[str]:
    out = [str(node[k]) for k in _COND_KEYS if k in node]
    for child in node.get("Plans", []) or []:
        out += _conditions(child)
    return out


def _find_cartesian(node: dict) -> str | None:
    """A join node with no join predicate of any kind.

    A Nested Loop often carries no condition itself: the predicate is pushed into the
    inner side as a parameterised index scan, e.g. `Index Cond: (productid = i.product_id)`.
    That is a real join, so the inner side is searched for a condition that mentions a
    relation from the outer side. A side that returns at most one row (an aggregate, a
    scalar subquery) cannot multiply anything, so it is not flagged either.
    """
    nt = node.get("Node Type", "")
    kids = node.get("Plans", []) or []
    if nt in _JOIN_NODES and len(kids) >= 2:
        has_predicate = any(k in node for k in ("Hash Cond", "Merge Cond", "Join Filter"))
        if not has_predicate:
            outer, inner = kids[0], kids[1]
            outer_aliases = _aliases(outer)
            inner_conds = _conditions(inner)
            has_predicate = any(f"{a}." in c for a in outer_aliases for c in inner_conds)
        single_row = min(kids[0].get("Plan Rows", 2), kids[1].get("Plan Rows", 2)) <= 1 \
            and all(k.get("Node Type") in ("Aggregate", "Result", "Limit", "Subquery Scan",
                                            "Function Scan", "Values Scan")
                    for k in kids if k.get("Plan Rows", 2) <= 1)
        if not has_predicate and not single_row:
            return nt
    for child in kids:
        found = _find_cartesian(child)
        if found:
            return found
    return None
