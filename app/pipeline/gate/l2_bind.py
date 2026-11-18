"""L2 — do all the tables and columns actually exist?

This is the layer that catches hallucinated identifiers, deterministically and at zero
model cost. It is also the layer that produces the most useful repair hint, because it can
suggest the nearest real name:

    column orders.total does not exist; nearest match is orders.total_amount

CTE names and aliases are collected first so they are not mistaken for missing tables.
"""
from __future__ import annotations

import difflib
import time

import sqlglot
from sqlglot import exp

from app.core.models import Catalog, CheckResult, Layer


def _catalog_maps(cat: Catalog) -> tuple[dict[str, set[str]], dict[str, str]]:
    """columns_by_table[lower table name] = {lower column names}, plus a display map."""
    cols: dict[str, set[str]] = {}
    display: dict[str, str] = {}
    for t in cat.tables:
        names = {c.name.lower() for c in t.columns}
        cols[t.qualified.lower()] = names
        cols.setdefault(t.name.lower(), set()).update(names)
        display[t.qualified.lower()] = t.qualified
        display.setdefault(t.name.lower(), t.qualified)
    return cols, display


def check(sql: str, cat: Catalog) -> CheckResult:
    t0 = time.perf_counter()
    cols_by_table, display = _catalog_maps(cat)

    def done(passed: bool, message: str = "", hint: str = "", **details) -> CheckResult:
        return CheckResult(
            layer=Layer.L2_BIND, passed=passed, message=message, repair_hint=hint,
            duration_ms=(time.perf_counter() - t0) * 1000, details=details,
        )

    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except Exception as exc:
        return done(False, f"unparseable at bind time: {exc}", "Your SQL did not parse.")

    # Local names that are legitimately not in the catalog.
    local: set[str] = set()
    for cte in tree.find_all(exp.CTE):
        if cte.alias:
            local.add(cte.alias.lower())
    for sub in tree.find_all(exp.Subquery):
        if sub.alias:
            local.add(sub.alias.lower())

    # Output aliases: `SUM(qty) AS units ... ORDER BY units` is valid SQL, so `units`
    # must not be reported as a missing column.
    output_aliases: set[str] = set()
    for alias in tree.find_all(exp.Alias):
        if alias.alias:
            output_aliases.add(alias.alias.lower())

    alias_to_table: dict[str, str] = {}
    unknown_tables: list[str] = []
    used_tables: list[str] = []

    for tbl in tree.find_all(exp.Table):
        name = (tbl.name or "").lower()
        if not name or name in local:
            continue
        schema = (tbl.db or "").lower()
        qualified = f"{schema}.{name}" if schema else name
        key = qualified if qualified in cols_by_table else name
        if key not in cols_by_table:
            unknown_tables.append(qualified)
            continue
        used_tables.append(display.get(key, key))
        alias = (tbl.alias or "").lower()
        if alias:
            alias_to_table[alias] = key
        alias_to_table.setdefault(name, key)

    if unknown_tables:
        bad = unknown_tables[0]
        suggestion = _closest(bad.split(".")[-1], list(display.keys()))
        hint = f"Table {bad} does not exist in the schema you were given."
        if suggestion:
            hint += f" The closest real table is {display.get(suggestion, suggestion)}."
        return done(False, f"Unknown table(s): {', '.join(sorted(set(unknown_tables)))}",
                    hint + " Use only the tables listed in the schema.",
                    unknown_tables=sorted(set(unknown_tables)))

    # Columns. Only qualified references can be checked safely; unqualified ones are
    # checked against the union of the tables in scope.
    in_scope: set[str] = set()
    for key in set(alias_to_table.values()):
        in_scope |= cols_by_table.get(key, set())

    unknown_cols: list[tuple[str, str]] = []
    for col in tree.find_all(exp.Column):
        cname = (col.name or "").lower()
        if not cname or cname == "*":
            continue
        qualifier = (col.table or "").lower()
        if qualifier and qualifier in local:
            continue
        if not qualifier and cname in output_aliases:
            continue
        if qualifier:
            key = alias_to_table.get(qualifier)
            if key is None:
                continue  # alias of a CTE/derived table we cannot resolve: not an error
            if cname not in cols_by_table.get(key, set()):
                unknown_cols.append((f"{display.get(key, key)}.{col.name}", key))
        elif in_scope and cname not in in_scope:
            unknown_cols.append((col.name, ""))

    if unknown_cols:
        bad, key = unknown_cols[0]
        pool = sorted(cols_by_table.get(key, in_scope))
        suggestion = _closest(bad.split(".")[-1].lower(), pool)
        hint = f"Column {bad} does not exist."
        if suggestion:
            prefix = f"{display.get(key, key)}." if key else ""
            hint += f" The nearest real column is {prefix}{suggestion}."
        return done(False, f"Unknown column(s): {', '.join(c for c, _ in unknown_cols[:5])}",
                    hint + " Use only columns from the schema given to you.",
                    unknown_columns=[c for c, _ in unknown_cols])

    # Case: PostgreSQL folds unquoted names to lower case, so `CDSCode` written bare means
    # `cdscode`, which does not exist when the real column is "CDSCode". Real databases
    # (and BIRD) are full of these; catching them here costs nothing and gives a precise fix.
    exact = _exact_names(cat)
    for node in list(tree.find_all(exp.Table)) + list(tree.find_all(exp.Column)):
        ident = node.this if isinstance(node.this, exp.Identifier) else None
        if ident is None:
            continue
        written, quoted = ident.name, bool(ident.args.get("quoted"))
        if isinstance(node, exp.Table):
            if written.lower() in local:
                continue
            real = exact["tables"].get(written.lower())
        else:
            qualifier = (node.table or "").lower()
            if (not qualifier and written.lower() in output_aliases) or qualifier in local:
                continue
            key = alias_to_table.get(qualifier) if qualifier else None
            keys = [key] if key else list(set(alias_to_table.values()))
            reals = {exact["columns"].get(k, {}).get(written.lower()) for k in keys} - {None}
            real = reals.pop() if len(reals) == 1 else None
        if real is None:
            continue
        resolves = (written == real) if quoted else (written.lower() == real)
        if not resolves:
            return done(False, f'Name {written} must be written as "{real}"',
                        f'PostgreSQL will not find {written}: the real name is "{real}" '
                        f'(capitals, spaces or symbols). Write it exactly as "{real}", '
                        "including the double quotes.", case_mismatch=[written, real])

    return done(True, f"all identifiers resolve ({len(set(used_tables))} tables)",
                tables=sorted(set(used_tables)))


def _exact_names(cat: Catalog) -> dict:
    tables: dict[str, str] = {}
    columns: dict[str, dict[str, str]] = {}
    for t in cat.tables:
        tables.setdefault(t.name.lower(), t.name)
        m = {c.name.lower(): c.name for c in t.columns}
        columns[t.qualified.lower()] = m
        columns.setdefault(t.name.lower(), {}).update(m)
    return {"tables": tables, "columns": columns}


def _closest(name: str, pool: list[str]) -> str | None:
    matches = difflib.get_close_matches(name, pool, n=1, cutoff=0.6)
    return matches[0] if matches else None
