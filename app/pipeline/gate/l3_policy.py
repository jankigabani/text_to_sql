"""L3 — policy: is this a single, harmless read?

This is the layer the accuracy literature leaves out. It is checked on the parse tree, not
with a regex over the text, so `DELETE` hidden in a string literal is fine while a real
DELETE is not — and `-- ; DROP TABLE` style comment tricks cannot slip past, because the
parse tree has no comments in it.

Defence in depth: even if this layer were bypassed, stage 5 executes as a SELECT-only role
inside a read-only transaction. This layer exists so the attempt is *detected and logged*,
which is what the adversarial evaluation measures.
"""
from __future__ import annotations

import re
import time

import sqlglot
from sqlglot import exp

from app.core.models import CheckResult, Layer

# Statement types that must never appear anywhere in the tree.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Grant, exp.Merge, exp.Copy, exp.Command,
)

# Functions that read or write outside the queried rows.
FORBIDDEN_FUNCTIONS = {
    "pg_sleep", "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "dblink", "dblink_exec", "query_to_xml", "pg_logical_slot_get_changes",
    "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf", "set_config",
    "pg_create_physical_replication_slot", "current_setting",
}

# Writable/administrative keywords that sqlglot may surface as anonymous commands.
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert\s+into|update\s+\w|delete\s+from|drop\s+(table|database|schema|role|view)|"
    r"truncate\s+table|alter\s+(table|role|system)|create\s+(table|index|role|function)|"
    r"grant\s+|revoke\s+|vacuum\b|reindex\b|cluster\b|copy\s+\w+\s+(from|to)|"
    r"security\s+label|do\s+\$\$|call\s+\w+|set\s+role|reset\s+role)\b",
    re.I,
)

SYSTEM_SCHEMAS = {"pg_catalog", "information_schema", "pg_toast"}
SENSITIVE_TABLES = {"pg_authid", "pg_shadow", "pg_user_mapping", "pg_statistic"}


def check(sql: str, allow_system_catalogs: bool = False) -> CheckResult:
    t0 = time.perf_counter()

    def done(passed: bool, message: str = "", hint: str = "", **details) -> CheckResult:
        return CheckResult(
            layer=Layer.L3_POLICY, passed=passed, message=message, repair_hint=hint,
            duration_ms=(time.perf_counter() - t0) * 1000, details=details,
        )

    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception as exc:
        return done(False, f"unparseable at policy time: {exc}", "Your SQL did not parse.")

    if len(statements) != 1:
        return done(False, "more than one statement", "Output exactly one SELECT statement.",
                    statement_count=len(statements))

    tree = statements[0]

    for node_type in FORBIDDEN_NODES:
        node = tree.find(node_type)
        if node is not None:
            kind = node_type.__name__.upper()
            return done(False, f"blocked: the query contains a {kind} operation",
                        f"You produced a {kind} operation. Only read-only SELECT queries are "
                        "permitted. Rewrite it as a SELECT.", blocked=kind)

    if tree.find(exp.Into) is not None:
        return done(False, "blocked: SELECT ... INTO writes a new table",
                    "SELECT ... INTO creates a table. Use a plain SELECT instead.", blocked="INTO")

    for fn in tree.find_all(exp.Anonymous):
        name = (fn.this or "").lower() if isinstance(fn.this, str) else ""
        if name in FORBIDDEN_FUNCTIONS:
            return done(False, f"blocked: function {name}() is not permitted",
                        f"The function {name}() is not allowed. Answer using the data columns only.",
                        blocked_function=name)

    for tbl in tree.find_all(exp.Table):
        schema = (tbl.db or "").lower()
        name = (tbl.name or "").lower()
        is_system = schema in SYSTEM_SCHEMAS or name in SENSITIVE_TABLES or name.startswith("pg_")
        if not allow_system_catalogs and is_system:
            return done(False, f"blocked: access to system catalog {schema or name}",
                        "You referenced a PostgreSQL system catalog. Use only the application "
                        "tables in the schema you were given.", blocked_table=f"{schema}.{name}")

    # Belt and braces: the raw text, with comments and strings removed.
    stripped = _strip_literals_and_comments(sql)
    m = FORBIDDEN_KEYWORDS.search(stripped)
    if m:
        return done(False, f"blocked: '{m.group(0).strip()}' is not permitted",
                    f"Your query contains '{m.group(0).strip()}', which is not allowed. "
                    "Only a read-only SELECT is permitted.", blocked_keyword=m.group(0).strip())

    if ";" in stripped.rstrip().rstrip(";"):
        return done(False, "blocked: stacked statements", "Do not use ';' to add a second statement.",
                    blocked="stacked_statement")

    return done(True, "single read-only statement, no forbidden operations")


def _strip_literals_and_comments(sql: str) -> str:
    out, i, n = [], 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":  # string literal
            i += 1
            while i < n:
                if sql[i] == "'" and i + 1 < n and sql[i + 1] == "'":
                    i += 2
                    continue
                if sql[i] == "'":
                    i += 1
                    break
                i += 1
            out.append(" ")
            continue
        if sql.startswith("--", i):
            while i < n and sql[i] != "\n":
                i += 1
            out.append(" ")
            continue
        if sql.startswith("/*", i):
            end = sql.find("*/", i)
            i = n if end == -1 else end + 2
            out.append(" ")
            continue
        out.append(ch)
        i += 1
    return "".join(out)
