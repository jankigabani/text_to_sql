"""PostgreSQL identifier quoting, for showing names to a model exactly as SQL needs them.

AdventureWorks uses plain lower-case names, but real databases (and BIRD) have names like
`CDSCode` or `Free Meal Count (K-12)`. PostgreSQL folds unquoted names to lower case, so a
model that writes `frpm.CDSCode` gets "column cdscode does not exist". Showing such names
already quoted in the schema prompt removes that whole class of error.
"""
from __future__ import annotations

import re

_PLAIN = re.compile(r"^[a-z_][a-z0-9_$]*$")


def needs_quotes(name: str) -> bool:
    from app.pipeline.gate.l4_plan import RESERVED
    return not _PLAIN.match(name) or name in RESERVED


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"' if needs_quotes(name) else name


def qualified(schema: str, table: str) -> str:
    return f"{q(schema)}.{q(table)}"


def qualified_str(dotted: str) -> str:
    """'public.frpm' -> public.frpm ; 'public.Player Attributes' -> public."Player Attributes"."""
    schema, _, table = dotted.partition(".")
    return qualified(schema, table) if table else q(schema)
