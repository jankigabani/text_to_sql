"""Schema cards — the text we actually embed.

A "card" is a short natural-language description of one table or one column. Embedding
cards rather than raw identifiers is what lets a question like "who sold the most last
year" find a table called `salesorderheader`.
"""
from __future__ import annotations

from app.core.models import Catalog, Column, Table


def humanise(identifier: str) -> str:
    """salesorderheader -> sales order header ; ProductID -> Product ID"""
    out, prev = [], ""
    for ch in identifier.replace("_", " "):
        if ch.isupper() and prev and not prev.isupper() and prev != " ":
            out.append(" ")
        out.append(ch)
        prev = ch
    words = " ".join("".join(out).split())
    for long, short in (("Qty", "quantity"), ("Num", "number"), ("Amt", "amount"),
                        ("Pct", "percent"), ("Desc", "description"), ("ID", "id")):
        words = words.replace(long, short)
    return words.lower()


def table_card(t: Table, fk_names: list[str]) -> str:
    cols = ", ".join(humanise(c.name) for c in t.columns[:25])
    parts = [
        f"table {t.schema_name}.{t.name} ({humanise(t.name)})",
        f"columns: {cols}" if cols else "",
        f"comment: {t.comment}" if t.comment else "",
        f"related to: {', '.join(fk_names)}" if fk_names else "",
        f"approximately {t.estimated_rows} rows" if t.estimated_rows else "",
    ]
    return ". ".join(p for p in parts if p)


def column_card(c: Column, t: Table) -> str:
    parts = [
        f"column {c.name} ({humanise(c.name)}) of table {c.table_name} ({humanise(c.table_name)})",
        f"type {c.data_type}",
        "primary key" if c.is_primary_key else "",
        f"comment: {c.comment}" if c.comment else "",
        f"example values: {', '.join(c.sample_values[:4])}" if c.sample_values else "",
    ]
    return ". ".join(p for p in parts if p)


def build_cards(cat: Catalog) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Returns (table_cards, column_cards) as (key, text) pairs."""
    neighbours: dict[str, list[str]] = {}
    for fk in cat.foreign_keys:
        neighbours.setdefault(fk.left, []).append(humanise(fk.ref_table))
        neighbours.setdefault(fk.right, []).append(humanise(fk.table_name))

    tcards = [(t.qualified, table_card(t, neighbours.get(t.qualified, [])[:8])) for t in cat.tables]
    ccards = [
        (c.qualified, column_card(c, t))
        for t in cat.tables
        for c in t.columns
    ]
    return tcards, ccards
