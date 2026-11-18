"""Stage 2 — bidirectional schema retrieval.

Three paths, unioned:

1. **table-first**  — embed the question, find the best table cards, keep their columns.
2. **column-first** — find the best column cards, then pull in their parent tables. This
   recovers tables whose *name* says nothing useful ("dimension_3") but whose columns do.
3. **value lookup** — if the question contains a literal ("Thunder Bay"), find columns that
   actually contain that string, and pull in their tables.

Then `fk_closure` adds any table needed to connect the chosen ones, so the generator is
never asked to invent a bridge table it cannot see.

`retrieval_margin` (the gap between the last kept table and the first dropped one) is fed
into the confidence score: a narrow margin means retrieval was unsure, which is exactly
when the answer deserves less trust.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from psycopg import sql

from app.core.config import get_settings
from app.core.models import Catalog, SubSchema, Table
from app.db.pool import readonly_pool
from app.retrieval.index import SchemaIndex
from app.retrieval.question import ExpandedQuestion, expand

log = logging.getLogger(__name__)


def retrieve(
    question: str,
    cat: Catalog,
    idx: SchemaIndex,
    top_tables: int | None = None,
    top_columns: int | None = None,
    value_lookup: bool = True,
) -> SubSchema:
    s = get_settings()
    top_tables = top_tables or s.retrieve_top_tables
    top_columns = top_columns or s.retrieve_top_columns

    eq = expand(question)
    qvec = idx.embed_query(eq.search_text)

    scores: dict[str, float] = defaultdict(float)
    reasons: dict[str, str] = {}

    # 1 ── table-first
    table_hits = idx.tables.search(qvec, top_tables * 2)
    for key, score in table_hits[:top_tables]:
        scores[key] += score
        reasons.setdefault(key, "matched the table description")

    # 2 ── column-first
    col_hits = idx.columns.search(qvec, top_columns)
    for key, score in col_hits:
        parent = ".".join(key.split(".")[:2])
        scores[parent] += 0.6 * score
        reasons.setdefault(parent, f"column {key.split('.')[-1]} matched the question")

    # 3 ── value lookup
    if value_lookup and eq.literals:
        for parent, col, literal in _tables_containing(cat, eq):
            scores[parent] += 0.5
            reasons[parent] = f"column {col} contains the value '{literal}'"

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    kept = [k for k, _ in ranked[:top_tables]]
    margin = _margin(ranked, top_tables)

    # 4 ── FK closure
    added = fk_closure(cat, kept)
    for key in added:
        reasons[key] = "needed to join the selected tables"

    tables = [t for t in (cat.table(k) for k in kept + added) if t is not None]
    tables = _dedupe(tables)
    names = {t.qualified for t in tables}
    fks = [f for f in cat.foreign_keys if f.left in names and f.right in names]

    sub = SubSchema(
        tables=tables,
        foreign_keys=fks,
        reasons={k: v for k, v in reasons.items() if k in names},
        table_scores={k: round(v, 4) for k, v in ranked[: top_tables * 2]},
        retrieval_margin=round(margin, 4),
        n_columns=sum(len(t.columns) for t in tables),
    )
    log.info(
        "retrieved %d/%d tables (%d cols), margin=%.3f",
        len(tables), len(cat.tables), sub.n_columns, sub.retrieval_margin,
    )
    return sub


def fk_closure(cat: Catalog, chosen: list[str], max_added: int = 4) -> list[str]:
    """Add bridge tables that connect two chosen tables through one hop."""
    chosen_set = set(chosen)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for fk in cat.foreign_keys:
        adjacency[fk.left].add(fk.right)
        adjacency[fk.right].add(fk.left)

    added: list[str] = []
    for mid, neighbours in adjacency.items():
        if mid in chosen_set or len(added) >= max_added:
            continue
        if len(neighbours & chosen_set) >= 2:
            added.append(mid)
    return added


def _tables_containing(cat: Catalog, eq: ExpandedQuestion) -> list[tuple[str, str, str]]:
    """Cheap value lookup: only text-ish columns, only equality, always LIMIT 1."""
    out: list[tuple[str, str, str]] = []
    candidates = [
        (t, c)
        for t in cat.tables
        if t.estimated_rows and t.estimated_rows < 5_000_000
        for c in t.columns
        if any(k in c.data_type for k in ("char", "text", "name"))
    ][:400]
    if not candidates:
        return out

    try:
        with readonly_pool().connection() as conn:
            conn.read_only = True
            for literal in eq.literals[:4]:
                for t, c in candidates:
                    q = sql.SQL("SELECT 1 FROM {}.{} WHERE {} = %s LIMIT 1").format(
                        sql.Identifier(t.schema_name), sql.Identifier(t.name),
                        sql.Identifier(c.name),
                    )
                    try:
                        with conn.cursor() as cur:
                            cur.execute(q, (literal,))
                            if cur.fetchone():
                                out.append((t.qualified, c.name, literal))
                                break
                    except Exception:
                        conn.rollback()
                        continue
    except Exception as exc:
        log.debug("value lookup skipped (%s)", exc)
    return out


def _margin(ranked: list[tuple[str, float]], k: int) -> float:
    if len(ranked) <= k:
        return 1.0
    last_kept, first_dropped = ranked[k - 1][1], ranked[k][1]
    if last_kept <= 0:
        return 0.0
    return max(0.0, min(1.0, (last_kept - first_dropped) / last_kept))


def _dedupe(tables: list[Table]) -> list[Table]:
    seen: set[str] = set()
    out = []
    for t in tables:
        if t.qualified not in seen:
            seen.add(t.qualified)
            out.append(t)
    return out


def full_schema(cat: Catalog) -> SubSchema:
    """Baseline A: hand the generator everything. The thing RQ1 measures against."""
    return SubSchema(
        tables=cat.tables,
        foreign_keys=cat.foreign_keys,
        reasons={t.qualified: "baseline: full schema" for t in cat.tables},
        retrieval_margin=0.0,
        n_columns=sum(len(t.columns) for t in cat.tables),
    )
