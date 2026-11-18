"""Stage 3 — the LLM shrinks the schema to what the question needs.

The model reads a compact rendering of the *whole* schema (every table, every column, its
type, its keys) and returns, as JSON, the tables and columns needed to answer the
question. That shrunk schema is all the SQL-writing step ever sees.

The model's choice is never trusted blindly:

* every table and column it names is checked against the real catalog — invented names
  are dropped and *counted* (``hallucinated_tables`` / ``hallucinated_columns``), which is
  a measurable result in its own right;
* primary keys and the foreign-key columns that connect the chosen tables are always
  added, because a model that picks the right tables but forgets the join key produces
  a cartesian product;
* bridge tables needed to connect two chosen tables are added (FK closure);
* if the full schema is too large for the model's context, a keyword pre-filter narrows
  it first; if the model's reply is unusable, a keyword ranking takes over. Both are
  recorded in ``strategy`` so the result says exactly what happened.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
import time
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.ident import q, qualified, qualified_str
from app.core.models import Catalog, SubSchema, Table
from app.retrieval.cards import humanise
from app.retrieval.embedder import HashingEmbedder, cosine
from app.retrieval.retriever import fk_closure

log = logging.getLogger(__name__)

SYSTEM = """You are a database expert. You are given a PostgreSQL schema and a question.
Select ONLY the tables and columns needed to write one SQL query that answers the question.

Rules:
- Use the exact table and column names from the schema. Never invent a name.
- Include every table needed to JOIN the data together, including bridge tables, and the
  key columns used in those joins.
- Include the columns needed for SELECT, WHERE, GROUP BY, ORDER BY and any aggregate.
- Prefer as few tables as possible. Select at most {max_tables} tables.
- If the question cannot be answered from this schema at all, set "answerable" to false
  and return an empty "tables" list.

Reply with JSON only, exactly in this shape:
{{"answerable": true, "tables": [{{"table": "schema.table", "columns": ["column_a", "column_b"], "reason": "why it is needed"}}]}}"""

USER = """# Schema ({n_tables} tables)
Format: schema.table (~rows): column type [PK] [-> referenced table.column], ...

{schema}

# Question

{question}"""


@dataclass
class Selection:
    table: str
    columns: list[str]
    reason: str
    source: str  # llm | bridge | fallback


@dataclass
class PruneResult:
    sub: SubSchema
    selections: list[Selection]
    strategy: str
    answerable: bool = True
    hallucinated_tables: list[str] = field(default_factory=list)
    hallucinated_columns: list[str] = field(default_factory=list)
    corrected_tables: dict[str, str] = field(default_factory=dict)
    prefiltered_to: int | None = None
    tables_before: int = 0
    columns_before: int = 0
    prompt_tokens: int = 0
    full_schema_tokens: int = 0
    shrunk_schema_tokens: int = 0
    llm_ms: float = 0.0
    llm_raw: str = ""
    model: str = ""
    error: str | None = None

    @property
    def tables_after(self) -> int:
        return len(self.sub.tables)

    @property
    def columns_after(self) -> int:
        return sum(len(t.columns) for t in self.sub.tables)

    def summary(self) -> dict:
        reduction = 0.0
        if self.full_schema_tokens:
            reduction = 100 * (1 - self.shrunk_schema_tokens / self.full_schema_tokens)
        return {
            "strategy": self.strategy,
            "answerable": self.answerable,
            "tables_before": self.tables_before,
            "tables_after": self.tables_after,
            "columns_before": self.columns_before,
            "columns_after": self.columns_after,
            "full_schema_tokens": self.full_schema_tokens,
            "shrunk_schema_tokens": self.shrunk_schema_tokens,
            "reduction_pct": round(max(0.0, reduction), 1),
            "prefiltered_to": self.prefiltered_to,
            "hallucinated_tables": self.hallucinated_tables,
            "hallucinated_columns": self.hallucinated_columns,
            "corrected_tables": self.corrected_tables,
            "model": self.model,
            "llm_ms": round(self.llm_ms, 1),
            "error": self.error,
        }


# ── rendering ─────────────────────────────────────────────────────────────────
_TYPE_SHORT = [
    ("character varying", "varchar"), ("timestamp without time zone", "timestamp"),
    ("timestamp with time zone", "timestamptz"), ("double precision", "float8"),
    ("integer", "int"), ("boolean", "bool"), ("character", "char"),
]


def short_type(t: str) -> str:
    for long, short in _TYPE_SHORT:
        t = t.replace(long, short)
    return t


def fk_map(cat: Catalog) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for fk in cat.foreign_keys:
        for col, ref in zip(fk.columns, fk.ref_columns):
            out[(fk.left, col)] = f"{qualified_str(fk.right)}.{q(ref)}"
    return out


def render_compact(tables: list[Table], fks: dict[tuple[str, str], str]) -> str:
    lines = []
    for t in sorted(tables, key=lambda x: x.qualified):
        cols = []
        for c in t.columns:
            bit = f"{q(c.name)} {short_type(c.data_type)}"
            if c.is_primary_key:
                bit += " PK"
            ref = fks.get((t.qualified, c.name))
            if ref:
                bit += f" -> {ref}"
            cols.append(bit)
        rows = f" (~{t.estimated_rows:,} rows)" if t.estimated_rows else ""
        lines.append(f"{qualified(t.schema_name, t.name)}{rows}: {', '.join(cols)}")
    return "\n".join(lines)


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── keyword ranking (pre-filter and fallback; no model needed) ────────────────
def keyword_rank(cat: Catalog, question: str, k: int) -> list[tuple[str, float]]:
    cards = []
    for t in cat.tables:
        words = " ".join([t.qualified, humanise(t.name), t.comment or ""] +
                         [f"{c.name} {humanise(c.name)}" for c in t.columns])
        cards.append((t.qualified, words))
    emb = HashingEmbedder()
    emb.fit([w for _, w in cards])
    vecs = emb.embed([w for _, w in cards])
    q = emb.embed([question])[0]
    scored = sorted(((key, cosine(q, v)) for (key, _), v in zip(cards, vecs)),
                    key=lambda kv: kv[1], reverse=True)
    return scored[:k]


# ── parsing the model's reply ─────────────────────────────────────────────────
def _extract_json(text: str) -> object:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON object in the reply")
    for end in range(len(text), start, -1):
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
    raise ValueError("the reply is not valid JSON")


def parse_selection(text: str) -> tuple[bool, list[dict]]:
    obj = _extract_json(text)
    answerable = True
    items: list = []
    if isinstance(obj, dict):
        answerable = bool(obj.get("answerable", True))
        items = obj.get("tables") or obj.get("relevant_tables") or obj.get("schema") or []
    elif isinstance(obj, list):
        items = obj
    out = []
    for it in items:
        if isinstance(it, str):
            out.append({"table": it, "columns": [], "reason": ""})
        elif isinstance(it, dict):
            name = it.get("table") or it.get("name") or it.get("table_name") or ""
            cols = it.get("columns") or it.get("cols") or []
            if isinstance(cols, str):
                cols = [c.strip() for c in cols.split(",")]
            out.append({"table": str(name), "columns": [str(c) for c in cols],
                        "reason": str(it.get("reason", ""))[:200]})
    return answerable, out


# ── validation against the real catalog ───────────────────────────────────────
def _resolve_table(name: str, cat: Catalog) -> tuple[Table | None, bool]:
    """Returns (table, corrected?)."""
    want = name.strip().strip('"').lower()
    by_q = {t.qualified.lower(): t for t in cat.tables}
    if want in by_q:
        return by_q[want], False
    bare = want.split(".")[-1]
    matches = [t for t in cat.tables if t.name.lower() == bare]
    if len(matches) == 1:
        return matches[0], False
    close = difflib.get_close_matches(want, list(by_q), n=1, cutoff=0.88)
    if close:
        return by_q[close[0]], True
    return None, False


def validate(raw: list[dict], cat: Catalog, prune_columns: bool, max_tables: int) -> tuple[
        list[Selection], list[str], list[str], dict[str, str]]:
    selections: list[Selection] = []
    bad_tables: list[str] = []
    bad_cols: list[str] = []
    corrected: dict[str, str] = {}
    seen: set[str] = set()

    for item in raw:
        t, was_corrected = _resolve_table(item["table"], cat)
        if t is None:
            if item["table"]:
                bad_tables.append(item["table"])
            continue
        if was_corrected:
            corrected[item["table"]] = t.qualified
        if t.qualified in seen:
            continue
        seen.add(t.qualified)

        real = {c.name.lower(): c.name for c in t.columns}
        cols: list[str] = []
        for c in item["columns"]:
            bare = c.strip().strip('"').split(".")[-1].lower()
            if bare in real:
                cols.append(real[bare])
            elif bare and bare != "*":
                bad_cols.append(f"{t.qualified}.{c}")
        if not prune_columns or not cols or "*" in item["columns"]:
            cols = [c.name for c in t.columns]  # no usable column list: keep the whole table
        selections.append(Selection(t.qualified, cols, item["reason"], "llm"))
        if len(selections) >= max_tables:
            break
    return selections, bad_tables, bad_cols, corrected


def build_subschema(cat: Catalog, selections: list[Selection]) -> SubSchema:
    """Apply the selection, then add keys and bridge tables so joins stay possible."""
    chosen = {s.table for s in selections}
    for bridge in fk_closure(cat, list(chosen), max_added=3):
        t = cat.table(bridge)
        if t is not None:
            selections.append(Selection(t.qualified, [], "connects two selected tables",
                                        "bridge"))
            chosen.add(t.qualified)

    keep: dict[str, set[str]] = {s.table: set(s.columns) for s in selections}
    for s in selections:
        t = cat.table(s.table)
        if t is not None:
            keep[s.table] |= set(t.primary_key)
    for fk in cat.foreign_keys:
        if fk.left in chosen and fk.right in chosen:
            keep[fk.left] |= set(fk.columns)
            keep[fk.right] |= set(fk.ref_columns)

    tables: list[Table] = []
    for s in selections:
        t = cat.table(s.table)
        if t is None:
            continue
        cols = [c for c in t.columns if c.name in keep[s.table]]
        s.columns = [c.name for c in cols]
        tables.append(t.model_copy(update={"columns": cols}))

    names = {t.qualified for t in tables}
    fks = [f for f in cat.foreign_keys if f.left in names and f.right in names]
    return SubSchema(
        tables=tables, foreign_keys=fks,
        reasons={s.table: s.reason or s.source for s in selections},
        n_columns=sum(len(t.columns) for t in tables),
    )


# ── the step itself ───────────────────────────────────────────────────────────
def shrink(question: str, cat: Catalog, llm, prune_columns: bool = True,
           max_tables: int | None = None) -> PruneResult:
    s = get_settings()
    max_tables = max_tables or s.prune_max_tables
    fks = fk_map(cat)

    full_text = render_compact(cat.tables, fks)
    full_tokens = est_tokens(full_text)
    candidate_tables = cat.tables
    prefiltered = None
    strategy = "llm"

    # A 7B model with a 16k context cannot read an arbitrarily large schema. Above the
    # budget, keep the best keyword matches — and their FK neighbours, so joins survive.
    if full_tokens > s.prune_max_prompt_tokens:
        top = [k for k, _ in keyword_rank(cat, question, s.prune_prefilter_tables)]
        keep = set(top) | set(fk_closure(cat, top, max_added=10))
        candidate_tables = [t for t in cat.tables if t.qualified in keep]
        prefiltered = len(candidate_tables)
        strategy = "keyword-prefilter+llm"

    schema_text = render_compact(candidate_tables, fks)
    system = SYSTEM.format(max_tables=max_tables)
    user = USER.format(n_tables=len(candidate_tables), schema=schema_text, question=question)

    result = PruneResult(
        sub=SubSchema(), selections=[], strategy=strategy, prefiltered_to=prefiltered,
        tables_before=len(cat.tables), columns_before=sum(len(t.columns) for t in cat.tables),
        prompt_tokens=est_tokens(system + user), full_schema_tokens=full_tokens,
        model=getattr(llm, "spec", type(llm).__name__),
    )

    raw: list[dict] = []
    t0 = time.perf_counter()
    try:
        reply = llm.complete(system, user, json_mode=True)
        result.llm_raw = reply.text[:4000]
        result.answerable, raw = parse_selection(reply.text)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:300]
        log.warning("schema shrink failed (%s); falling back to keyword ranking", exc)
    result.llm_ms = (time.perf_counter() - t0) * 1000

    selections, bad_t, bad_c, corrected = validate(raw, cat, prune_columns, max_tables)
    result.hallucinated_tables, result.hallucinated_columns = bad_t, bad_c
    result.corrected_tables = corrected

    if not selections:
        if result.error is None and not result.answerable:
            # The model says the schema cannot answer this — keep that verdict, it is the
            # abstention signal. The engine decides what to do with it.
            result.sub = SubSchema()
            return result
        # Unusable reply: fall back rather than fail, and say so.
        result.strategy = "keyword-fallback"
        result.answerable = True
        for key, score in keyword_rank(cat, question, min(5, max_tables)):
            t = cat.table(key)
            if t is not None:
                selections.append(Selection(t.qualified, [c.name for c in t.columns],
                                            f"keyword match ({score:.2f})", "fallback"))

    result.selections = selections
    result.sub = build_subschema(cat, selections)
    result.shrunk_schema_tokens = est_tokens(render_compact(result.sub.tables, fks))
    return result


def full(cat: Catalog) -> PruneResult:
    """Baseline: no shrinking — the SQL step gets the whole schema."""
    fks = fk_map(cat)
    tokens = est_tokens(render_compact(cat.tables, fks))
    sub = SubSchema(tables=cat.tables, foreign_keys=cat.foreign_keys,
                    reasons={t.qualified: "full schema (baseline)" for t in cat.tables},
                    n_columns=sum(len(t.columns) for t in cat.tables))
    return PruneResult(
        sub=sub, selections=[Selection(t.qualified, [c.name for c in t.columns],
                                       "full schema (baseline)", "full") for t in cat.tables],
        strategy="full", tables_before=len(cat.tables), columns_before=sub.n_columns,
        full_schema_tokens=tokens, shrunk_schema_tokens=tokens,
    )
