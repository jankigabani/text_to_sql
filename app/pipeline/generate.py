"""Stage 3 — prompt building and SQL extraction.

Two decisions here that matter for the write-up:

1. The prompt contains **only the retrieved sub-schema**, plus sample values and the FK
   edges between the chosen tables. Nothing else about the database is revealed.
2. The model is asked for a short **reasoning block before the SQL**, so the join path is
   inspectable by a human and by the error taxonomy in Presentation 2. The reasoning is
   never executed and never shown to the user as the answer.
"""
from __future__ import annotations

import logging
import re

from app.core.ident import q, qualified, qualified_str
from app.core.models import SubSchema
from app.retrieval.question import ExpandedQuestion

log = logging.getLogger(__name__)

SYSTEM = """You are a careful PostgreSQL analyst. You translate a question into exactly one
read-only SQL query for PostgreSQL 16.

Rules you must follow:
- Use ONLY the tables and columns given in the schema. Never invent a name.
- Always schema-qualify tables, e.g. sales.salesorderheader.
- Write every name exactly as shown. Names shown in "double quotes" (capitals, spaces,
  symbols) must keep their double quotes in your SQL, e.g. "CDSCode".
- Do not use SQL keywords (is, as, on, in, to, do, end, ...) as table aliases.
- Hints may be written in SQLite syntax (strftime, julianday, IIF, CAST(x AS REAL)).
  Translate them to PostgreSQL (EXTRACT, to_char, date arithmetic, CASE, ::numeric).
- Join only along the foreign keys listed. If a join you need is not listed, say so instead.
- One single SELECT statement. No INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, GRANT, COPY.
- No semicolon-separated second statement. No comments.
- If the question cannot be answered from this schema, output exactly: NOT_ANSWERABLE

Answer in this format and nothing else:

REASONING: <two or three sentences: which tables, which join keys, which filter>
SQL:
```sql
<your query>
```"""

USER_TEMPLATE = """# Database schema (PostgreSQL)

{schema}

# Foreign keys you may join on

{fks}

# Question

{question}
{hints}"""


def render_schema(sub: SubSchema, max_sample_values: int = 4) -> str:
    blocks = []
    for t in sorted(sub.tables, key=lambda x: x.qualified):
        header = f"TABLE {qualified(t.schema_name, t.name)}"
        if t.comment:
            header += f"  -- {t.comment}"
        if t.estimated_rows:
            header += f"  [~{t.estimated_rows:,} rows]"
        lines = [header]
        for c in t.columns:
            flags = []
            if c.is_primary_key:
                flags.append("PK")
            if not c.nullable:
                flags.append("NOT NULL")
            bits = f"  {q(c.name)} {c.data_type}"
            if flags:
                bits += f" [{', '.join(flags)}]"
            if c.sample_values:
                vals = ", ".join(v for v in c.sample_values[:max_sample_values])
                bits += f"  e.g. {vals}"
            if c.comment:
                bits += f"  -- {c.comment}"
            lines.append(bits)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_fks(sub: SubSchema) -> str:
    if not sub.foreign_keys:
        return "(none between the selected tables)"
    return "\n".join(
        f"{qualified_str(fk.left)}({', '.join(q(c) for c in fk.columns)}) -> "
        f"{qualified_str(fk.right)}({', '.join(q(c) for c in fk.ref_columns)})"
        for fk in sub.foreign_keys
    )


def render_hints(eq: ExpandedQuestion) -> str:
    hints = []
    if eq.literals:
        hints.append(f"Literal values mentioned: {', '.join(eq.literals[:8])}")
    if eq.aggregations:
        hints.append(f"Likely aggregation: {', '.join(eq.aggregations)}")
    if eq.wants_ranking:
        hints.append("The question asks for a ranking, so ORDER BY ... LIMIT is probably needed.")
    if eq.mentions_time:
        hints.append("The question refers to a time period; filter on the relevant date column.")
    return "\n\n# Hints\n\n" + "\n".join(f"- {h}" for h in hints) if hints else ""


def build_prompt(question: str, sub: SubSchema, eq: ExpandedQuestion,
                 repair_hint: str = "") -> tuple[str, str]:
    user = USER_TEMPLATE.format(
        schema=render_schema(sub),
        fks=render_fks(sub),
        question=question,
        hints=render_hints(eq),
    )
    if repair_hint:
        user += (
            "\n\n# Your previous attempt was rejected\n\n"
            f"{repair_hint}\n\n"
            "Fix exactly that problem and output the corrected query in the same format."
        )
    return SYSTEM, user


_FENCE = re.compile(r"```(?:sql|postgresql)?\s*(.+?)```", re.S | re.I)
_SQL_TAG = re.compile(r"\bSQL\s*:\s*(.+)", re.S | re.I)
_REASONING = re.compile(r"\bREASONING\s*:\s*(.+?)(?:\n\s*SQL\s*:|\Z)", re.S | re.I)
_LEADING = re.compile(r"^(with|select)\b", re.I)


def parse_completion(text: str) -> tuple[str, str]:
    """Returns (sql, reasoning). sql == '' means the model declined or produced nothing."""
    reasoning = ""
    m = _REASONING.search(text)
    if m:
        reasoning = " ".join(m.group(1).split())[:1000]

    if "NOT_ANSWERABLE" in text.upper():
        return "", reasoning or "model reported the question is not answerable"

    body = text
    m = _SQL_TAG.search(text)
    if m:
        body = m.group(1)

    m = _FENCE.search(body) or _FENCE.search(text)
    sql = (m.group(1) if m else body).strip()

    for i, line in enumerate(sql.splitlines()):
        if _LEADING.match(line.strip()):
            sql = "\n".join(sql.splitlines()[i:])
            break
    sql = sql.strip().rstrip(";").strip()
    return sql, reasoning
