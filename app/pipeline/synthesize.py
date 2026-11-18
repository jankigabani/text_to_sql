"""Stage 6 — rows back into a sentence, with a grounding check.

The model sees the question and the result rows and nothing else — not the schema, not the
SQL, and certainly not the database. Then every number in its sentence is checked against
the result set. An ungrounded number fails the sentence and the user gets the table
instead. Execution accuracy never measures this step; this is where we do.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

log = logging.getLogger(__name__)

SYSTEM = """You turn a database result into one short, plain answer.

Rules:
- Use ONLY the numbers and values present in the result. Never add a figure of your own.
- Do not round, do not estimate, do not describe a trend that is not in the result.
- If the result does not answer the question, say so in one sentence.
- One or two sentences. No preamble, no bullet points, no SQL."""

USER = """Question: {question}

Result ({n} row{s}){trunc}:
{table}

Answer in one or two sentences."""

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class Synthesis:
    answer: str
    grounded: bool
    ungrounded_numbers: list[str]
    fell_back_to_table: bool = False


def render_rows(rows: list[dict[str, Any]], max_rows: int = 30) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    head = " | ".join(cols)
    body = "\n".join(
        " | ".join("NULL" if r.get(c) is None else str(r.get(c)) for c in cols)
        for r in rows[:max_rows]
    )
    more = f"\n... and {len(rows) - max_rows} more rows" if len(rows) > max_rows else ""
    return f"{head}\n{'-' * len(head)}\n{body}{more}"


def build_prompt(question: str, rows: list[dict[str, Any]], truncated: bool) -> tuple[str, str]:
    return SYSTEM, USER.format(
        question=question,
        n=len(rows),
        s="" if len(rows) == 1 else "s",
        trunc=", truncated" if truncated else "",
        table=render_rows(rows),
    )


def _numbers_in_rows(rows: list[dict[str, Any]]) -> set[Decimal]:
    out: set[Decimal] = set()
    for r in rows:
        for v in r.values():
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float, Decimal)):
                out.add(_q(Decimal(str(v))))
            else:
                for m in _NUMBER.finditer(str(v)):
                    try:
                        out.add(_q(Decimal(m.group(0).replace(",", ""))))
                    except InvalidOperation:
                        continue
    # The row count itself is a legitimate thing to state.
    out.add(Decimal(len(rows)))
    return out


def _q(d: Decimal) -> Decimal:
    try:
        return d.normalize()
    except InvalidOperation:
        return d


def check_grounding(answer: str, rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Every numeral in the answer must appear in the result (or be the row count)."""
    allowed = _numbers_in_rows(rows)
    ungrounded: list[str] = []
    for m in _NUMBER.finditer(answer):
        token = m.group(0)
        try:
            val = _q(Decimal(token.replace(",", "")))
        except InvalidOperation:
            continue
        if val in allowed:
            continue
        # Accept a rounded restatement of a real number (14.2 for 14.23...).
        if any(abs(val - a) <= abs(a) * Decimal("0.005") for a in allowed if a != 0):
            continue
        # Accept years that appear as a substring of a date value in the result.
        if re.fullmatch(r"(19|20)\d{2}", token) and any(token in str(v)
                                                        for r in rows for v in r.values()):
            continue
        ungrounded.append(token)
    return (not ungrounded), ungrounded


def fallback_answer(question: str, rows: list[dict[str, Any]], reason: str) -> str:
    if not rows:
        return "The query ran successfully but returned no rows, so I cannot answer this."
    return (
        f"I ran the query but could not phrase a trustworthy sentence ({reason}), "
        f"so here is the raw result:\n\n{render_rows(rows, max_rows=10)}"
    )


def synthesize(question: str, rows: list[dict[str, Any]], truncated: bool, llm) -> Synthesis:
    if not rows:
        return Synthesis(
            answer="The query returned no rows, so there is no answer to report.",
            grounded=True, ungrounded_numbers=[],
        )
    system, user = build_prompt(question, rows, truncated)
    try:
        result = llm.complete(system, user)
    except Exception as exc:
        log.warning("synthesis LLM failed: %s", exc)
        return Synthesis(fallback_answer(question, rows, "the model was unreachable"),
                         grounded=True, ungrounded_numbers=[], fell_back_to_table=True)

    answer = " ".join(result.text.split()).strip()
    grounded, bad = check_grounding(answer, rows)
    if not grounded:
        log.info("synthesis rejected: ungrounded numbers %s", bad)
        return Synthesis(
            answer=fallback_answer(question, rows, f"it contained numbers not in the result: {', '.join(bad)}"),
            grounded=False, ungrounded_numbers=bad, fell_back_to_table=True,
        )
    return Synthesis(answer=answer, grounded=True, ungrounded_numbers=[])
