"""Stage 7 — the result becomes a sentence, and the sentence is checked.

The model receives three things: the question, the SQL that was run, and the result rows.
It is told to copy values exactly. Then every number in its sentence is checked against
the result. If any number does not appear there, the sentence is discarded and the user
sees the result table instead — a modified answer is never shown.
"""
from __future__ import annotations

import datetime as dt
import logging
from decimal import Decimal
from typing import Any

from app.pipeline.synthesize import check_grounding, render_rows

log = logging.getLogger(__name__)

SYSTEM = """You turn the result of a database query into a short, plain-English answer.

Rules:
- Use ONLY the values in the result. Copy numbers, names and dates exactly as written.
- Do not round, convert, estimate, add up, or add any figure of your own.
- Do not describe trends, reasons or comparisons that are not in the result.
- If the result is empty or does not answer the question, say so in one sentence.
- One to three sentences. No preamble, no bullet points, no SQL, no markdown."""

USER = """Question: {question}

SQL that was run:
{sql}

Result ({n} row{s}{trunc}):
{table}

Answer the question using only this result."""


def display_value(v: Any) -> Any:
    """What the model sees. Long decimals are shown to 2 places so the model has no reason
    to round them itself; the grounding check accepts both the raw and displayed value."""
    if isinstance(v, Decimal):
        if v == v.to_integral_value():
            return int(v)
        return round(float(v), 2) if abs(v) >= 1 else round(float(v), 4)
    if isinstance(v, float):
        return round(v, 2) if abs(v) >= 1 else round(v, 4)
    if isinstance(v, dt.datetime):
        return v.strftime("%Y-%m-%d %H:%M") if (v.hour or v.minute) else v.strftime("%Y-%m-%d")
    if isinstance(v, dt.date):
        return v.isoformat()
    return v


def display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: display_value(v) for k, v in r.items()} for r in rows]


def answer(question: str, sql: str, rows: list[dict[str, Any]], truncated: bool, llm) -> dict:
    if not rows:
        return {"answer": "The query ran successfully but returned no rows, so there is "
                          "nothing to report for this question.",
                "grounded": True, "ungrounded_numbers": [], "fell_back_to_table": False,
                "llm_used": False}

    shown = display_rows(rows)
    user = USER.format(question=question, sql=sql, n=len(rows), s="" if len(rows) == 1 else "s",
                       trunc=", showing the first rows" if truncated else "",
                       table=render_rows(shown, max_rows=40))
    try:
        reply = llm.complete(SYSTEM, user)
        text = " ".join(reply.text.replace("**", "").split()).strip()
    except Exception as exc:
        log.warning("answer step failed: %s", exc)
        return {"answer": None, "grounded": True, "ungrounded_numbers": [],
                "fell_back_to_table": True, "llm_used": True,
                "error": f"the model could not be reached: {exc}"}

    # Accept a number if it appears in the raw result *or* in what the model was shown.
    # (Checked separately, not on rows + shown, so the row count is not doubled.)
    grounded, bad = check_grounding(text, rows)
    if not grounded:
        _, bad_shown = check_grounding(text, shown)
        bad = [b for b in bad if b in bad_shown]
        grounded = not bad
    if not grounded:
        log.info("answer rejected, numbers not in result: %s", bad)
        return {"answer": None, "rejected_answer": text, "grounded": False,
                "ungrounded_numbers": bad, "fell_back_to_table": True, "llm_used": True}
    return {"answer": text, "grounded": True, "ungrounded_numbers": [],
            "fell_back_to_table": False, "llm_used": True}
