"""Every question, and what the gate decided, goes in a table.

Two reasons: the "report as wrong" button needs somewhere to put feedback, and the error
taxonomy in Presentation 2 is built by querying this log rather than by re-running
everything.
"""
from __future__ import annotations

import json
import logging

from psycopg import sql as pgsql

from app.core.config import get_settings
from app.core.models import AskResponse, FeedbackRequest
from app.db.pool import owner_pool

log = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.runs (
    run_id         text PRIMARY KEY,
    asked_at       timestamptz NOT NULL DEFAULT now(),
    question       text        NOT NULL,
    arm            text        NOT NULL,
    status         text        NOT NULL,
    answer         text,
    sql_text       text,
    confidence     numeric(5,4),
    abstain_reason text,
    row_count      integer,
    llm_calls      integer,
    prompt_tokens  integer,
    total_ms       numeric(10,2),
    tables_used    text[],
    gate           jsonb,
    timings        jsonb
);

CREATE TABLE IF NOT EXISTS {schema}.feedback (
    id        bigserial PRIMARY KEY,
    run_id    text NOT NULL REFERENCES {schema}.runs(run_id) ON DELETE CASCADE,
    given_at  timestamptz NOT NULL DEFAULT now(),
    is_wrong  boolean NOT NULL,
    category  text,
    note      text
);

CREATE INDEX IF NOT EXISTS runs_status_idx ON {schema}.runs(status);
CREATE INDEX IF NOT EXISTS runs_arm_idx    ON {schema}.runs(arm);
"""


def ensure_schema() -> None:
    s = get_settings()
    stmt = DDL.format(schema=f'"{s.run_log_schema}"')
    with owner_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(stmt)
        conn.commit()
    log.info("run log ready in schema %s", s.run_log_schema)


def record(resp: AskResponse) -> None:
    s = get_settings()
    if not s.run_log_enabled or not resp.run_id:
        return
    gate_json = [
        a.gate.model_dump(mode="json") for a in resp.attempts if a.gate is not None
    ]
    q = pgsql.SQL(
        """INSERT INTO {}.runs (run_id, question, arm, status, answer, sql_text, confidence,
               abstain_reason, row_count, llm_calls, prompt_tokens, total_ms, tables_used,
               gate, timings)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (run_id) DO NOTHING"""
    ).format(pgsql.Identifier(s.run_log_schema))
    try:
        with owner_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(q, (
                    resp.run_id, resp.question, resp.arm, resp.status, resp.answer, resp.sql,
                    resp.confidence, resp.abstain_reason, resp.row_count, resp.llm_calls,
                    resp.prompt_tokens_est, sum(t.ms for t in resp.timings), resp.tables_used,
                    json.dumps(gate_json), json.dumps([t.model_dump() for t in resp.timings]),
                ))
            conn.commit()
    except Exception as exc:  # logging must never break answering
        log.warning("run log insert failed: %s", exc)


def record_feedback(fb: FeedbackRequest) -> bool:
    s = get_settings()
    q = pgsql.SQL(
        "INSERT INTO {}.feedback (run_id, is_wrong, category, note) VALUES (%s,%s,%s,%s)"
    ).format(pgsql.Identifier(s.run_log_schema))
    try:
        with owner_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(q, (fb.run_id, fb.is_wrong, fb.category, fb.note))
            conn.commit()
        return True
    except Exception as exc:
        log.warning("feedback insert failed: %s", exc)
        return False
