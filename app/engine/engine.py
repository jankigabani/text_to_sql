"""The v2 engine — the seven stages the UI shows, as a stream of events.

    1 connect    open a read-only session to the database the user supplied
    2 schema     read every table, column, type, primary and foreign key (cached)
    3 shrink     the LLM picks only the relevant tables and columns  ← validated
    4 generate   the LLM writes SQL from the shrunk schema
    5 verify     L1 parse · L3 policy · L2 bind · L4 plan · L5 sniff  ← the contribution
                 (a failure is fed back to step 4, at most `max_repairs` times)
    6 execute    run it read-only, with a timeout and a row limit
    7 answer     the LLM phrases the result; every number is checked against the rows

`run()` is a generator of events, so the API can stream each stage to the browser as it
happens. `run_sync()` drains it and returns only the final result, for the plain JSON
endpoint and for the evaluation harness. Both produce identical results.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from fastapi.encoders import jsonable_encoder

from app.core.config import get_settings
from app.core.models import GateReport, Layer
from app.engine import answer as answer_step
from app.engine import prune
from app.engine.catalog import get_catalog
from app.engine.connections import ConnectionFailed, pool_for, test_connection
from app.engine.schemas import AskV2Request, AskV2Response, AttemptOut, CheckOut
from app.engine.target import DBTarget
from app.llm.providers import default_spec, provider_for_spec
from app.pipeline import execute, generate
from app.pipeline.gate import label, run_gate, score_confidence
from app.retrieval.question import expand

log = logging.getLogger(__name__)

LAYER_NAMES = {
    "l1": "Valid SQL?",
    "l2": "Tables & columns exist?",
    "l3": "Read-only & safe?",
    "l4": "Sensible query plan?",
    "l5": "Result looks right?",
}
STAGES = ["connect", "schema", "shrink", "generate", "verify", "execute", "answer"]


def _ev(stage: str, status: str, **kw: Any) -> dict:
    return {"type": "stage", "stage": stage, "status": status, **kw}


def _checks_out(report: GateReport) -> list[CheckOut]:
    out = []
    for c in report.checks:
        details = {k: v for k, v in c.details.items() if k != "rows"}
        out.append(CheckOut(
            layer=c.layer.value, name=LAYER_NAMES.get(c.layer.value, c.layer.value),
            passed=c.passed, skipped=c.skipped, warning=c.warning, message=c.message,
            ms=round(c.duration_ms, 2), details=jsonable_encoder(details),
        ))
    return out


def resolve_models(req: AskV2Request) -> dict[str, str]:
    base = req.model or default_spec()
    m = req.models
    return {
        "schema": (m.schema_model if m and m.schema_model else base),
        "sql": (m.sql if m and m.sql else base),
        "answer": (m.answer if m and m.answer else base),
    }


def run(req: AskV2Request, llm_override=None) -> Iterator[dict]:
    """Yields stage events, then one {"type": "result"} event."""
    s = get_settings()
    t_start = time.perf_counter()
    o = req.options
    run_id = uuid.uuid4().hex[:10]
    models = resolve_models(req)
    res = AskV2Response(run_id=run_id, question=req.question, status="error", models=models)
    llms: dict[str, Any] = {}

    def llm(step: str):
        if llm_override is not None:
            return llm_override
        spec = models[step]
        if spec not in llms:
            llms[spec] = provider_for_spec(spec)
        return llms[spec]

    def finish(status: str, reason: str | None = None) -> dict:
        res.status = status  # type: ignore[assignment]
        if reason:
            res.abstain_reason = reason
        res.total_ms = round((time.perf_counter() - t_start) * 1000, 1)
        if o.log:
            _log_run(res)
        return {"type": "result", "data": jsonable_encoder(res)}

    # ── 1 connect ────────────────────────────────────────────────────────────
    yield _ev("connect", "running")
    t0 = time.perf_counter()
    try:
        target: DBTarget = req.connection.to_target()
        info = test_connection(target)
    except (ConnectionFailed, ValueError) as exc:
        res.timings["connect"] = _ms(t0)
        yield _ev("connect", "error", ms=res.timings["connect"], error=str(exc))
        yield finish("error", f"Could not connect to the database: {exc}")
        return
    res.timings["connect"] = _ms(t0)
    res.connection = {**target.public(), "server": info["server"],
                      "server_version": info["server_version"]}
    yield _ev("connect", "done", ms=res.timings["connect"], data={
        **target.public(), "server": info["server"], "latency_ms": info["latency_ms"],
        "databases": info["databases"]})
    pool = pool_for(target)

    # ── 2 schema ─────────────────────────────────────────────────────────────
    yield _ev("schema", "running")
    t0 = time.perf_counter()
    try:
        cat, cinfo = get_catalog(target, refresh=o.refresh_schema)
    except Exception as exc:
        res.timings["schema"] = _ms(t0)
        yield _ev("schema", "error", ms=res.timings["schema"], error=str(exc))
        yield finish("error", f"Could not read the schema: {exc}")
        return
    res.timings["schema"] = _ms(t0)
    n_cols = sum(len(t.columns) for t in cat.tables)
    res.schema_info = {"database": cat.database, "tables": len(cat.tables), "columns": n_cols,
                       "foreign_keys": len(cat.foreign_keys), "cached": cinfo["cached"],
                       "schema_version": cat.schema_version}
    yield _ev("schema", "done", ms=res.timings["schema"], data=res.schema_info)
    if not cat.tables:
        yield finish("abstained", "This database has no tables I can read.")
        return

    # ── 3 shrink ─────────────────────────────────────────────────────────────
    yield _ev("shrink", "running", model=models["schema"])
    t0 = time.perf_counter()
    if o.schema_strategy == "full":
        pr = prune.full(cat)
    else:
        pr = prune.shrink(req.question, cat, llm("schema"), prune_columns=o.prune_columns,
                          max_tables=o.max_tables)
        res.llm_calls += 1
        res.prompt_tokens_est += pr.prompt_tokens
    res.timings["shrink"] = _ms(t0)
    res.shrink = pr.summary()
    res.selected_tables = [
        {"table": sel.table, "columns": sel.columns, "reason": sel.reason, "source": sel.source}
        for sel in pr.selections
    ]
    yield _ev("shrink", "done", ms=res.timings["shrink"],
              data={**res.shrink, "selected": res.selected_tables})
    if not pr.answerable and not pr.sub.tables:
        yield finish("abstained", "None of the tables in this database hold what the question "
                                  "asks about, so I am not going to guess.")
        return

    # ── 4–5 generate + verify, with the bounded repair loop ─────────────────
    eq = expand(req.question)
    repair_hint = ""
    sql = ""
    gate: GateReport | None = None
    max_attempts = (o.max_repairs + 1) if o.verify else 1

    for n in range(1, max_attempts + 1):
        yield _ev("generate", "running", attempt=n, model=models["sql"])
        t0 = time.perf_counter()
        system, user = generate.build_prompt(req.question, pr.sub, eq, repair_hint)
        res.prompt_tokens_est += (len(system) + len(user)) // 4
        try:
            reply = llm("sql").complete(system, user)
        except Exception as exc:
            res.timings[f"generate_{n}"] = _ms(t0)
            yield _ev("generate", "error", attempt=n, ms=res.timings[f"generate_{n}"],
                      error=str(exc))
            yield finish("error", f"The language model failed: {exc}")
            return
        res.llm_calls += 1
        sql, reasoning = generate.parse_completion(reply.text)
        res.timings[f"generate_{n}"] = _ms(t0)
        attempt = AttemptOut(n=n, sql=sql, reasoning=reasoning, model=models["sql"],
                             llm_ms=round(reply.latency_ms, 1), repair_hint=repair_hint)
        res.attempts.append(attempt)
        yield _ev("generate", "done", attempt=n, ms=res.timings[f"generate_{n}"],
                  data={"sql": sql, "reasoning": reasoning, "repair_hint": repair_hint})

        if not sql:
            yield finish("abstained", "The model reported that the question cannot be answered "
                                      "from the selected tables.")
            return

        if not o.verify:
            break

        yield _ev("verify", "running", attempt=n)
        t0 = time.perf_counter()
        gate = run_gate(sql, cat, req.question, attempt=n,
                        disabled={x.lower() for x in o.disabled_layers},
                        pool=pool, row_limit=o.row_limit)
        res.timings[f"verify_{n}"] = _ms(t0)
        attempt.checks = _checks_out(gate)
        attempt.passed = gate.passed
        failed = gate.failed_check()
        yield _ev("verify", "done" if gate.passed else "failed", attempt=n,
                  ms=res.timings[f"verify_{n}"],
                  data={"passed": gate.passed,
                        "checks": [c.model_dump() for c in attempt.checks],
                        "warnings": gate.warnings})

        if gate.passed:
            break
        # Harmful requests are refused outright, never "repaired".
        if failed and (failed.layer is Layer.L3_POLICY or failed.details.get("blocked")):
            res.sql = sql
            msg = failed.message.removeprefix("blocked: ").rstrip(".")
            yield finish("blocked", f"Blocked by the safety check: {msg}. "
                                    "Only read-only questions are allowed, so this query was not run.")
            return
        repair_hint = failed.repair_hint if failed else ""
        if n == max_attempts:
            res.sql = sql
            yield finish("abstained", "I could not produce a query I trust after "
                                      f"{n} attempt{'s' if n > 1 else ''}. Last problem: "
                                      f"{failed.message if failed else 'unknown'}")
            return

    res.sql = sql

    # ── 6 execute ────────────────────────────────────────────────────────────
    yield _ev("execute", "running")
    t0 = time.perf_counter()
    ex = execute.run_query(sql, row_limit=o.row_limit, pool=pool)
    res.timings["execute"] = _ms(t0)
    if ex.error:
        yield _ev("execute", "error", ms=res.timings["execute"], error=ex.error)
        yield finish("abstained", f"The query failed when it ran: {ex.error}")
        return
    res.rows = ex.rows[:200]
    res.columns = list(ex.rows[0].keys()) if ex.rows else []
    res.row_count, res.truncated = ex.row_count, ex.truncated
    yield _ev("execute", "done", ms=res.timings["execute"],
              data={"row_count": ex.row_count, "truncated": ex.truncated,
                    "columns": res.columns, "rows": jsonable_encoder(ex.rows[:50]),
                    "db_ms": round(ex.runtime_ms, 1)})

    # Confidence from the gate's own evidence (the RQ3 signal).
    if gate is not None:
        margin = 1.0 if pr.strategy == "llm" and not pr.hallucinated_tables else 0.5
        res.confidence = score_confidence(gate, margin, len(res.attempts))
    else:
        res.confidence = 0.0  # unverified: no evidence either way
    res.confidence_label = label(res.confidence)

    if gate is not None and res.confidence < s.gate_confidence_threshold:
        yield finish("abstained", "I am not confident in this result: "
                     + ("; ".join(gate.warnings) if gate.warnings else "the checks were not clean")
                     + ". The query and its rows are shown so you can judge for yourself.")
        return

    # ── 7 answer ─────────────────────────────────────────────────────────────
    if not o.answer:  # benchmarks score the SQL only; skip the extra model call
        yield finish("answered")
        return
    yield _ev("answer", "running", model=models["answer"])
    t0 = time.perf_counter()
    ans = answer_step.answer(req.question, sql, ex.rows, ex.truncated, llm("answer"))
    res.timings["answer"] = _ms(t0)
    if ans.get("llm_used"):
        res.llm_calls += 1
    res.answer_info = ans
    res.grounded = ans.get("grounded")
    res.answer = ans.get("answer") or (
        "Here is the result. I did not phrase it as a sentence because "
        + ("the wording I produced contained numbers that are not in the result."
           if ans.get("ungrounded_numbers") else "the language model was unavailable.")
    )
    yield _ev("answer", "done", ms=res.timings["answer"], data=ans)
    yield finish("answered")


def run_sync(req: AskV2Request, llm_override=None,
             on_event=None) -> AskV2Response:
    final: dict | None = None
    for ev in run(req, llm_override=llm_override):
        if on_event:
            on_event(ev)
        if ev["type"] == "result":
            final = ev["data"]
    assert final is not None
    return AskV2Response.model_validate(final)


# ── run log (a local file: nothing is written into the user's database) ───────
def _runs_file() -> Path:
    return Path(get_settings().index_dir).parent / "runs.jsonl"


def _log_run(res: AskV2Response) -> None:
    try:
        p = _runs_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        slim = res.model_dump(mode="json", exclude={"rows"})
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **slim},
                               default=str) + "\n")
    except Exception as exc:
        log.warning("run log write failed: %s", exc)


def recent_runs(limit: int = 30) -> list[dict]:
    p = _runs_file()
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()[-limit:]
    out = []
    for line in reversed(lines):
        try:
            r = json.loads(line)
            out.append({k: r.get(k) for k in ("run_id", "logged_at", "question", "status",
                                                "confidence_label", "total_ms", "models",
                                                "connection")})
        except Exception:
            continue
    return out


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
