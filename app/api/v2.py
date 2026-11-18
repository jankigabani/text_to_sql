"""The /api endpoints the web UI uses.

Every question carries its own connection details, so the same server can answer
questions about any PostgreSQL database — switch databases in the UI and ask again.

    GET  /api/health               server, database and model status
    GET  /api/models               models that will actually answer (Ollama + Gemini)
    GET  /api/connections/default  the .env database, without its password
    POST /api/connections/test     check connection details before using them
    POST /api/schema               every table / column / type / key of a database
    POST /api/ask                  run the 7-stage pipeline, return the full trace
    POST /api/ask/stream           same, streamed stage by stage (Server-Sent Events)
    GET  /api/examples             example questions for the UI
    GET  /api/runs                 recent questions (local run log)
    POST /api/feedback             "this answer is wrong"
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.engine import engine
from app.engine.catalog import describe, get_catalog
from app.engine.connections import ConnectionFailed, test_connection
from app.engine.schemas import AskV2Request, AskV2Response, FeedbackV2, SchemaRequest
from app.engine.target import ConnectionIn, DBTarget
from app.llm.providers import available_models, default_spec
from eval.datasets.suites import ADVENTUREWORKS

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["api (used by the UI)"])


@router.get("/health")
def health() -> dict:
    s = get_settings()
    out: dict = {"status": "ok", "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        info = test_connection(DBTarget.from_settings())
        out["database"] = {"ok": True, "name": info["database"], "server": info["server"],
                           "tables": info["tables"]}
    except Exception as exc:
        out["status"] = "degraded"
        out["database"] = {"ok": False, "detail": str(exc)}
    m = available_models()
    out["models"] = m["providers"]
    out["default_model"] = m["default"]
    if not any(p.get("ok") for k, p in m["providers"].items() if k != "mock"):
        out["status"] = "degraded"
    out["gate"] = {"max_repairs": s.gate_max_repairs,
                   "confidence_threshold": s.gate_confidence_threshold,
                   "plan_cost_ceiling": s.gate_plan_cost_ceiling}
    return out


@router.get("/models")
def models(refresh: bool = False) -> dict:
    return available_models(refresh=refresh)


@router.get("/connections/default")
def default_connection() -> dict:
    s = get_settings()
    return {"host": s.pghost, "port": s.pgport, "database": s.pgdatabase, "user": s.pguser,
            "has_password": bool(s.pgpassword)}


@router.post("/connections/test")
def connections_test(conn: ConnectionIn) -> dict:
    try:
        return test_connection(conn.to_target())
    except (ConnectionFailed, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/schema")
def schema(req: SchemaRequest) -> dict:
    try:
        target = req.connection.to_target()
        test_connection(target)
        cat, info = get_catalog(target, refresh=req.refresh)
    except (ConnectionFailed, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {**describe(cat), "cache": info}


@router.post("/ask", response_model=AskV2Response)
def ask(req: AskV2Request) -> AskV2Response:
    return engine.run_sync(req)


@router.post("/ask/stream")
def ask_stream(req: AskV2Request) -> StreamingResponse:
    """Server-Sent Events: one `stage` event as each stage starts and ends, then `result`."""

    def events():
        try:
            for ev in engine.run(req):
                yield f"event: {ev['type']}\ndata: {json.dumps(jsonable_encoder(ev))}\n\n"
        except Exception as exc:  # never leave the browser waiting on a dead stream
            log.exception("stream failed")
            payload = {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/examples")
def examples() -> dict:
    groups: dict[str, list[dict]] = {"answerable": [], "unanswerable": [], "unsafe": []}
    for it in ADVENTUREWORKS:
        groups[it.kind].append({"id": it.id, "question": it.question,
                                "difficulty": it.difficulty})
    return {"database": "adventureworks", "groups": groups, "default_model": default_spec()}


@router.get("/runs")
def runs(limit: int = 30) -> list[dict]:
    return engine.recent_runs(limit)


@router.post("/feedback")
def feedback(fb: FeedbackV2) -> dict:
    p = Path(get_settings().index_dir).parent / "feedback.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            **fb.model_dump()}) + "\n")
    return {"recorded": True, "run_id": fb.run_id}
