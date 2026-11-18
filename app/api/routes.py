from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.core.config import get_settings
from app.core.models import AskRequest, AskResponse, FeedbackRequest
from app.db import introspect, runlog
from app.db.pool import owner_pool
from app.llm.providers import get_provider
from app.pipeline.orchestrator import get_pipeline, reset_pipeline
from app.retrieval.question import expand
from app.retrieval.retriever import retrieve

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", tags=["ops"])
def health() -> dict:
    s = get_settings()
    out: dict = {"status": "ok", "checks": {}}

    try:
        with owner_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
        out["checks"]["postgres"] = {"ok": True, "detail": version.split(",")[0]}
    except Exception as exc:
        out["status"] = "degraded"
        out["checks"]["postgres"] = {"ok": False, "detail": str(exc)}

    ok, detail = get_provider().health()
    out["checks"]["llm"] = {"ok": ok, "provider": s.llm_provider, "model": s.llm_model,
                            "detail": detail}
    if not ok:
        out["status"] = "degraded"

    cat = introspect.load_cached()
    out["checks"]["catalog"] = {
        "ok": cat is not None,
        "tables": len(cat.tables) if cat else 0,
        "columns": sum(len(t.columns) for t in cat.tables) if cat else 0,
        "foreign_keys": len(cat.foreign_keys) if cat else 0,
        "detail": "cached" if cat else "not built — run scripts/04_build_schema_index.py",
    }
    out["gate"] = {
        "disabled_layers": sorted(s.disabled_layers),
        "max_repairs": s.gate_max_repairs,
        "confidence_threshold": s.gate_confidence_threshold,
        "plan_cost_ceiling": s.gate_plan_cost_ceiling,
    }
    return out


@router.post("/ask", response_model=AskResponse, tags=["pipeline"])
def ask(req: AskRequest, background: BackgroundTasks) -> AskResponse:
    resp = get_pipeline().ask(req)
    background.add_task(runlog.record, resp)
    return resp


@router.post("/feedback", tags=["pipeline"])
def feedback(fb: FeedbackRequest) -> dict:
    ok = runlog.record_feedback(fb)
    if not ok:
        raise HTTPException(status_code=400, detail="could not record feedback")
    return {"recorded": True, "run_id": fb.run_id}


@router.get("/schema/search", tags=["schema"])
def schema_search(q: str = Query(min_length=2), top: int = 8) -> dict:
    """Stage 2 on its own — useful for the demo and for debugging retrieval."""
    p = get_pipeline()
    sub = retrieve(q, p.catalog, p.index, top_tables=top)
    return {
        "question": q,
        "expanded": expand(q).__dict__,
        "tables": [
            {
                "table": t.qualified,
                "rows": t.estimated_rows,
                "why": sub.reasons.get(t.qualified, ""),
                "columns": [c.name for c in t.columns],
            }
            for t in sub.tables
        ],
        "foreign_keys": [f"{f.left} -> {f.right}" for f in sub.foreign_keys],
        "retrieval_margin": sub.retrieval_margin,
        "columns_shown": sub.n_columns,
        "columns_total": sum(len(t.columns) for t in p.catalog.tables),
    }


@router.get("/schema/stats", tags=["schema"])
def schema_stats() -> dict:
    cat = get_pipeline().catalog
    biggest = sorted(cat.tables, key=lambda t: t.estimated_rows, reverse=True)[:10]
    return {
        "database": cat.database,
        "tables": len(cat.tables),
        "columns": sum(len(t.columns) for t in cat.tables),
        "foreign_keys": len(cat.foreign_keys),
        "estimated_rows_total": sum(t.estimated_rows for t in cat.tables),
        "schema_version": cat.schema_version,
        "largest_tables": [{"table": t.qualified, "rows": t.estimated_rows} for t in biggest],
    }


@router.post("/schema/refresh", tags=["schema"])
def schema_refresh() -> dict:
    cat = introspect.get_catalog(refresh=True)
    reset_pipeline()
    return {"refreshed": True, "tables": len(cat.tables), "schema_version": cat.schema_version}
