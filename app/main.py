from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as v1_router
from app.api.v2 import router as api_router
from app.core.config import get_settings
from app.db.pool import close_pools
from app.engine.connections import close_all

log = logging.getLogger(__name__)
FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    configure_logging(s.log_level)
    log.info("starting · default model=%s · db=%s@%s:%s/%s",
             s.default_model or f"{s.llm_provider}:{s.llm_model}",
             s.pguser, s.pghost, s.pgport, s.pgdatabase)
    log.info("UI:  http://127.0.0.1:8000/      API docs: http://127.0.0.1:8000/docs")
    yield
    close_all()
    close_pools()


app = FastAPI(
    title="Text-to-SQL with a Verification Gate",
    description=(
        "COMP 5112 · Group 6. Ask any PostgreSQL database a question in plain English.\n\n"
        "**/api** — the endpoints the web UI uses: connect to any database, read its schema, "
        "let the LLM shrink it to the relevant tables, generate SQL, verify it with five "
        "deterministic checks, run it read-only, and phrase a grounded answer.\n\n"
        "**v1** — the original single-database endpoints used by the evaluation harness."
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

app.include_router(api_router)
app.include_router(v1_router, prefix="/v1", tags=["v1 (evaluation harness)"])

# The UI is served last, at "/", so every API route above takes precedence.
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="ui")
