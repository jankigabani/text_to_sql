"""Request and response models for the /api endpoints the UI uses."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.engine.target import ConnectionIn


class ModelChoice(BaseModel):
    """One model for every step, or a different one per step.

    Specs look like ``ollama:qwen2.5-coder:7b``, ``ollama:llama3.1:latest`` or
    ``gemini:gemini-3.5-flash-lite``. Empty means the server default.
    """
    schema_model: str = Field("", alias="schema")
    sql: str = ""
    answer: str = ""

    model_config = {"populate_by_name": True}


class AskOptions(BaseModel):
    # "llm" = the model shrinks the schema (the proposed method); "full" = baseline.
    schema_strategy: Literal["llm", "full"] = "llm"
    prune_columns: bool = True
    max_tables: int = Field(8, ge=1, le=30)
    verify: bool = True                         # False = no gate (baseline)
    disabled_layers: list[str] = Field(default_factory=list)
    max_repairs: int = Field(2, ge=0, le=5)
    row_limit: int = Field(200, ge=1, le=5000)
    refresh_schema: bool = False
    answer: bool = True                          # False = stop after execution (benchmarks)
    log: bool = True                             # False = keep out of "Recent questions"


class AskV2Request(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    connection: ConnectionIn = Field(default_factory=lambda: ConnectionIn(use_default=True))
    model: str = ""                              # one model for every step
    models: ModelChoice | None = None            # or per step
    options: AskOptions = Field(default_factory=AskOptions)

    model_config = {"json_schema_extra": {"examples": [
        {
            "question": "Which five products were ordered in the largest quantities?",
            "connection": {"use_default": True},
            "model": "ollama:qwen2.5-coder:7b",
        },
        {
            "question": "How many employees work for the company?",
            "connection": {"host": "localhost", "port": 5432, "database": "adventureworks",
                           "user": "postgres", "password": "your_password"},
            "model": "ollama:llama3.1:latest",
        },
    ]}}


class CheckOut(BaseModel):
    layer: str
    name: str
    passed: bool
    skipped: bool = False
    warning: bool = False
    message: str = ""
    ms: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class AttemptOut(BaseModel):
    n: int
    sql: str = ""
    reasoning: str = ""
    model: str = ""
    llm_ms: float = 0.0
    checks: list[CheckOut] = Field(default_factory=list)
    passed: bool | None = None
    repair_hint: str = ""


class AskV2Response(BaseModel):
    run_id: str
    question: str
    status: Literal["answered", "abstained", "blocked", "error"]
    answer: str | None = None
    abstain_reason: str | None = None
    sql: str | None = None
    confidence: float = 0.0
    confidence_label: str = "none"
    grounded: bool | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    connection: dict[str, Any] = Field(default_factory=dict)
    schema_info: dict[str, Any] = Field(default_factory=dict)
    shrink: dict[str, Any] = Field(default_factory=dict)
    selected_tables: list[dict[str, Any]] = Field(default_factory=list)
    attempts: list[AttemptOut] = Field(default_factory=list)
    answer_info: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    llm_calls: int = 0
    prompt_tokens_est: int = 0
    total_ms: float = 0.0


class SchemaRequest(BaseModel):
    connection: ConnectionIn = Field(default_factory=lambda: ConnectionIn(use_default=True))
    refresh: bool = False


class FeedbackV2(BaseModel):
    run_id: str
    is_wrong: bool = True
    category: str | None = None
    note: str | None = None
