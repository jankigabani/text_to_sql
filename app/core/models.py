"""Request/response and internal trace models. These double as the data the paper reports."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── catalog ───────────────────────────────────────────────────────────────────
class Column(BaseModel):
    schema_name: str
    table_name: str
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    comment: str | None = None
    sample_values: list[str] = Field(default_factory=list)
    n_distinct: float | None = None

    @property
    def qualified(self) -> str:
        return f"{self.schema_name}.{self.table_name}.{self.name}"


class ForeignKey(BaseModel):
    schema_name: str
    table_name: str
    columns: list[str]
    ref_schema: str
    ref_table: str
    ref_columns: list[str]

    @property
    def left(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    @property
    def right(self) -> str:
        return f"{self.ref_schema}.{self.ref_table}"


class Table(BaseModel):
    schema_name: str
    name: str
    comment: str | None = None
    estimated_rows: int = 0
    columns: list[Column] = Field(default_factory=list)
    primary_key: list[str] = Field(default_factory=list)

    @property
    def qualified(self) -> str:
        return f"{self.schema_name}.{self.name}"


class Catalog(BaseModel):
    database: str
    tables: list[Table] = Field(default_factory=list)
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    introspected_at: datetime = Field(default_factory=datetime.utcnow)
    schema_version: str = ""

    def table(self, qualified: str) -> Table | None:
        want = qualified.lower()
        for t in self.tables:
            if t.qualified.lower() == want or t.name.lower() == want:
                return t
        return None


# ── retrieval ─────────────────────────────────────────────────────────────────
class SubSchema(BaseModel):
    """What stage 3 is allowed to see."""
    tables: list[Table] = Field(default_factory=list)
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    reasons: dict[str, str] = Field(default_factory=dict)
    table_scores: dict[str, float] = Field(default_factory=dict)
    retrieval_margin: float = 0.0
    n_columns: int = 0


# ── gate ──────────────────────────────────────────────────────────────────────
class Layer(str, Enum):
    L1_PARSE = "l1"
    L2_BIND = "l2"
    L3_POLICY = "l3"
    L4_PLAN = "l4"
    L5_SNIFF = "l5"


class CheckResult(BaseModel):
    layer: Layer
    passed: bool
    skipped: bool = False
    duration_ms: float = 0.0
    # Human-readable reason (shown to the user on abstain)
    message: str = ""
    # Machine-readable, fed back to the LLM as a repair instruction
    repair_hint: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    # A soft finding (L5): the query ran but the result looks suspicious.
    warning: bool = False


class GateReport(BaseModel):
    passed: bool
    attempt: int = 1
    checks: list[CheckResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    total_ms: float = 0.0

    def failed_check(self) -> CheckResult | None:
        return next((c for c in self.checks if not c.passed and not c.skipped), None)


# ── pipeline ──────────────────────────────────────────────────────────────────
class StageTiming(BaseModel):
    stage: str
    ms: float


class Attempt(BaseModel):
    n: int
    sql: str
    reasoning: str = ""
    gate: GateReport | None = None
    repair_hint: str = ""


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    # Per-request overrides, used by the evaluation harness
    arm: Literal["ours", "baseline_a", "baseline_b", "baseline_c"] = "ours"
    disabled_layers: list[str] = Field(default_factory=list)
    explain: bool = True


class AskResponse(BaseModel):
    question: str
    status: Literal["answered", "abstained", "blocked", "error"]
    answer: str | None = None
    sql: str | None = None
    confidence: float = 0.0
    confidence_label: Literal["high", "medium", "low", "none"] = "none"
    abstain_reason: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    tables_used: list[str] = Field(default_factory=list)
    attempts: list[Attempt] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens_est: int = 0
    run_id: str | None = None
    arm: str = "ours"


class FeedbackRequest(BaseModel):
    run_id: str
    is_wrong: bool = True
    note: str | None = None
    category: str | None = None
