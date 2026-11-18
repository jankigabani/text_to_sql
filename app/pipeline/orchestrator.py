"""The whole pipeline, in one place — and the four experimental arms.

| arm         | schema given to the LLM | verification            | abstention |
|-------------|-------------------------|-------------------------|------------|
| baseline_a  | the entire database     | none                    | no         |
| baseline_b  | retrieved sub-schema    | none                    | no         |
| baseline_c  | retrieved sub-schema    | the LLM re-checks itself| no         |
| ours        | retrieved sub-schema    | L1–L5 deterministic gate| yes        |

Baseline C is the DIN-SQL-style configuration: one extra model call that asks the model to
review its own SQL. It is the arm the contribution has to beat.
"""
from __future__ import annotations

import logging
import time
import uuid

from app.core.config import get_settings
from app.core.models import (
    Attempt, AskRequest, AskResponse, Catalog, GateReport, StageTiming,
)
from app.db import introspect
from app.llm.providers import get_provider
from app.pipeline import execute, generate, synthesize
from app.pipeline.gate import label, run_gate, score_confidence
from app.retrieval.index import get_index
from app.retrieval.question import expand
from app.retrieval.retriever import full_schema, retrieve

log = logging.getLogger(__name__)

SELF_CHECK_SYSTEM = """You review a PostgreSQL query against a schema and a question.
If it is correct, reply exactly: OK
Otherwise reply with the corrected query in a ```sql block."""


class Pipeline:
    """Holds the catalog and index so they are introspected once, not per request."""

    def __init__(self, catalog: Catalog | None = None, llm=None) -> None:
        self.catalog = catalog or introspect.get_catalog()
        self.index = get_index(self.catalog)
        self.llm = llm or get_provider()

    # ── main entry point ─────────────────────────────────────────────────────
    def ask(self, req: AskRequest) -> AskResponse:
        s = get_settings()
        run_id = str(uuid.uuid4())[:8]
        timings: list[StageTiming] = []
        attempts: list[Attempt] = []
        llm_calls = 0
        prompt_chars = 0

        def tick(stage: str, t0: float) -> None:
            timings.append(StageTiming(stage=stage, ms=round((time.perf_counter() - t0) * 1000, 1)))

        use_gate = req.arm == "ours"
        use_retrieval = req.arm != "baseline_a"
        use_self_check = req.arm == "baseline_c"
        disabled = {x.lower() for x in req.disabled_layers} or s.disabled_layers

        # ── stages 1–2 ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        eq = expand(req.question)
        sub = retrieve(req.question, self.catalog, self.index) if use_retrieval \
            else full_schema(self.catalog)
        tick("retrieval", t0)

        # ── stages 3–4, with the bounded repair loop ─────────────────────────
        repair_hint = ""
        gate: GateReport | None = None
        sql = ""
        max_attempts = (s.gate_max_repairs + 1) if use_gate else 1

        for n in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            system, user = generate.build_prompt(req.question, sub, eq, repair_hint)
            prompt_chars += len(system) + len(user)
            try:
                completion = self.llm.complete(system, user)
            except Exception as exc:
                tick(f"generate_{n}", t0)
                return self._error(req, run_id, timings, f"the language model failed: {exc}")
            llm_calls += 1
            sql, reasoning = generate.parse_completion(completion.text)
            tick(f"generate_{n}", t0)

            if not sql:
                attempts.append(Attempt(n=n, sql="", reasoning=reasoning))
                return self._abstain(
                    req, run_id, timings, attempts, llm_calls, prompt_chars,
                    "The question cannot be answered from the tables I can see.",
                )

            if use_self_check and n == 1:
                t0 = time.perf_counter()
                sql, calls = self._self_check(req.question, sub, sql)
                llm_calls += calls
                tick("self_check", t0)

            if not use_gate:
                attempts.append(Attempt(n=n, sql=sql, reasoning=reasoning))
                break

            t0 = time.perf_counter()
            gate = run_gate(sql, self.catalog, req.question, attempt=n, disabled=disabled)
            tick(f"gate_{n}", t0)
            attempts.append(Attempt(n=n, sql=sql, reasoning=reasoning, gate=gate,
                                    repair_hint=repair_hint))

            if gate.passed:
                break

            failed = gate.failed_check()
            # A policy violation is never repaired: a request that would write to the
            # database is refused, not negotiated with. L1 also flags write statements
            # this way, since a DELETE is a policy problem even though it parses.
            if failed and (failed.layer.value == "l3" or failed.details.get("blocked")):
                return self._blocked(req, run_id, timings, attempts, llm_calls, prompt_chars,
                                     failed.message)
            repair_hint = failed.repair_hint if failed else ""
            if n == max_attempts:
                return self._abstain(
                    req, run_id, timings, attempts, llm_calls, prompt_chars,
                    f"I could not produce a query I trust. Last problem: "
                    f"{failed.message if failed else 'unknown'}",
                    sql=sql,
                )

        # ── stage 5 ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        ex = execute.run_query(sql)
        tick("execute", t0)
        if ex.error:
            return self._abstain(req, run_id, timings, attempts, llm_calls, prompt_chars,
                                 f"The query failed when executed: {ex.error}", sql=sql)

        # ── confidence / abstention ──────────────────────────────────────────
        confidence = 1.0
        if use_gate and gate is not None:
            confidence = score_confidence(gate, sub.retrieval_margin, len(attempts))
            if confidence < s.gate_confidence_threshold:
                return self._abstain(
                    req, run_id, timings, attempts, llm_calls, prompt_chars,
                    "I am not confident in this answer: "
                    + ("; ".join(gate.warnings) if gate.warnings else "the checks were not clean")
                    + ". Here is the query I would have run, for you to verify.",
                    sql=sql, confidence=confidence, rows=ex.rows,
                )

        # ── stage 6 ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        syn = synthesize.synthesize(req.question, ex.rows, ex.truncated, self.llm)
        llm_calls += 1
        tick("synthesize", t0)

        tables = self._tables_used(gate)
        return AskResponse(
            question=req.question, status="answered", answer=syn.answer, sql=sql,
            confidence=confidence, confidence_label=label(confidence),
            rows=ex.rows[:50], row_count=ex.row_count, truncated=ex.truncated,
            tables_used=tables, attempts=attempts if req.explain else [],
            timings=timings, llm_calls=llm_calls, prompt_tokens_est=prompt_chars // 4,
            run_id=run_id, arm=req.arm,
        )

    # ── baseline C: the model reviews its own SQL ────────────────────────────
    def _self_check(self, question: str, sub, sql: str) -> tuple[str, int]:
        user = (
            f"# Schema\n\n{generate.render_schema(sub)}\n\n"
            f"# Question\n\n{question}\n\n# Query\n\n```sql\n{sql}\n```"
        )
        try:
            out = self.llm.complete(SELF_CHECK_SYSTEM, user)
        except Exception:
            return sql, 0
        if out.text.strip().upper().startswith("OK"):
            return sql, 1
        revised, _ = generate.parse_completion(out.text)
        return (revised or sql), 1

    # ── response helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _tables_used(gate: GateReport | None) -> list[str]:
        if gate is None:
            return []
        for c in gate.checks:
            if c.layer.value == "l2" and c.details.get("tables"):
                return list(c.details["tables"])
        return []

    def _abstain(self, req, run_id, timings, attempts, llm_calls, prompt_chars, reason,
                 sql: str | None = None, confidence: float = 0.0, rows=None) -> AskResponse:
        return AskResponse(
            question=req.question, status="abstained", answer=None, sql=sql,
            confidence=confidence, confidence_label=label(confidence),
            abstain_reason=reason, rows=(rows or [])[:20], row_count=len(rows or []),
            tables_used=[], attempts=attempts if req.explain else [], timings=timings,
            llm_calls=llm_calls, prompt_tokens_est=prompt_chars // 4, run_id=run_id, arm=req.arm,
        )

    def _blocked(self, req, run_id, timings, attempts, llm_calls, prompt_chars,
                 reason) -> AskResponse:
        return AskResponse(
            question=req.question, status="blocked", abstain_reason=reason,
            attempts=attempts if req.explain else [], timings=timings, llm_calls=llm_calls,
            prompt_tokens_est=prompt_chars // 4, run_id=run_id, arm=req.arm,
        )

    def _error(self, req, run_id, timings, reason) -> AskResponse:
        return AskResponse(question=req.question, status="error", abstain_reason=reason,
                           timings=timings, run_id=run_id, arm=req.arm)


_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None
