"""The verification gate: L1 → L2 → L3 → L4 → L5, plus confidence scoring.

Ordering is deliberate and cost-ordered: microseconds first, the database last. A query
that fails L1 never reaches the planner; a query that is not read-only never runs at all.

`run_gate` is pure in the sense that matters for the experiment: given the same SQL,
catalog and settings, it returns the same report. Layers can be switched off individually
(`disabled`), which is how the per-layer ablation in the evaluation plan is produced.
"""
from __future__ import annotations

import time

from app.core.config import get_settings
from app.core.models import Catalog, CheckResult, GateReport, Layer
from app.pipeline.gate import l1_parse, l2_bind, l3_policy, l4_plan, l5_sniff

# Policy (L3) runs before bind (L2). Both cost microseconds, but the order decides how a
# request is *classified*: `SELECT * FROM pg_authid` must be reported as blocked, not as
# "unknown table" — otherwise an attack would be quietly sent back for repair.
ORDER = [Layer.L1_PARSE, Layer.L3_POLICY, Layer.L2_BIND, Layer.L4_PLAN, Layer.L5_SNIFF]


def run_gate(
    sql: str,
    cat: Catalog,
    question: str = "",
    attempt: int = 1,
    disabled: set[str] | None = None,
    run_l5: bool = True,
    pool=None,
    row_limit: int | None = None,
) -> GateReport:
    s = get_settings()
    off = {x.lower() for x in (disabled if disabled is not None else s.disabled_layers)}
    t0 = time.perf_counter()
    checks: list[CheckResult] = []

    for layer in ORDER:
        if layer.value in off or (layer is Layer.L5_SNIFF and not run_l5):
            checks.append(CheckResult(layer=layer, passed=True, skipped=True,
                                      message="disabled for this run"))
            continue

        if layer is Layer.L1_PARSE:
            res = l1_parse.check(sql)
        elif layer is Layer.L2_BIND:
            res = l2_bind.check(sql, cat)
        elif layer is Layer.L3_POLICY:
            res = l3_policy.check(sql)
        elif layer is Layer.L4_PLAN:
            res = l4_plan.check(sql, pool=pool)
        else:
            res = l5_sniff.check(sql, question, row_limit=row_limit, pool=pool)

        checks.append(res)
        if not res.passed:
            break  # fail fast: later layers would be meaningless

    passed = all(c.passed for c in checks)
    return GateReport(
        passed=passed,
        attempt=attempt,
        checks=checks,
        warnings=[c.message for c in checks if c.warning],
        total_ms=(time.perf_counter() - t0) * 1000,
    )


# ── confidence ────────────────────────────────────────────────────────────────
# Weights: how much each signal contributes when it is clean. Chosen once, documented,
# and never tuned per question — the evaluation reports calibration as it stands.
W_LAYERS = 0.55          # all layers passed
W_FIRST_ATTEMPT = 0.20   # no repair was needed
W_MARGIN = 0.15          # retrieval was confident about the tables
W_RESULT_SHAPE = 0.10    # the result looks like an answer


def score_confidence(report: GateReport, retrieval_margin: float, attempts: int) -> float:
    if not report.passed:
        return 0.0

    ran = [c for c in report.checks if not c.skipped]
    layer_score = (sum(1 for c in ran if c.passed) / len(ran)) if ran else 0.0

    attempt_score = 1.0 if attempts <= 1 else max(0.0, 1.0 - 0.4 * (attempts - 1))
    margin_score = max(0.0, min(1.0, retrieval_margin))

    sniff = next((c for c in report.checks if c.layer is Layer.L5_SNIFF and not c.skipped), None)
    if sniff is None:
        shape_score = 0.5
    elif sniff.warning:
        shape_score = 0.0
    else:
        shape_score = 1.0

    raw = (
        W_LAYERS * layer_score
        + W_FIRST_ATTEMPT * attempt_score
        + W_MARGIN * margin_score
        + W_RESULT_SHAPE * shape_score
    )
    # A warning is a strong negative signal on its own.
    if report.warnings:
        raw *= 0.75
    return round(max(0.0, min(1.0, raw)), 4)


def label(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.55:
        return "medium"
    if confidence > 0:
        return "low"
    return "none"
