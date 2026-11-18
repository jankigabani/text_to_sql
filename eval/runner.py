"""Run a suite through one or more arms and write the results.

    python -m eval.runner --suite adventureworks --arm ours
    python -m eval.runner --suite adventureworks --arm all
    python -m eval.runner --suite adventureworks --arm ours --ablate l4
    python -m eval.runner --suite adventureworks --arm ours --seeds 3

The v2 engine (what the web UI runs: LLM schema shrink → SQL → gate → grounded answer) can
be evaluated with any model, so qwen, llama and Gemini can be compared on the same suite:

    python -m eval.runner --engine v2 --model ollama:qwen2.5-coder:7b --arm all
    python -m eval.runner --engine v2 --model ollama:llama3.1:latest  --arm ours
    python -m eval.runner --engine v2 --model gemini:gemini-2.5-flash --arm ours

For v2 the arms mean: baseline_a = full schema, no gate · baseline_b = LLM-shrunk schema,
no gate · baseline_c = full schema + gate · ours = LLM-shrunk schema + gate.

Results land in results/<suite>_<arm>[_ablate-xx]_<timestamp>.json, and a one-line summary
is appended to results/summary.csv so several runs can be compared later.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings, reload_settings   
from app.core.models import AskRequest   
from app.db import introspect   
from app.pipeline import execute   
from app.pipeline.orchestrator import Pipeline   
from eval.datasets.suites import Item, get_suite   
from eval.metrics import Outcome, Summary, execution_match, summarise   

ARMS = ["baseline_a", "baseline_b", "baseline_c", "ours"]
RESULTS = Path("results")


def run_item(p: Pipeline, item: Item, arm: str, disabled: list[str]) -> Outcome:
    t0 = time.perf_counter()
    resp = p.ask(AskRequest(question=item.question, arm=arm,
                            disabled_layers=disabled, explain=True))
    elapsed = (time.perf_counter() - t0) * 1000

    correct: bool | None = None
    if item.kind == "answerable" and item.gold_sql:
        if resp.status == "answered" and resp.sql:
            gold = execute.run_query(item.gold_sql, row_limit=1000)
            pred = execute.run_query(resp.sql, row_limit=1000)
            correct = (
                gold.error is None and pred.error is None
                and execution_match(pred.rows, gold.rows)
            )
        else:
            correct = False

    failed_layer = None
    for a in resp.attempts:
        if a.gate and not a.gate.passed:
            fc = a.gate.failed_check()
            failed_layer = fc.layer.value if fc else None

    return Outcome(
        item_id=item.id, kind=item.kind, status=resp.status, correct=correct,
        confidence=resp.confidence, llm_calls=resp.llm_calls,
        prompt_tokens=resp.prompt_tokens_est, total_ms=elapsed,
        gate_layer_failed=failed_layer, tables_predicted=resp.tables_used,
        tables_gold=item.gold_tables, sql=resp.sql, note=item.note,
    )


V2_ARMS = {
    "baseline_a": {"schema_strategy": "full", "verify": False},
    "baseline_b": {"schema_strategy": "llm", "verify": False},
    "baseline_c": {"schema_strategy": "full", "verify": True},
    "ours": {"schema_strategy": "llm", "verify": True},
}


def run_item_v2(item: Item, arm: str, disabled: list[str], model: str) -> Outcome:
    from app.engine import engine
    from app.engine.schemas import AskOptions, AskV2Request

    t0 = time.perf_counter()
    res = engine.run_sync(AskV2Request(
        question=item.question, model=model,
        options=AskOptions(**V2_ARMS[arm], disabled_layers=disabled)))
    elapsed = (time.perf_counter() - t0) * 1000

    correct: bool | None = None
    if item.kind == "answerable" and item.gold_sql:
        if res.status == "answered" and res.sql:
            gold = execute.run_query(item.gold_sql, row_limit=1000)
            pred = execute.run_query(res.sql, row_limit=1000)
            correct = (gold.error is None and pred.error is None
                       and execution_match(pred.rows, gold.rows))
        else:
            correct = False

    failed_layer = None
    for a in res.attempts:
        bad = next((c for c in a.checks if not c.passed and not c.skipped), None)
        if bad:
            failed_layer = bad.layer
    return Outcome(
        item_id=item.id, kind=item.kind, status=res.status, correct=correct,
        confidence=res.confidence, llm_calls=res.llm_calls,
        prompt_tokens=res.prompt_tokens_est, total_ms=elapsed,
        gate_layer_failed=failed_layer,
        tables_predicted=[t["table"] for t in res.selected_tables if t.get("source") != "full"],
        tables_gold=item.gold_tables, sql=res.sql, note=item.note,
    )


def print_summary(s: Summary) -> None:
    print(f"\n── {s.suite} / {s.arm} ──────────────────────────────")
    print(f"  questions                 {s.n}")
    print(f"  execution accuracy (EX)   {s.ex:.1%}   ({s.n_correct}/{s.n_answerable})")
    print(f"  accuracy when it answered {s.ex_of_answered:.1%}")
    print(f"  confidently wrong         {s.wrong_confident}")
    print(f"  abstained                 {s.n_abstained}")
    print(f"  blocked                   {s.n_blocked}")
    print(f"  unsafe blocked            {s.unsafe_block_rate:.0%}  "
          f"({s.n_unsafe_blocked}/{s.n_unsafe})")
    print(f"  unanswerable handled      {s.n_unanswerable_handled}/{s.n_unanswerable}")
    if s.schema_recall is not None:
        fpr = "n/a" if s.schema_fpr is None else f"{s.schema_fpr:.1%}"
        print(f"  schema recall / FPR       {s.schema_recall:.1%} / {fpr}")
    print(f"  model calls per question  {s.calls_per_question:.2f}")
    print(f"  ms per question           {s.ms_per_question:,.0f}")
    for name, value in s.scores.items():
        c = name.split("_")[1]
        print(f"  abstention score (c={c:>2}) {value:+.3f}")


def append_csv(s: Summary, path: Path, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "suite": s.suite, "arm": s.arm, **extra,
        "n": s.n, "ex": round(s.ex, 4), "ex_of_answered": round(s.ex_of_answered, 4),
        "wrong_confident": s.wrong_confident, "abstained": s.n_abstained,
        "blocked": s.n_blocked, "unsafe_block_rate": round(s.unsafe_block_rate, 4),
        "schema_recall": None if s.schema_recall is None else round(s.schema_recall, 4),
        "schema_fpr": None if s.schema_fpr is None else round(s.schema_fpr, 4),
        "calls_per_q": round(s.calls_per_question, 3),
        "ms_per_q": round(s.ms_per_question, 1),
        **{k: v for k, v in s.scores.items()},
    }
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="adventureworks")
    ap.add_argument("--arm", default="ours", choices=[*ARMS, "all"])
    ap.add_argument("--ablate", default="", help="layers to disable, e.g. l4 or l2,l5")
    ap.add_argument("--seeds", type=int, default=1, help="repeat the suite N times")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--provider", default="", help="override LLM_PROVIDER for this run")
    ap.add_argument("--engine", default="v1", choices=["v1", "v2"],
                    help="v1 = embedding retrieval pipeline; v2 = the web UI engine")
    ap.add_argument("--model", default="", help="v2 only: provider:model, e.g. ollama:llama3.1:latest")
    a = ap.parse_args()

    if a.provider:
        os.environ["LLM_PROVIDER"] = a.provider
        reload_settings()

    items = get_suite(a.suite)
    if a.limit:
        items = items[: a.limit]
    arms = ARMS if a.arm == "all" else [a.arm]
    disabled = [x.strip() for x in a.ablate.split(",") if x.strip()]

    s = get_settings()
    cat = introspect.get_catalog()
    if a.engine == "v2":
        from app.llm.providers import default_spec, parse_spec
        model_spec = a.model or default_spec()
        provider_name, model_name = parse_spec(model_spec)
    else:
        model_spec, provider_name, model_name = "", s.llm_provider, s.llm_model
    print(f"engine={a.engine} suite={a.suite} items={len(items)} arms={arms} "
          f"ablate={disabled or 'none'} provider={provider_name} model={model_name}")
    print(f"database={s.pgdatabase} tables={len(cat.tables)}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for arm in arms:
        all_outcomes: list[Outcome] = []
        for seed in range(1, a.seeds + 1):
            p = Pipeline(catalog=cat) if a.engine == "v1" else None
            for i, item in enumerate(items, 1):
                out = (run_item(p, item, arm, disabled) if p is not None
                       else run_item_v2(item, arm, disabled, model_spec))
                all_outcomes.append(out)
                flag = {"answered": "✓" if out.correct is not False else "✗",
                        "abstained": "~", "blocked": "⊘", "error": "!"}[out.status]
                print(f"  [{seed}] {i:>3}/{len(items)} {flag} {item.id:<7} "
                      f"{item.question[:58]:<58} conf={out.confidence:.2f}")

        summary = summarise(arm, a.suite, all_outcomes)
        print_summary(summary)

        tag = f"_ablate-{'-'.join(disabled)}" if disabled else ""
        if a.engine == "v2":
            tag = f"_v2-{model_name.replace(':', '-').replace('/', '-')}{tag}"
        out_path = RESULTS / f"{a.suite}_{arm}{tag}_{stamp}.json"
        out_path.write_text(json.dumps({
            "suite": a.suite, "arm": arm, "ablated": disabled, "seeds": a.seeds,
            "engine": a.engine, "provider": provider_name, "model": model_name,
            "database": s.pgdatabase,
            "settings": {
                "max_repairs": s.gate_max_repairs,
                "confidence_threshold": s.gate_confidence_threshold,
                "plan_cost_ceiling": s.gate_plan_cost_ceiling,
            },
            "summary": summary.__dict__ | {"ex": summary.ex,
                                           "ex_of_answered": summary.ex_of_answered,
                                           "unsafe_block_rate": summary.unsafe_block_rate},
            "outcomes": [o.__dict__ for o in all_outcomes],
        }, indent=2, default=str), encoding="utf-8")
        append_csv(summary, RESULTS / "summary.csv",
                   {"ablated": ",".join(disabled), "model": model_name,
                    "provider": provider_name, "seeds": a.seeds, "engine": a.engine})
        print(f"\n  written: {out_path}")

    print(f"\nsummary table: {RESULTS / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
