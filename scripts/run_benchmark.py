"""The experiment for Presentation 2 and the paper: our pipeline vs. a plain LLM, on a
public benchmark (BIRD Mini-Dev) and on our own enterprise database (AdventureWorks).

    python scripts\\run_benchmark.py                                  # Gemini, 100 BIRD + 30 AW
    python scripts\\run_benchmark.py --n 60                           # smaller
    python scripts\\run_benchmark.py --model ollama:qwen2.5-coder:7b  # a local model
    python scripts\\run_benchmark.py --report                         # just print the tables

Three arms, same model, same questions, same prompt wording:

  plain   the usual approach: whole schema + question → SQL → run it        (baseline)
  shrink  + the LLM shrinks the schema first, no checks                     (ablation)
  ours    + shrink + the 5-check gate + repair loop + abstention            (proposed)

BIRD Mini-Dev (500 questions over 11 real databases, Li et al., NeurIPS 2023 / bird-bench
Mini-Dev 2024) is the public benchmark: its authors publish PostgreSQL accuracy for
GPT-4, Llama-3 and others, so our numbers sit next to theirs. A stratified sample keeps
the run inside the Gemini free tier; the sample is seeded and saved, so every arm sees the
same questions and the run is reproducible.

Correctness follows BIRD: the gold SQL is executed on the ORIGINAL SQLite database, the
predicted SQL on the PostgreSQL copy, and the two result sets are compared
(order-insensitive). Nothing is judged by a model.

The run is resumable: every finished (question, arm) is appended to a .jsonl file, so if
the free-tier daily quota runs out, run the same command again later and it continues.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sqlite3
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import os  

os.environ.setdefault("GEMINI_MIN_INTERVAL_S", "4.5")  
os.environ.setdefault("GEMINI_RETRIES", "6")            
os.environ.setdefault("LLM_TIMEOUT_S", "300")           
os.environ.setdefault("OLLAMA_NUM_CTX", "16384")        

TRANSIENT = ("503", "429", "500 Internal Server Error", "high demand", "unreachable", "Connection refused", "ConnectError")


def is_transient(reason: str) -> bool:
    """Service hiccups are retried later; anything else is a real failure of that pipeline."""
    return any(t.lower() in (reason or "").lower() for t in TRANSIENT)

import psycopg   

from app.core.config import get_settings   
from app.engine import engine   
from app.engine.connections import pool_for   
from app.engine.schemas import AskOptions, AskV2Request   
from app.engine.target import ConnectionIn   
from app.pipeline import execute   
from app.pipeline.gate import l1_parse, l3_policy   
from eval.datasets.suites import ADVENTUREWORKS   
from eval.metrics import execution_match   
from scripts import _bird_pg   

BIRD_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
BIRD_DIR = ROOT / "data" / "bird"
RESULTS = ROOT / "results"

ARMS = {
    "plain": dict(schema_strategy="full", verify=False, max_repairs=0),
    "shrink": dict(schema_strategy="llm", verify=False, max_repairs=0),
    "ours": dict(schema_strategy="llm", verify=True, max_repairs=2),
}
ARM_LABEL = {"plain": "Plain LLM (full schema, no checks)",
             "shrink": "+ LLM schema shrink",
             "ours": "Ours: shrink + 5-check gate + repair + abstain"}

PUBLISHED_PG = [
    ("Mixtral-8x7B", 12.40), ("Llama-3-8B-Instruct", 18.40), ("Phi-3-medium", 21.60),
    ("GPT-3.5-turbo-instruct", 26.60), ("GPT-3.5-turbo", 27.40), ("Llama-3-70B-Instruct", 29.40),
    ("GPT-4-32k", 35.00), ("GPT-4", 35.80), ("GPT-4-turbo", 36.00),
]

def ensure_bird(zip_path: Path | None) -> Path:
    """Returns the folder holding mini_dev_sqlite.json and dev_databases/."""
    BIRD_DIR.mkdir(parents=True, exist_ok=True)
    qfile = BIRD_DIR / "mini_dev_sqlite.json"
    if qfile.exists():
        return BIRD_DIR
    zp = zip_path or BIRD_DIR / "minidev.zip"
    if not zp.exists():
        print(f"downloading BIRD Mini-Dev (~1 GB, once) from\n  {BIRD_URL}")
        try:
            with urllib.request.urlopen(BIRD_URL, timeout=120) as r, open(zp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, last = 0, 0.0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 2:
                        last = time.time()
                        pct = f"{done / total:.0%}" if total else ""
                        print(f"  {done / 1e6:,.0f} MB {pct}", end="\r", flush=True)
            print()
        except Exception as exc:
            zp.unlink(missing_ok=True)
            raise SystemExit(
                f"\nDownload failed ({exc}).\nDownload minidev.zip in your browser from\n  {BIRD_URL}\n"
                "or https://huggingface.co/datasets/birdsql/bird_mini_dev ,\n"
                f"save it as {BIRD_DIR / 'minidev.zip'} and run this again.")
    print("reading the question file from the zip ...")
    with zipfile.ZipFile(zp) as z:
        member = next((m for m in z.namelist() if m.endswith("mini_dev_sqlite.json")), None)
        if member is None:
            raise SystemExit("mini_dev_sqlite.json not found in the zip.")
        with z.open(member) as src, open(qfile, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return BIRD_DIR


def sqlite_file(db_id: str, zip_path: Path | None) -> Path:
    target = BIRD_DIR / "dev_databases" / db_id / f"{db_id}.sqlite"
    if target.exists():
        return target
    zp = zip_path or BIRD_DIR / "minidev.zip"
    with zipfile.ZipFile(zp) as z:
        member = next((m for m in z.namelist()
                       if re.search(rf"(^|/){re.escape(db_id)}/{re.escape(db_id)}\.sqlite$", m)), None)
        if member is None:
            raise SystemExit(f"{db_id}.sqlite not found in {zp}")
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  extracting {db_id}.sqlite ...")
        with z.open(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return target


def sample_bird(n: int, seed: int) -> list[dict]:
    records = json.loads((BIRD_DIR / "mini_dev_sqlite.json").read_text(encoding="utf-8"))
    by_level: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_level[r.get("difficulty", "moderate")].append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for level, rows in sorted(by_level.items()):
        k = round(n * len(rows) / len(records))
        out += rng.sample(rows, min(k, len(rows)))
    out.sort(key=lambda r: r["question_id"])
    return out[:n] if len(out) > n else out


def pg_db_name(db_id: str) -> str:
    return f"bird_{db_id}".lower()


def ensure_pg(db_id: str, zip_path: Path | None) -> None:
    s = get_settings()
    base = f"host={s.pghost} port={s.pgport} user={s.pguser} password={s.pgpassword}"
    name = pg_db_name(db_id)
    sq = sqlite3.connect(str(sqlite_file(db_id, zip_path)))
    n_src = sq.execute("SELECT count(*) FROM sqlite_master WHERE type='table' "
                       "AND name NOT LIKE 'sqlite_%'").fetchone()[0]
    sq.close()
    try:
        with psycopg.connect(f"{base} dbname={name}", connect_timeout=8) as c:
            n_pg = c.execute("SELECT count(*) FROM pg_tables WHERE schemaname='public'").fetchone()[0]
        if n_pg == n_src:
            return
    except psycopg.OperationalError:
        pass
    print(f"  loading {db_id} into PostgreSQL database '{name}' ...")
    rep = _bird_pg.convert(sqlite_file(db_id, zip_path), f"{base} dbname=postgres",
                           lambda db: f"{base} dbname={db}", name, log=lambda m: None)
    print(f"    {rep['tables']} tables, {rep['rows']:,} rows, {rep['primary_keys']} PKs, "
          f"{rep['foreign_keys']} FKs in {rep['seconds']}s")


def gold_rows(db_id: str, sql: str, zip_path: Path | None, timeout_s: float = 60) -> list[dict] | None:
    con = sqlite3.connect(str(sqlite_file(db_id, zip_path)))
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.time() + timeout_s
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 20000)
    try:
        cur = con.execute(sql)
        names = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip([f"c{i}" for i in range(len(names))], r)) for r in cur.fetchall()]
    except Exception as exc:
        print(f"    gold SQL failed on SQLite ({exc}); question skipped")
        return None
    finally:
        con.close()

def harmful(sql: str | None) -> bool:
    """A write, a stacked statement, a system-catalog read or a dangerous function."""
    if not sql:
        return False
    return any(c.message.startswith("blocked") for c in (l1_parse.check(sql), l3_policy.check(sql)))


def run_one(question: str, database: str | None, arm: str, model: str) -> dict:
    conn = ConnectionIn(use_default=True, database=database or "")
    req = AskV2Request(question=question, connection=conn, model=model,
                       options=AskOptions(**ARMS[arm], answer=False, row_limit=5000, log=False))
    t0 = time.perf_counter()
    res = engine.run_sync(req)
    out = {
        "status": res.status, "sql": res.sql, "reason": res.abstain_reason,
        "attempts": len(res.attempts), "confidence": res.confidence,
        "llm_calls": res.llm_calls, "prompt_tokens": res.prompt_tokens_est,
        "ms": round((time.perf_counter() - t0) * 1000),
        "tables_before": res.shrink.get("tables_before"), "tables_after": res.shrink.get("tables_after"),
        "schema_tokens_full": res.shrink.get("full_schema_tokens"),
        "schema_tokens_used": res.shrink.get("shrunk_schema_tokens"),
        "hallucinated": len(res.shrink.get("hallucinated_tables") or []) +
                        len(res.shrink.get("hallucinated_columns") or []),
        "failed_layers": [next((c.layer for c in a.checks if not c.passed and not c.skipped), None)
                          for a in res.attempts],
    }
    sql_any = res.sql or (res.attempts[-1].sql if res.attempts else None)
    out["harmful_sql_generated"] = harmful(sql_any)
    # The plain and shrink arms have no gate: whatever SQL the model wrote goes to the DB.
    out["harmful_sql_reached_db"] = out["harmful_sql_generated"] and arm != "ours"
    return out


def predicted_rows(sql: str, database: str | None) -> list[dict] | None:
    target = ConnectionIn(use_default=True, database=database or "").to_target()
    ex = execute.run_query(sql, row_limit=100000, pool=pool_for(target))
    return None if ex.error else ex.rows

def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for arm in ARMS:
        r = [x for x in rows if x["arm"] == arm]
        if not r:
            continue
        ans = [x for x in r if x["kind"] == "answerable"]
        answered = [x for x in ans if x["status"] == "answered"]
        correct = [x for x in ans if x.get("correct")]
        wrong_conf = [x for x in answered if not x.get("correct")]
        unsafe = [x for x in r if x["kind"] == "unsafe"]
        unans = [x for x in r if x["kind"] == "unanswerable"]
        by_level = {}
        for lvl in sorted({x.get("difficulty") for x in ans}):
            g = [x for x in ans if x.get("difficulty") == lvl]
            by_level[lvl] = {"n": len(g), "ex": round(100 * sum(1 for x in g if x.get("correct")) / len(g), 1)}
        exec_err = [x for x in ans if x["status"] != "answered" and arm != "ours"]
        out[arm] = {
            "label": ARM_LABEL[arm], "n": len(ans),
            "ex": round(100 * len(correct) / len(ans), 1) if ans else None,
            "answered": len(answered),
            "precision": round(100 * len(correct) / len(answered), 1) if answered else None,
            "wrong_answers_shown": len(wrong_conf),
            "abstained": sum(1 for x in ans if x["status"] == "abstained" and arm == "ours"),
            "failed_to_run": len(exec_err),
            "fixed_by_repair": sum(1 for x in correct if x["attempts"] > 1),
            "by_difficulty": by_level,
            "unsafe_n": len(unsafe),
            "unsafe_blocked": sum(1 for x in unsafe if x["status"] == "blocked"),
            "unsafe_reached_db": sum(1 for x in unsafe if x["harmful_sql_reached_db"]),
            "unanswerable_n": len(unans),
            "unanswerable_declined": sum(1 for x in unans if x["status"] in ("abstained", "blocked")),
            "avg_llm_calls": round(sum(x["llm_calls"] for x in r) / len(r), 2),
            "avg_prompt_tokens": round(sum(x["prompt_tokens"] for x in r) / len(r)),
            "avg_seconds": round(sum(x["ms"] for x in r) / len(r) / 1000, 1),
            "avg_tables_used": round(sum((x["tables_after"] or 0) for x in r) / len(r), 1),
            "avg_tables_total": round(sum((x["tables_before"] or 0) for x in r) / len(r), 1),
            "hallucinated_names_caught": sum(x.get("hallucinated", 0) for x in r),
        }
    return out


def print_report(suite: str, s: dict, model: str) -> str:
    lines = [f"\n### {suite} — {model}\n",
             "| | " + " | ".join(ARM_LABEL[a] for a in s) + " |",
             "|---|" + "---|" * len(s)]

    def row(label, key, fmt="{}"):
        lines.append(f"| {label} | " + " | ".join(
            (fmt.format(s[a][key]) if s[a].get(key) is not None else "–") for a in s) + " |")

    row("questions with an answer", "n")
    row("execution accuracy (EX) %", "ex")
    row("accuracy when it answered %", "precision")
    row("wrong answers shown to the user", "wrong_answers_shown")
    row("abstained (said 'not sure')", "abstained")
    row("query crashed at run time", "failed_to_run")
    row("fixed by the repair loop", "fixed_by_repair")
    for lvl in next(iter(s.values()))["by_difficulty"]:
        lines.append(f"| EX {lvl} % | " + " | ".join(
            str(s[a]["by_difficulty"].get(lvl, {}).get("ex", "–")) for a in s) + " |")
    if any(s[a]["unsafe_n"] for a in s):
        row("harmful requests", "unsafe_n")
        row("harmful SQL blocked", "unsafe_blocked")
        row("harmful SQL sent to the database", "unsafe_reached_db")
        row("out-of-scope questions declined", "unanswerable_declined")
    row("avg model calls / question", "avg_llm_calls")
    row("avg prompt tokens / question", "avg_prompt_tokens")
    row("avg seconds / question", "avg_seconds")
    row("avg tables given to the SQL step", "avg_tables_used")
    text = "\n".join(lines)
    print(text)
    return text

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemini:" + get_settings().gemini_model)
    ap.add_argument("--n", type=int, default=100, help="BIRD questions (stratified sample)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="plain,shrink,ours")
    ap.add_argument("--skip-bird", action="store_true")
    ap.add_argument("--skip-aw", action="store_true")
    ap.add_argument("--zip", type=Path, default=None, help="minidev.zip if downloaded by hand")
    ap.add_argument("--report", action="store_true", help="only print the tables")
    ap.add_argument("--pause", type=float, default=60.0,
                    help="seconds to wait between runs so the model server can cool down (default 60)")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x in ARMS]
    tag = re.sub(r"[^a-z0-9.]+", "-", a.model.lower()).strip("-")
    RESULTS.mkdir(exist_ok=True)
    log_path = RESULTS / f"benchmark_{tag}.jsonl"
    done = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("status") == "error" and is_transient(r.get("reason") or ""):
                continue  # a service hiccup is not a result: it is retried
            done[(r["suite"], r["qid"], r["arm"])] = r

    if not a.report:
        print(f"model: {a.model}   arms: {', '.join(arms)}   results: {log_path}")
        plan: list[tuple[str, dict]] = []
        if not a.skip_aw:
            plan += [("adventureworks", {"qid": it.id, "question": it.question, "kind": it.kind,
                                         "difficulty": it.difficulty, "gold": it.gold_sql, "db": None})
                     for it in ADVENTUREWORKS]
        if not a.skip_bird:
            ensure_bird(a.zip)
            sample = sample_bird(a.n, a.seed)
            (RESULTS / f"bird_sample_seed{a.seed}_n{a.n}.json").write_text(
                json.dumps([r["question_id"] for r in sample]), encoding="utf-8")
            print(f"BIRD sample: {len(sample)} questions, "
                  f"{dict(Counter(r['difficulty'] for r in sample))}, "
                  f"{len({r['db_id'] for r in sample})} databases")
            for db in sorted({r["db_id"] for r in sample}):
                ensure_pg(db, a.zip)
            for r in sample:
                ev = (r.get("evidence") or "").strip()
                plan.append(("bird", {"qid": f"bird-{r['question_id']}", "db_id": r["db_id"],
                                      "question": f"{r['question'].strip()} (Hint: {ev})" if ev else r["question"].strip(),
                                      "kind": "answerable", "difficulty": r.get("difficulty"),
                                      "gold": r["SQL"], "db": pg_db_name(r["db_id"])}))

        todo = [(suite, q, arm) for suite, q in plan for arm in arms
                if (suite, q["qid"], arm) not in done]
        print(f"{len(todo)} runs to do ({len(done)} already done)\n")
        gold_cache: dict[str, list | None] = {}
        aw_gold: dict[str, list | None] = {}
        t_start = time.time()
        errors_in_row = 0
        with log_path.open("a", encoding="utf-8") as log:
            for i, (suite, q, arm) in enumerate(todo, 1):
                out = run_one(q["question"], q["db"], arm, a.model)
                reason = (out["reason"] or "")
                if out["status"] == "error" and ("429" in reason or "quota" in reason.lower()
                                                 and "exhausted" in reason.lower()):
                    print(f"\nThe free-tier quota is used up for now ({reason[:160]}).\n"
                          "Everything so far is saved. Run the same command later to continue.")
                    break
                if out["status"] == "error" and is_transient(reason):
                    print(f"  {i:>4}/{len(todo)} ! {arm:<6} {q['qid']:<11} skipped for now: {reason[:90]}")
                    errors_in_row += 1
                    if errors_in_row >= 8:
                        print("\nThe model keeps failing (overloaded or not running?). Everything so far is saved; "
                              "run the same command again in a few minutes to continue.")
                        break
                    time.sleep(max(20, a.pause))
                    continue
                errors_in_row = 0
                if a.pause and i < len(todo):
                    time.sleep(a.pause)
                correct = None
                if q["kind"] == "answerable" and q["gold"]:
                    if suite == "bird":
                        if q["qid"] not in gold_cache:
                            gold_cache[q["qid"]] = gold_rows(q["db_id"], q["gold"], a.zip)
                        gold = gold_cache[q["qid"]]
                    else:
                        if q["qid"] not in aw_gold:
                            ex = execute.run_query(q["gold"], row_limit=100000, pool=pool_for(
                                ConnectionIn(use_default=True).to_target()))
                            aw_gold[q["qid"]] = None if ex.error else ex.rows
                        gold = aw_gold[q["qid"]]
                    if gold is None:
                        continue
                    if out["status"] == "answered" and out["sql"]:
                        pred = predicted_rows(out["sql"], q["db"])
                        correct = pred is not None and execution_match(pred, gold)
                    else:
                        correct = False
                rec = {"suite": suite, "qid": q["qid"], "arm": arm, "kind": q["kind"],
                       "difficulty": q["difficulty"], "question": q["question"],
                       "correct": correct, **out}
                log.write(json.dumps(rec, default=str) + "\n")
                log.flush()
                done[(suite, q["qid"], arm)] = rec
                mark = {True: "✓", False: "✗", None: "·"}[correct]
                if q["kind"] == "unsafe":
                    mark = "⊘" if out["status"] == "blocked" else "!"
                eta = (time.time() - t_start) / i * (len(todo) - i) / 60
                print(f"  {i:>4}/{len(todo)} {mark} {arm:<6} {q['qid']:<11} {out['status']:<9} "
                      f"{out['ms'] / 1000:5.1f}s  ~{eta:.0f} min left")

    rows = list(done.values())
    report = [f"# Benchmark results — {a.model}\n",
              f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Same model, same questions, same prompt "
              "for every arm; correctness by executing the gold SQL (BIRD: on the original SQLite "
              "database) and comparing result sets."]
    summary = {}
    for suite in ("bird", "adventureworks"):
        sr = [r for r in rows if r["suite"] == suite]
        if sr:
            summary[suite] = summarise(sr)
            report.append(print_report("BIRD Mini-Dev (public benchmark, PostgreSQL)" if suite == "bird"
                                       else "AdventureWorks (enterprise database, 74 tables, 10M rows)",
                                       summary[suite], a.model))
    if "bird" in summary:
        pub = "\n\n### Published BIRD Mini-Dev PostgreSQL EX (full 500 questions)\n\n| model | EX % |\n|---|---|\n"
        pub += "\n".join(f"| {m} | {v:.1f} |" for m, v in PUBLISHED_PG)
        report.append(pub)
        print(pub)
    (RESULTS / f"benchmark_{tag}_summary.json").write_text(
        json.dumps({"model": a.model, "summary": summary, "published_pg": PUBLISHED_PG}, indent=2),
        encoding="utf-8")
    (RESULTS / f"benchmark_{tag}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nwritten: {RESULTS / f'benchmark_{tag}_summary.json'}\n         {RESULTS / f'benchmark_{tag}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
