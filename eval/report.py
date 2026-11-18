"""Turn results/summary.csv into the tables that go in the paper and in Presentation 2.

    python -m eval.report                 # markdown tables
    python -m eval.report --latex         # LaTeX for the IEEE paper
    python -m eval.report --errors        # error taxonomy from the run log
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SUMMARY = Path("results/summary.csv")

MAIN_COLUMNS = [
    ("arm", "Arm"), ("ablated", "Ablated"), ("n", "N"), ("ex", "EX"),
    ("ex_of_answered", "EX when answered"), ("wrong_confident", "Confidently wrong"),
    ("abstained", "Abstained"), ("unsafe_block_rate", "Unsafe blocked"),
    ("penalty_10", "Score (c=10)"), ("calls_per_q", "Calls/q"), ("ms_per_q", "ms/q"),
]


def load() -> list[dict]:
    if not SUMMARY.exists():
        raise SystemExit(f"{SUMMARY} not found — run `python -m eval.runner` first.")
    with SUMMARY.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(key: str, value: str) -> str:
    if value in ("", "None"):
        return "—"
    if key in ("ex", "ex_of_answered", "unsafe_block_rate", "schema_recall", "schema_fpr"):
        return f"{float(value) * 100:.1f}%"
    if key.startswith("penalty"):
        return f"{float(value):+.3f}"
    if key in ("calls_per_q",):
        return f"{float(value):.2f}"
    if key in ("ms_per_q",):
        return f"{float(value):,.0f}"
    return value


def markdown(rows: list[dict]) -> str:
    keys = [k for k, _ in MAIN_COLUMNS]
    head = "| " + " | ".join(h for _, h in MAIN_COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in MAIN_COLUMNS) + "|"
    body = [
        "| " + " | ".join(fmt(k, r.get(k, "")) for k in keys) + " |"
        for r in rows
    ]
    return "\n".join([head, rule, *body])


def latex(rows: list[dict]) -> str:
    keys = [k for k, _ in MAIN_COLUMNS]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Execution accuracy, abstention behaviour and cost per arm.}",
        r"\label{tab:results}",
        r"\begin{tabular}{l" + "r" * (len(keys) - 1) + "}", r"\hline",
        " & ".join(h.replace("|", r"$\mid$") for _, h in MAIN_COLUMNS) + r" \\", r"\hline",
    ]
    for r in rows:
        lines.append(" & ".join(fmt(k, r.get(k, "")).replace("%", r"\%")
                                for k in keys) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def errors() -> str:
    """Error taxonomy straight from the run log — the Presentation 2 table."""
    from app.db.pool import owner_pool
    from app.core.config import get_settings

    s = get_settings()
    out = ["### Where the gate intervened", ""]
    with owner_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT status, count(*) FROM {s.run_log_schema}.runs GROUP BY status ORDER BY 2 DESC
        """)
        out += ["| status | runs |", "|---|---|"]
        out += [f"| {st} | {n} |" for st, n in cur.fetchall()]

        cur.execute(f"""
            SELECT g->>'attempt' AS attempt, c->>'layer' AS layer, count(*)
              FROM {s.run_log_schema}.runs r,
                   jsonb_array_elements(r.gate) AS g,
                   jsonb_array_elements(g->'checks') AS c
             WHERE (c->>'passed')::boolean IS FALSE
             GROUP BY 1, 2 ORDER BY 3 DESC
        """)
        rows = cur.fetchall()
        if rows:
            out += ["", "### Which layer rejected the query", "",
                    "| attempt | layer | times |", "|---|---|---|"]
            out += [f"| {a} | {layer} | {n} |" for a, layer, n in rows]

        cur.execute(f"""
            SELECT COALESCE(f.category, 'uncategorised'), count(*)
              FROM {s.run_log_schema}.feedback f WHERE f.is_wrong GROUP BY 1 ORDER BY 2 DESC
        """)
        rows = cur.fetchall()
        if rows:
            out += ["", "### Reported as wrong by a user", "",
                    "| category | count |", "|---|---|"]
            out += [f"| {c} | {n} |" for c, n in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--suite", default="")
    a = ap.parse_args()

    if a.errors:
        print(errors())
        return 0

    rows = load()
    if a.suite:
        rows = [r for r in rows if r["suite"] == a.suite]

    by_suite: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_suite[r["suite"]].append(r)

    for suite, group in by_suite.items():
        print(f"\n## Suite: {suite}\n")
        print(latex(group) if a.latex else markdown(group))

    print("\n_Runs:_ " + ", ".join(
        f"{k}×{v}" for k, v in Counter(r["model"] for r in rows).items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
