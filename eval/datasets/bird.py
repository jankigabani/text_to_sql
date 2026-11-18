"""BIRD suite loader.

BIRD is distributed by its authors and cannot be downloaded unattended, so this reads the
files from a local folder. `scripts/05_load_bird.py` prints the download steps and converts
the SQLite databases to PostgreSQL.

Expected layout (BIRD_DIR, default data/bird):

    data/bird/
      mini_dev_postgresql.json      or  dev.json / mini_dev_sqlite.json
      dev_databases/<db_id>/<db_id>.sqlite      (only needed for the conversion step)

Each BIRD record becomes an `Item` with the gold SQL attached, so the same runner and the
same metrics are used for BIRD and for the AdventureWorks suite — which is the point: one
harness, several suites.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from eval.datasets.suites import Item

CANDIDATES = (
    "mini_dev_postgresql.json",
    "mini_dev_sqlite.json",
    "dev.json",
    "mini_dev.json",
)


def bird_dir() -> Path:
    return Path(os.environ.get("BIRD_DIR", "data/bird"))


def _difficulty(raw: str | None) -> str:
    v = (raw or "").lower()
    return v if v in ("easy", "medium", "hard") else "medium"


def load_bird(limit: int = 0, db_id: str | None = None) -> list[Item]:
    folder = bird_dir()
    path = next((folder / name for name in CANDIDATES if (folder / name).exists()), None)
    if path is None:
        raise SystemExit(
            f"No BIRD question file found in {folder.resolve()}.\n"
            "Run `python scripts/05_load_bird.py --help` for the download steps."
        )

    records = json.loads(path.read_text(encoding="utf-8"))
    items: list[Item] = []
    for i, r in enumerate(records):
        if db_id and r.get("db_id") != db_id:
            continue
        gold = r.get("SQL") or r.get("sql") or r.get("query")
        question = r.get("question", "").strip()
        if not question:
            continue
        evidence = (r.get("evidence") or "").strip()
        items.append(Item(
            id=f"bird-{r.get('question_id', i)}",
            # BIRD ships external knowledge separately; appending it is the standard
            # protocol, and it must be reported as part of the setup.
            question=f"{question} ({evidence})" if evidence else question,
            kind="answerable" if gold else "unanswerable",
            gold_sql=gold,
            difficulty=_difficulty(r.get("difficulty")),
            note=f"db_id={r.get('db_id', '?')}",
        ))
        if limit and len(items) >= limit:
            break
    return items
