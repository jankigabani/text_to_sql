"""Introspect the database and build the two retrieval indexes (stages 1–2).

Run this once after loading data, and again whenever the schema changes. The catalog and
the index are cached under INDEX_DIR, so the API starts instantly.

    python scripts/04_build_schema_index.py
    python scripts/04_build_schema_index.py --no-samples   # faster, skips value sampling
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings   
from app.db import introspect   
from app.db import runlog   
from app.retrieval.index import build_index   


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-samples", action="store_true",
                    help="skip per-column sample values (much faster, slightly worse prompts)")
    a = ap.parse_args()

    print(f"introspecting {s.pgdatabase} (schemas: {', '.join(s.schema_allowlist)})")
    t0 = time.perf_counter()
    cat = introspect.introspect(sample_values=not a.no_samples)
    introspect.save(cat)
    print(f"  {len(cat.tables)} tables, "
          f"{sum(len(t.columns) for t in cat.tables)} columns, "
          f"{len(cat.foreign_keys)} foreign keys   ({time.perf_counter() - t0:.1f}s)")
    print(f"  schema fingerprint: {cat.schema_version}")

    sampled = sum(1 for t in cat.tables for c in t.columns if c.sample_values)
    print(f"  sample values collected for {sampled} columns")

    print(f"building retrieval index (embeddings: {s.embed_provider})")
    t0 = time.perf_counter()
    idx = build_index(cat)
    idx.save()
    print(f"  {len(idx.tables.keys)} table cards, {len(idx.columns.keys)} column cards "
          f"({time.perf_counter() - t0:.1f}s, provider={idx.provider})")
    if idx.provider == "hashing":
        print("  note: Ollama embeddings were unavailable, so the keyword fallback was used.")
        print("        For the reported results run: ollama pull nomic-embed-text")

    if s.run_log_enabled:
        try:
            runlog.ensure_schema()
            print(f"run log ready: schema '{s.run_log_schema}'")
        except Exception as exc:
            print(f"  warning: could not create the run log: {exc}")

    print(f"\nIndex written to {Path(s.index_dir).resolve()}")
    print("Next: uvicorn app.main:app --reload   then open http://127.0.0.1:8000/docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
