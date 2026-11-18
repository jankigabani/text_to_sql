"""Schema per database, cached — and invalidated automatically when the schema changes.

Introspecting AdventureWorks with sample values takes a few seconds, which is too slow to
repeat on every question. So the catalog is cached in memory and on disk per target. On
each request a single cheap query computes a hash of every table and column name and
type; if it differs from the cached one, the catalog is rebuilt. Adding a table in
pgAdmin is therefore picked up on the next question with no manual refresh.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.models import Catalog
from app.db.introspect import SYSTEM_SCHEMAS, introspect
from app.engine.connections import pool_for
from app.engine.target import DBTarget

log = logging.getLogger(__name__)

# Our own bookkeeping schema is never shown to the model.
EXCLUDED_SCHEMAS = ["t2s"]

_VERSION_SQL = """
SELECT md5(COALESCE(string_agg(
           n.nspname || '.' || c.relname || '.' || a.attname || ':' || a.atttypid::text,
           ',' ORDER BY n.nspname, c.relname, a.attnum), ''))
  FROM pg_attribute a
  JOIN pg_class c     ON c.oid = a.attrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE a.attnum > 0 AND NOT a.attisdropped
   AND c.relkind::text = ANY(%(kinds)s)
   AND {schema_filter}
"""

_mem: dict[str, tuple[str, Catalog]] = {}
_lock = threading.Lock()


def _cache_file(target: DBTarget) -> Path:
    return Path(get_settings().index_dir).parent / "cache" / f"catalog_{target.fingerprint()}.json"


def _kinds(target: DBTarget) -> list[str]:
    return ["r", "p", "m"] + (["v"] if target.include_views else [])


def live_version(target: DBTarget) -> str:
    if target.schemas:
        flt, params = "n.nspname = ANY(%(schemas)s)", {"schemas": target.schemas}
    else:
        flt = ("n.nspname <> ALL(%(exclude)s) AND n.nspname NOT LIKE 'pg\\_%%'")
        params = {"exclude": list(SYSTEM_SCHEMAS) + EXCLUDED_SCHEMAS}
    with pool_for(target).connection() as conn, conn.cursor() as cur:
        cur.execute(_VERSION_SQL.format(schema_filter=flt), {"kinds": _kinds(target), **params})
        return cur.fetchone()[0]


def get_catalog(target: DBTarget, refresh: bool = False,
                sample_values: bool = True) -> tuple[Catalog, dict]:
    """Returns (catalog, info). info says whether it came from cache and how long it took."""
    t0 = time.perf_counter()
    key = target.fingerprint()
    version = live_version(target)

    if not refresh:
        with _lock:
            hit = _mem.get(key)
        if hit and hit[0] == version:
            return hit[1], {"cached": "memory", "ms": _ms(t0), "version": version}

        f = _cache_file(target)
        if f.exists():
            try:
                blob = json.loads(f.read_text(encoding="utf-8"))
                if blob.get("version") == version:
                    cat = Catalog.model_validate(blob["catalog"])
                    with _lock:
                        _mem[key] = (version, cat)
                    return cat, {"cached": "disk", "ms": _ms(t0), "version": version}
            except Exception as exc:
                log.warning("catalog cache unreadable (%s); rebuilding", exc)

    cat = introspect(
        sample_values=sample_values,
        pool=pool_for(target),
        schemas=target.schemas,
        include_views=target.include_views,
        exclude_schemas=EXCLUDED_SCHEMAS,
    )
    with _lock:
        _mem[key] = (version, cat)
    try:
        f = _cache_file(target)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"version": version, "target": target.public(),
                                 "catalog": json.loads(cat.model_dump_json())}),
                     encoding="utf-8")
    except Exception as exc:  # a read-only disk must not break answering
        log.warning("could not write catalog cache: %s", exc)
    return cat, {"cached": False, "ms": _ms(t0), "version": version}


def describe(cat: Catalog) -> dict:
    """The schema as the UI's explorer shows it."""
    fk_by_col: dict[tuple[str, str], str] = {}
    for fk in cat.foreign_keys:
        for col, ref_col in zip(fk.columns, fk.ref_columns):
            fk_by_col[(fk.left, col)] = f"{fk.right}.{ref_col}"

    schemas: dict[str, list[dict]] = {}
    for t in sorted(cat.tables, key=lambda x: x.qualified):
        schemas.setdefault(t.schema_name, []).append({
            "name": t.name,
            "qualified": t.qualified,
            "rows": t.estimated_rows,
            "comment": t.comment,
            "columns": [
                {
                    "name": c.name,
                    "type": c.data_type.strip('"'),
                    "pk": c.is_primary_key,
                    "nullable": c.nullable,
                    "fk": fk_by_col.get((t.qualified, c.name)),
                    "samples": c.sample_values[:4],
                }
                for c in t.columns
            ],
        })
    return {
        "database": cat.database,
        "schema_version": cat.schema_version,
        "stats": {
            "schemas": len(schemas),
            "tables": len(cat.tables),
            "columns": sum(len(t.columns) for t in cat.tables),
            "foreign_keys": len(cat.foreign_keys),
            "estimated_rows": sum(t.estimated_rows for t in cat.tables),
        },
        "schemas": [{"name": k, "tables": v} for k, v in schemas.items()],
    }


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
