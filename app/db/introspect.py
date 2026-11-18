"""Stage 1 — introspection.

Everything the pipeline knows about the database comes from `pg_catalog`, never from a
hand-written description. That matters for the research claim: the system is given no
curated knowledge about the schema that a real deployment would not have.

Also collects, per column, a few representative values, so the LLM learns that a status
column holds 'A'/'I' rather than 'active'/'inactive' — one of the most common causes of
wrong-but-executable SQL.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from psycopg.rows import dict_row

from app.core.config import get_settings
from app.core.models import Catalog, Column, ForeignKey, Table
from app.db.pool import owner_pool

log = logging.getLogger(__name__)

_TABLES_SQL = """
SELECT n.nspname               AS schema_name,
       c.relname               AS table_name,
       obj_description(c.oid)  AS comment,
       GREATEST(c.reltuples, 0)::bigint AS estimated_rows
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE c.relkind::text = ANY(%(kinds)s)
   AND n.nspname = ANY(%(schemas)s)
 ORDER BY 1, 2;
"""

_COLUMNS_SQL = """
SELECT n.nspname AS schema_name,
       c.relname AS table_name,
       a.attname AS name,
       format_type(a.atttypid, a.atttypmod) AS data_type,
       NOT a.attnotnull AS nullable,
       col_description(c.oid, a.attnum) AS comment,
       s.n_distinct,
       COALESCE(pk.is_pk, false) AS is_primary_key
  FROM pg_attribute a
  JOIN pg_class c      ON c.oid = a.attrelid
  JOIN pg_namespace n  ON n.oid = c.relnamespace
  LEFT JOIN pg_stats s ON s.schemaname = n.nspname AND s.tablename = c.relname
                      AND s.attname = a.attname
  LEFT JOIN (
        SELECT con.conrelid, unnest(con.conkey) AS attnum, true AS is_pk
          FROM pg_constraint con WHERE con.contype = 'p'
  ) pk ON pk.conrelid = c.oid AND pk.attnum = a.attnum
 WHERE a.attnum > 0 AND NOT a.attisdropped
   AND c.relkind::text = ANY(%(kinds)s)
   AND n.nspname = ANY(%(schemas)s)
 ORDER BY n.nspname, c.relname, a.attnum;
"""

_FK_SQL = """
SELECT ns.nspname AS schema_name,
       cl.relname AS table_name,
       ARRAY(SELECT a.attname FROM pg_attribute a
              WHERE a.attrelid = con.conrelid AND a.attnum = ANY(con.conkey)
              ORDER BY array_position(con.conkey, a.attnum)) AS columns,
       fns.nspname AS ref_schema,
       fcl.relname AS ref_table,
       ARRAY(SELECT a.attname FROM pg_attribute a
              WHERE a.attrelid = con.confrelid AND a.attnum = ANY(con.confkey)
              ORDER BY array_position(con.confkey, a.attnum)) AS ref_columns
  FROM pg_constraint con
  JOIN pg_class cl       ON cl.oid = con.conrelid
  JOIN pg_namespace ns   ON ns.oid = cl.relnamespace
  JOIN pg_class fcl      ON fcl.oid = con.confrelid
  JOIN pg_namespace fns  ON fns.oid = fcl.relnamespace
 WHERE con.contype = 'f' AND ns.nspname = ANY(%(schemas)s)
 ORDER BY 1, 2;
"""

_PK_SQL = """
SELECT ns.nspname AS schema_name, cl.relname AS table_name,
       ARRAY(SELECT a.attname FROM pg_attribute a
              WHERE a.attrelid = con.conrelid AND a.attnum = ANY(con.conkey)
              ORDER BY array_position(con.conkey, a.attnum)) AS columns
  FROM pg_constraint con
  JOIN pg_class cl     ON cl.oid = con.conrelid
  JOIN pg_namespace ns ON ns.oid = cl.relnamespace
 WHERE con.contype = 'p' AND ns.nspname = ANY(%(schemas)s);
"""

_SAMPLE_SQL = """
SELECT DISTINCT {col}::text AS v
  FROM {tbl} TABLESAMPLE SYSTEM (%(pct)s)
 WHERE {col} IS NOT NULL
 LIMIT %(n)s;
"""
_SAMPLE_FALLBACK_SQL = """
SELECT DISTINCT {col}::text AS v FROM (SELECT {col} FROM {tbl} LIMIT 5000) z
 WHERE {col} IS NOT NULL LIMIT %(n)s;
"""

_SAMPLEABLE = ("char", "text", "bool", "enum", "date", "timestamp", "uuid", "name")


SYSTEM_SCHEMAS = ("pg_catalog", "information_schema", "pg_toast")

_SCHEMAS_SQL = """
SELECT nspname FROM pg_namespace
 WHERE nspname <> ALL(%(exclude)s)
   AND nspname NOT LIKE 'pg\\_temp%%' AND nspname NOT LIKE 'pg\\_toast%%'
 ORDER BY 1;
"""


def introspect(
    sample_values: bool = True,
    pool=None,
    schemas: list[str] | None = None,
    database: str | None = None,
    include_views: bool = True,
    exclude_schemas: list[str] | None = None,
) -> Catalog:
    """Read the catalog of any database.

    With no arguments it reads the database from .env (the original behaviour). The v2
    engine passes its own pool, so the same code introspects whatever database the user
    connects to from the UI. `schemas=None` with a pool means "every non-system schema".
    """
    s = get_settings()
    kinds = ["r", "p", "m"] + (["v"] if include_views else [])
    use_pool = pool if pool is not None else owner_pool()
    with use_pool.connection() as conn:
        # row_factory goes on the cursor, never on the connection: these connections are
        # pooled and reused by code that expects plain tuples.
        with conn.cursor(row_factory=dict_row) as cur:
            if schemas is None:
                if pool is None:
                    schemas = s.schema_allowlist
                else:
                    excl = list(SYSTEM_SCHEMAS) + list(exclude_schemas or [])
                    cur.execute(_SCHEMAS_SQL, {"exclude": excl})
                    schemas = [r["nspname"] for r in cur.fetchall()]
            if database is None:
                cur.execute("SELECT current_database() AS db")
                database = cur.fetchone()["db"]
            cur.execute(_TABLES_SQL, {"schemas": schemas, "kinds": kinds})
            tables = {
                f"{r['schema_name']}.{r['table_name']}": Table(
                    schema_name=r["schema_name"], name=r["table_name"],
                    comment=r["comment"], estimated_rows=r["estimated_rows"],
                )
                for r in cur.fetchall()
            }

            cur.execute(_PK_SQL, {"schemas": schemas})
            for r in cur.fetchall():
                t = tables.get(f"{r['schema_name']}.{r['table_name']}")
                if t:
                    t.primary_key = list(r["columns"])

            cur.execute(_COLUMNS_SQL, {"schemas": schemas, "kinds": kinds})
            for r in cur.fetchall():
                t = tables.get(f"{r['schema_name']}.{r['table_name']}")
                if t:
                    t.columns.append(Column(
                        schema_name=r["schema_name"], table_name=r["table_name"],
                        name=r["name"], data_type=r["data_type"], nullable=r["nullable"],
                        is_primary_key=r["is_primary_key"], comment=r["comment"],
                        n_distinct=r["n_distinct"],
                    ))

            cur.execute(_FK_SQL, {"schemas": schemas})
            fks = [
                ForeignKey(
                    schema_name=r["schema_name"], table_name=r["table_name"],
                    columns=list(r["columns"]), ref_schema=r["ref_schema"],
                    ref_table=r["ref_table"], ref_columns=list(r["ref_columns"]),
                )
                for r in cur.fetchall()
            ]

        cat = Catalog(database=database, tables=list(tables.values()), foreign_keys=fks)
        if sample_values:
            _add_samples(conn, cat, s.retrieve_sample_values)

    cat.schema_version = _fingerprint(cat)
    log.info(
        "introspected %d tables / %d columns / %d FKs",
        len(cat.tables), sum(len(t.columns) for t in cat.tables), len(cat.foreign_keys),
    )
    return cat


def _add_samples(conn, cat: Catalog, n: int) -> None:
    for t in cat.tables:
        pct = 1 if t.estimated_rows > 1_000_000 else 10
        for c in t.columns:
            if not any(k in c.data_type for k in _SAMPLEABLE):
                continue
            ident = f'"{c.schema_name}"."{c.table_name}"'
            col = f'"{c.name}"'
            for tmpl in (_SAMPLE_SQL, _SAMPLE_FALLBACK_SQL):
                try:
                    with conn.cursor(row_factory=dict_row) as cur:
                        cur.execute(tmpl.format(col=col, tbl=ident), {"pct": pct, "n": n})
                        c.sample_values = [str(r["v"])[:60] for r in cur.fetchall()]
                    break
                except Exception:  # views, permissions, unsupported sampling
                    conn.rollback()
                    continue


def _fingerprint(cat: Catalog) -> str:
    import hashlib

    h = hashlib.sha256()
    for t in sorted(cat.tables, key=lambda x: x.qualified):
        h.update(t.qualified.encode())
        for c in t.columns:
            h.update(f"{c.name}:{c.data_type}".encode())
    return h.hexdigest()[:16]

def cache_path() -> Path:
    return Path(get_settings().index_dir) / "catalog.json"


def save(cat: Catalog) -> Path:
    p = cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(cat.model_dump_json(indent=2), encoding="utf-8")
    return p


def load_cached() -> Catalog | None:
    p = cache_path()
    if not p.exists():
        return None
    try:
        return Catalog.model_validate(json.loads(p.read_text(encoding="utf-8")))
    except Exception as exc:  # corrupt cache must never be fatal
        log.warning("catalog cache unreadable (%s); re-introspecting", exc)
        return None


def get_catalog(refresh: bool = False) -> Catalog:
    if not refresh:
        cached = load_cached()
        if cached:
            return cached
    cat = introspect()
    save(cat)
    return cat
