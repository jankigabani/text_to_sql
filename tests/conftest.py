from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.models import Catalog, Column, ForeignKey, Table   


def _col(schema: str, table: str, name: str, dtype: str = "integer", pk: bool = False) -> Column:
    return Column(schema_name=schema, table_name=table, name=name, data_type=dtype,
                  is_primary_key=pk)


@pytest.fixture
def catalog() -> Catalog:
    """A tiny two-schema catalog. Enough to exercise L2 without a live database."""
    product = Table(
        schema_name="production", name="product", estimated_rows=504,
        primary_key=["productid"],
        columns=[
            _col("production", "product", "productid", "integer", True),
            _col("production", "product", "name", "character varying(50)"),
            _col("production", "product", "listprice", "numeric(19,4)"),
            _col("production", "product", "productsubcategoryid"),
        ],
    )
    orders = Table(
        schema_name="sales", name="salesorderheader", estimated_rows=31465,
        primary_key=["salesorderid"],
        columns=[
            _col("sales", "salesorderheader", "salesorderid", "integer", True),
            _col("sales", "salesorderheader", "customerid"),
            _col("sales", "salesorderheader", "orderdate", "timestamp without time zone"),
            _col("sales", "salesorderheader", "totaldue", "numeric(19,4)"),
            _col("sales", "salesorderheader", "status", "smallint"),
        ],
    )
    detail = Table(
        schema_name="sales", name="salesorderdetail", estimated_rows=121317,
        primary_key=["salesorderdetailid"],
        columns=[
            _col("sales", "salesorderdetail", "salesorderdetailid", "integer", True),
            _col("sales", "salesorderdetail", "salesorderid"),
            _col("sales", "salesorderdetail", "productid"),
            _col("sales", "salesorderdetail", "orderqty", "smallint"),
            _col("sales", "salesorderdetail", "unitprice", "numeric(19,4)"),
        ],
    )
    fks = [
        ForeignKey(schema_name="sales", table_name="salesorderdetail",
                   columns=["salesorderid"], ref_schema="sales",
                   ref_table="salesorderheader", ref_columns=["salesorderid"]),
        ForeignKey(schema_name="sales", table_name="salesorderdetail",
                   columns=["productid"], ref_schema="production",
                   ref_table="product", ref_columns=["productid"]),
    ]
    return Catalog(database="test", tables=[product, orders, detail], foreign_keys=fks,
                   schema_version="testfixture")


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch, tmp_path):
    """Tests never call a real model or a real database unless they ask for one."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBED_PROVIDER", "hashing")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("GATE_DISABLED_LAYERS", "")
    from app.core.config import reload_settings
    reload_settings()
    yield
    reload_settings()


def pytest_configure(config):
    config.addinivalue_line("markers", "db: needs a live PostgreSQL from .env")


def live_db_available() -> bool:
    if os.environ.get("T2S_SKIP_DB_TESTS"):
        return False
    try:
        import psycopg

        from app.core.config import get_settings
        with psycopg.connect(get_settings().dsn, connect_timeout=4):
            return True
    except Exception:
        return False
