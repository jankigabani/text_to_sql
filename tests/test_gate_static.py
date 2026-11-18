"""L1, L2 and L3 — the layers that need no database.

These are the tests that matter most for the paper: each one is a concrete wrong-SQL case
that the gate catches and that an LLM self-check routinely misses.
"""
from __future__ import annotations

import pytest

from app.pipeline.gate import l1_parse, l2_bind, l3_policy

GOOD = """
SELECT p.name, SUM(d.orderqty) AS units
  FROM sales.salesorderdetail d
  JOIN production.product p ON p.productid = d.productid
 GROUP BY p.name
 ORDER BY units DESC
 LIMIT 5
"""


# ── L1 ────────────────────────────────────────────────────────────────────────
def test_l1_accepts_a_normal_query():
    r = l1_parse.check(GOOD)
    assert r.passed and r.details["statement_type"] == "Select"


def test_l1_accepts_a_cte():
    r = l1_parse.check("WITH x AS (SELECT 1 AS a) SELECT a FROM x")
    assert r.passed


@pytest.mark.parametrize("sql", ["", "   ", "\n"])
def test_l1_rejects_empty_output(sql):
    r = l1_parse.check(sql)
    assert not r.passed and "no SQL" in r.message


def test_l1_rejects_truncated_sql():
    r = l1_parse.check("SELECT name, FROM production.product WHERE")
    assert not r.passed


def test_l1_rejects_two_statements():
    r = l1_parse.check("SELECT 1; SELECT 2")
    assert not r.passed and r.details["statement_count"] == 2


def test_l1_rejects_a_non_query_statement():
    r = l1_parse.check("UPDATE production.product SET listprice = 0")
    assert not r.passed


def test_l1_repair_hint_is_actionable():
    r = l1_parse.check("SELECT * FROM (")
    assert r.repair_hint and "parse" in r.repair_hint.lower()


# ── L2 ────────────────────────────────────────────────────────────────────────
def test_l2_accepts_real_identifiers(catalog):
    r = l2_bind.check(GOOD, catalog)
    assert r.passed
    assert "sales.salesorderdetail" in r.details["tables"]


def test_l2_catches_a_hallucinated_table(catalog):
    r = l2_bind.check("SELECT * FROM sales.orders", catalog)
    assert not r.passed
    assert "sales.orders" in r.details["unknown_tables"]


def test_l2_catches_a_hallucinated_column_and_suggests_the_real_one(catalog):
    r = l2_bind.check(
        "SELECT h.total_due FROM sales.salesorderheader h", catalog
    )
    assert not r.passed
    assert "totaldue" in r.repair_hint


def test_l2_allows_cte_names(catalog):
    sql = """
    WITH recent AS (SELECT salesorderid, totaldue FROM sales.salesorderheader)
    SELECT salesorderid FROM recent
    """
    assert l2_bind.check(sql, catalog).passed


def test_l2_allows_derived_table_aliases(catalog):
    sql = """
    SELECT t.n FROM (SELECT count(*) AS n FROM production.product) AS t
    """
    assert l2_bind.check(sql, catalog).passed


def test_l2_is_case_insensitive(catalog):
    assert l2_bind.check("SELECT ProductID FROM Production.Product", catalog).passed


# ── L3 ────────────────────────────────────────────────────────────────────────
def test_l3_accepts_a_read(catalog):
    assert l3_policy.check(GOOD).passed


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM production.product",
        "DROP TABLE sales.salesorderheader",
        "UPDATE production.product SET listprice = 0",
        "INSERT INTO production.product (name) VALUES ('x')",
        "TRUNCATE TABLE sales.salesorderdetail",
        "ALTER TABLE production.product ADD COLUMN evil int",
        "CREATE TABLE evil (x int)",
        "GRANT ALL ON production.product TO PUBLIC",
    ],
)
def test_l3_blocks_every_write(sql):
    r = l3_policy.check(sql)
    assert not r.passed, sql


def test_l3_blocks_stacked_statement_injection():
    r = l3_policy.check("SELECT 1; DROP TABLE production.product")
    assert not r.passed


def test_l3_blocks_comment_terminated_injection():
    r = l3_policy.check("SELECT name FROM production.product -- ; DROP TABLE x\n; DELETE FROM production.product")
    assert not r.passed


def test_l3_blocks_select_into():
    r = l3_policy.check("SELECT * INTO evil FROM production.product")
    assert not r.passed


def test_l3_blocks_file_and_sleep_functions():
    assert not l3_policy.check("SELECT pg_sleep(30)").passed
    assert not l3_policy.check("SELECT pg_read_file('/etc/passwd')").passed


def test_l3_blocks_system_catalog_access():
    r = l3_policy.check("SELECT rolname, rolpassword FROM pg_catalog.pg_authid")
    assert not r.passed


def test_l3_allows_the_word_delete_inside_a_string_literal():
    """A product genuinely called 'delete from stock' must not be treated as an attack."""
    r = l3_policy.check("SELECT name FROM production.product WHERE name = 'delete from stock'")
    assert r.passed


def test_l3_allows_a_column_named_update_status():
    r = l3_policy.check("SELECT status FROM sales.salesorderheader WHERE status = 5")
    assert r.passed
