"""L4, L5 and the full pipeline — these need the live database from .env.

Skipped automatically when no database is reachable, so `pytest` still passes on a laptop
with PostgreSQL switched off.
"""
from __future__ import annotations

import pytest

from app.core.models import AskRequest
from app.db import introspect
from app.llm.providers import MockProvider
from app.pipeline.gate import l4_plan, l5_sniff, run_gate, score_confidence
from app.pipeline.orchestrator import Pipeline
from tests.conftest import live_db_available

pytestmark = pytest.mark.skipif(not live_db_available(), reason="no live PostgreSQL")


@pytest.fixture(scope="module")
def cat():
    return introspect.get_catalog()


def sql_block(sql: str) -> str:
    return f"REASONING: test\nSQL:\n```sql\n{sql}\n```"


# ── L4 ────────────────────────────────────────────────────────────────────────
def test_l4_accepts_a_cheap_query():
    r = l4_plan.check("SELECT count(*) FROM production.product")
    assert r.passed and r.details["total_cost"] > 0


def test_l4_rejects_a_cartesian_product():
    r = l4_plan.check(
        "SELECT * FROM warehouse.fact_sales f, production.product p"
    )
    assert not r.passed
    assert "cartesian" in r.message or "cost" in r.message


def test_l4_reports_planner_errors_as_repair_hints():
    r = l4_plan.check("SELECT productid FROM production.product GROUP BY name")
    assert not r.passed and r.details.get("planner_error")


def test_l4_cost_ceiling_is_enforced():
    cheap = "SELECT count(*) FROM production.product"
    assert not l4_plan.check(cheap, cost_ceiling=0.0001).passed


# ── L5 ────────────────────────────────────────────────────────────────────────
def test_l5_returns_rows():
    r = l5_sniff.check("SELECT name FROM production.product LIMIT 3", "list some products")
    assert r.passed and r.details["row_count"] == 3


def test_l5_warns_on_an_empty_result_for_a_which_question():
    r = l5_sniff.check(
        "SELECT name FROM production.product WHERE name = 'no such product ever'",
        "which product costs the most?",
    )
    assert r.passed and r.warning


def test_l5_warns_on_a_single_null():
    r = l5_sniff.check(
        "SELECT sum(listprice) FROM production.product WHERE productid < 0",
        "what is the total price?",
    )
    assert r.passed and r.warning


def test_l5_reports_execution_errors():
    r = l5_sniff.check("SELECT 1/0", "what is one divided by zero?")
    assert not r.passed


# ── gate ordering and ablation ────────────────────────────────────────────────
def test_gate_stops_at_the_first_failure(cat):
    report = run_gate("DELETE FROM production.product", cat, "delete everything")
    assert not report.passed
    # L1 passes (it parses), L2 passes, L3 blocks; L4/L5 must never have run.
    ran = [c.layer.value for c in report.checks]
    assert "l4" not in ran and "l5" not in ran


def test_disabling_a_layer_lets_bad_sql_through(cat):
    """This is the ablation mechanism the evaluation plan relies on."""
    bad = "SELECT nonexistent_column FROM production.product"
    assert not run_gate(bad, cat, "x").passed
    report = run_gate(bad, cat, "x", disabled={"l2", "l4", "l5"})
    assert report.passed  # only L1 and L3 remain, and both are happy


def test_confidence_drops_after_a_repair(cat):
    good = "SELECT count(*) AS n FROM production.product"
    first = run_gate(good, cat, "how many products?")
    c1 = score_confidence(first, retrieval_margin=1.0, attempts=1)
    c2 = score_confidence(first, retrieval_margin=1.0, attempts=3)
    assert c1 > c2


# ── end to end ────────────────────────────────────────────────────────────────
def test_pipeline_answers_with_a_mock_model(cat):
    llm = MockProvider(responses=[
        sql_block("SELECT count(*) AS products FROM production.product"),
        "There are 504 products.",
    ])
    p = Pipeline(catalog=cat, llm=llm)
    resp = p.ask(AskRequest(question="How many products are there?"))
    assert resp.status == "answered"
    assert resp.sql and resp.confidence > 0
    assert resp.row_count == 1


def test_pipeline_repairs_a_hallucinated_column_then_succeeds(cat):
    llm = MockProvider(responses=[
        sql_block("SELECT made_up_column FROM production.product"),
        sql_block("SELECT count(*) AS n FROM production.product"),
        "There are 504 products.",
    ])
    p = Pipeline(catalog=cat, llm=llm)
    resp = p.ask(AskRequest(question="How many products are there?"))
    assert resp.status == "answered"
    assert len(resp.attempts) == 2
    assert resp.attempts[1].repair_hint  # the model was told what was wrong


def test_pipeline_blocks_a_destructive_request(cat):
    llm = MockProvider(responses=[sql_block("DELETE FROM production.product")])
    p = Pipeline(catalog=cat, llm=llm)
    resp = p.ask(AskRequest(question="Delete all the products"))
    assert resp.status == "blocked"
    assert resp.sql is None


def test_pipeline_abstains_when_it_cannot_be_repaired(cat):
    llm = MockProvider(responses=[sql_block("SELECT nope FROM production.product")] * 4)
    p = Pipeline(catalog=cat, llm=llm)
    resp = p.ask(AskRequest(question="What is the nope of each product?"))
    assert resp.status == "abstained"
    assert resp.abstain_reason


def test_pipeline_abstains_when_the_model_declines(cat):
    llm = MockProvider(responses=["REASONING: nothing here\nNOT_ANSWERABLE"])
    p = Pipeline(catalog=cat, llm=llm)
    resp = p.ask(AskRequest(question="What is the weather in Thunder Bay?"))
    assert resp.status == "abstained"


def test_baseline_a_runs_without_the_gate(cat):
    llm = MockProvider(responses=[
        sql_block("SELECT count(*) AS n FROM production.product"),
        "There are 504 products.",
    ])
    p = Pipeline(catalog=cat, llm=llm)
    resp = p.ask(AskRequest(question="How many products?", arm="baseline_a"))
    assert resp.status == "answered"
    assert all(a.gate is None for a in resp.attempts)


def test_l4_accepts_a_join_pushed_into_an_index_scan():
    """The join predicate sits in the inner Index Cond, not on the Nested Loop."""
    r = l4_plan.check(
        "SELECT DISTINCT p.name FROM warehouse.inventory_snapshot i "
        "JOIN production.product p ON p.productid = i.product_id "
        "WHERE i.reorder_flag AND i.taken_on = (SELECT max(taken_on) FROM warehouse.inventory_snapshot)")
    assert r.passed, r.message


def test_l4_accepts_a_cross_join_with_a_single_row():
    r = l4_plan.check("SELECT p.name, t.n FROM production.product p "
                      "CROSS JOIN (SELECT count(*) AS n FROM sales.store) t")
    assert r.passed, r.message
