"""The v2 engine pieces that need no database and no model: schema shrinking,
grounded answers, and the Gemini client (against a fake HTTP transport)."""
from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from app.engine import answer as answer_step
from app.engine import prune
from app.llm import providers
from app.llm.providers import GeminiProvider, MockProvider, list_gemini_models, parse_spec


# ── schema shrink: parsing the model's reply ──────────────────────────────────
def test_parse_selection_reads_fenced_json():
    text = 'Sure!\n```json\n{"answerable": true, "tables": [{"table": "sales.salesorderheader", ' \
           '"columns": ["orderdate", "totaldue"], "reason": "orders"}]}\n```'
    ok, items = prune.parse_selection(text)
    assert ok is True
    assert items == [{"table": "sales.salesorderheader", "columns": ["orderdate", "totaldue"],
                      "reason": "orders"}]


def test_parse_selection_accepts_a_bare_list_and_comma_columns():
    ok, items = prune.parse_selection('[{"name": "product", "columns": "name, listprice"}, "salesorderdetail"]')
    assert ok is True
    assert items[0] == {"table": "product", "columns": ["name", "listprice"], "reason": ""}
    assert items[1]["table"] == "salesorderdetail"


def test_parse_selection_keeps_the_not_answerable_verdict():
    ok, items = prune.parse_selection('{"answerable": false, "tables": []}')
    assert ok is False and items == []


# ── schema shrink: validation against the real catalogue ──────────────────────
def test_validate_drops_made_up_tables_and_columns(catalog):
    raw = [{"table": "sales.invoices", "columns": ["total"], "reason": ""},
           {"table": "production.product", "columns": ["name", "colour"], "reason": ""}]
    sel, bad_t, bad_c, corrected = prune.validate(raw, catalog, prune_columns=True, max_tables=8)
    assert [s.table for s in sel] == ["production.product"]
    assert sel[0].columns == ["name"]
    assert bad_t == ["sales.invoices"]
    assert bad_c == ["production.product.colour"]
    assert corrected == {}


def test_validate_corrects_a_near_miss_and_resolves_bare_names(catalog):
    raw = [{"table": "sales.salesorderheadr", "columns": [], "reason": ""},
           {"table": "product", "columns": [], "reason": ""}]
    sel, bad_t, _, corrected = prune.validate(raw, catalog, prune_columns=True, max_tables=8)
    assert [s.table for s in sel] == ["sales.salesorderheader", "production.product"]
    assert corrected == {"sales.salesorderheadr": "sales.salesorderheader"}
    assert not bad_t
    # no usable column list → the whole table is kept
    assert len(sel[1].columns) == 4


def test_subschema_adds_bridge_table_and_join_keys(catalog):
    raw = [{"table": "production.product", "columns": ["name"], "reason": ""},
           {"table": "sales.salesorderheader", "columns": ["orderdate"], "reason": ""}]
    sel, *_ = prune.validate(raw, catalog, prune_columns=True, max_tables=8)
    sub = prune.build_subschema(catalog, sel)
    names = [t.qualified for t in sub.tables]
    assert "sales.salesorderdetail" in names, "the bridge table must be added so a join is possible"
    by = {t.qualified: [c.name for c in t.columns] for t in sub.tables}
    assert "productid" in by["production.product"]          # primary key kept
    assert "salesorderid" in by["sales.salesorderheader"]   # primary key kept
    assert {"salesorderid", "productid"} <= set(by["sales.salesorderdetail"])  # FK columns
    assert any(s.source == "bridge" for s in sel)


def test_shrink_end_to_end_with_a_scripted_model(catalog):
    llm = MockProvider(responses=[json.dumps({"answerable": True, "tables": [
        {"table": "production.product", "columns": ["name", "listprice"], "reason": "prices"}]})])
    pr = prune.shrink("What is the most expensive product?", catalog, llm)
    assert pr.strategy == "llm"
    assert [t.qualified for t in pr.sub.tables] == ["production.product"]
    s = pr.summary()
    assert s["tables_before"] == 3 and s["tables_after"] == 1
    assert s["columns_after"] == 3  # name, listprice + the primary key
    assert s["shrunk_schema_tokens"] < s["full_schema_tokens"]


def test_shrink_falls_back_to_keywords_on_garbage(catalog):
    pr = prune.shrink("total due of orders", catalog, MockProvider(responses=["I think orders"]))
    assert pr.strategy == "keyword-fallback"
    assert pr.sub.tables, "a garbage reply must not leave the SQL step with nothing"


def test_shrink_respects_not_answerable(catalog):
    pr = prune.shrink("weather tomorrow?", catalog,
                      MockProvider(responses=['{"answerable": false, "tables": []}']))
    assert pr.answerable is False and not pr.sub.tables


# ── grounded answers ──────────────────────────────────────────────────────────
ROWS = [{"name": "Bikes", "revenue": Decimal("476256072.0595")}]


def test_answer_accepts_numbers_that_are_in_the_rows():
    out = answer_step.answer("q", "SELECT 1", ROWS, False,
                             MockProvider(responses=["Bikes earned 476,256,072.06 in revenue."]))
    assert out["grounded"] is True
    assert out["answer"].startswith("Bikes")


def test_answer_rejects_invented_numbers():
    out = answer_step.answer("q", "SELECT 1", ROWS, False,
                             MockProvider(responses=["Bikes earned about 500 million, 12% more."]))
    assert out["grounded"] is False
    assert out["answer"] is None and out["fell_back_to_table"] is True
    assert out["ungrounded_numbers"]


def test_answer_with_no_rows_needs_no_model():
    out = answer_step.answer("q", "SELECT 1", [], False, MockProvider(responses=[]))
    assert out["llm_used"] is False and out["grounded"] is True


# ── model specs ───────────────────────────────────────────────────────────────
def test_parse_spec_keeps_tags_after_the_provider():
    assert parse_spec("ollama:qwen2.5-coder:7b") == ("ollama", "qwen2.5-coder:7b")
    assert parse_spec("gemini:gemini-3.5-flash-lite") == ("gemini", "gemini-3.5-flash-lite")


# ── Gemini, against a fake transport (no network, no key used) ───────────────
def _gemini(handler) -> GeminiProvider:
    return GeminiProvider("test-key", "gemini-3.5-flash-lite", 0.0, 30,
                          transport=httpx.MockTransport(handler))


def test_gemini_sends_system_prompt_and_json_mode_and_skips_thoughts():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["key"] = req.headers.get("x-goog-api-key")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": "thinking...", "thought": True}, {"text": '{"tables": []}'}]}}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 4}})

    r = _gemini(handler).complete("SYS", "USER", json_mode=True)
    assert r.text == '{"tables": []}'
    assert seen["url"].endswith("/models/gemini-3.5-flash-lite:generateContent")
    assert seen["key"] == "test-key"
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert r.meta["prompt_tokens"] == 12


def test_gemini_retries_after_rate_limit(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    assert _gemini(handler).complete("s", "u").text == "ok"
    assert calls["n"] == 2


def test_gemini_reports_a_bad_key_clearly():
    def handler(req):
        return httpx.Response(400, json={"error": {"message": "API key not valid.", "status": "INVALID_ARGUMENT"}})

    with pytest.raises(RuntimeError, match="API key not valid"):
        _gemini(handler).complete("s", "u")


def test_list_gemini_models_keeps_only_text_generation_models():
    def handler(req):
        return httpx.Response(200, json={"models": [
            {"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-3.5-flash-lite-preview-tts", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/imagen-4", "supportedGenerationMethods": ["predict"]},
        ]})

    assert list_gemini_models("k", transport=httpx.MockTransport(handler)) == ["gemini-3.5-flash-lite"]


# ── gate hints and identifier case (real failures seen with llama3.1 and BIRD) ─
def test_l4_hint_for_reserved_word_alias():
    from app.pipeline.gate.l4_plan import planner_hint
    assert "reserved word" in planner_hint('syntax error at or near "is"', "")


def test_l4_hint_translates_sqlite_functions():
    from app.pipeline.gate.l4_plan import planner_hint
    assert "to_char" in planner_hint("function strftime(unknown, text) does not exist", "")


def test_l2_catches_unquoted_mixed_case_names():
    from app.core.models import Catalog, Column, Table
    from app.pipeline.gate import l2_bind
    t = Table(schema_name="public", name="frpm", primary_key=["CDSCode"], columns=[
        Column(schema_name="public", table_name="frpm", name="CDSCode", data_type="text"),
        Column(schema_name="public", table_name="frpm", name="County", data_type="text")])
    cat = Catalog(database="b", tables=[t], foreign_keys=[], schema_version="v")
    bad = l2_bind.check("SELECT CDSCode FROM public.frpm", cat)
    assert not bad.passed and '"CDSCode"' in bad.repair_hint
    assert l2_bind.check('SELECT "CDSCode" FROM public.frpm', cat).passed


def test_prompts_show_names_quoted_when_postgres_needs_it():
    from app.core.ident import q
    assert q("CDSCode") == '"CDSCode"' and q("Free Meal Count (K-12)") == '"Free Meal Count (K-12)"'
    assert q("county") == "county" and q("order") == '"order"'
