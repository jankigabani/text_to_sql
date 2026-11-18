# Text-to-SQL with a Verification Gate

**COMP 5112 — Research Methodology in Computer Science · Group 6**
Janki Gabani · Gautam Verma · Lakehead University

An AI-powered database question answering system over **PostgreSQL**. You connect it to
any PostgreSQL database, ask a question in plain English, and watch it work: it reads the
schema, lets an LLM shrink that schema to the tables the question needs, asks the LLM for
SQL, **checks that SQL with five deterministic checks before running it**, runs it
read-only, and turns the rows into a sentence whose every number is checked against the
result — or says *"I'm not sure"* instead of guessing.

The five checks and the abstention behaviour are the research contribution. Everything else is
careful engineering around them.

---

## Run it (Windows, PostgreSQL 18 + Ollama already installed)

```powershell
cd D:\Janki\researchmethodology
.\venv\Scripts\Activate.ps1           # first time: py -m venv venv
pip install -r requirements.txt

python scripts\setup_all.py           # loads the whole dataset into PostgreSQL (once)
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/** for the web UI, or **/docs** for the API.

`setup_all.py` checks the server, loads AdventureWorks (68 tables, 5 schemas), adds a
`warehouse` schema with ~10 million rows, creates the read-only role and builds the v1
index. Rerunning it skips whatever is already done. Then, in pgAdmin: *Databases ▸
right-click ▸ Refresh ▸ adventureworks ▸ Schemas*.

Full instructions, including the password reset, are in `SETUP_WINDOWS.md`.

## The pipeline (what the web UI shows, live)

```
question + connection details (any PostgreSQL database)
   │
[1] connect      read-only session (default_transaction_read_only, statement_timeout)
[2] schema       pg_catalog → schemas, tables, columns, types, PK/FK   (cached, auto-invalidated)
[3] shrink       LLM picks only the relevant tables + columns → validated against the catalogue,
   │             made-up names dropped, PK/FK join columns and bridge tables added
[4] write SQL    LLM writes one query from the shrunk schema
[5] VERIFY       L1 parse → L3 read-only & safe → L2 names exist → L4 plan → L5 sniff
   │                 harmful (DELETE/DROP/UPDATE/…) → BLOCKED, never repaired
   │                 other failure → reason fed back to [4] (max 2) → else ABSTAIN
[6] run          read-only, timeout, row limit
[7] answer       LLM phrases the rows; any number not in the rows → sentence rejected, table shown
   │
answer + confidence + SQL + rows + every intermediate step
```

See `ARCHITECTURE.md` for the design decisions and how each stage maps to RQ1–RQ4.

## Models

Pick one per question in the UI (or a different one per step):

| spec | where | notes |
|---|---|---|
| `ollama:qwen2.5-coder:7b` | your laptop | code-tuned; usually the strongest local SQL writer |
| `ollama:llama3.1:latest` | your laptop | general model; good at the answer sentence |
| `gemini:gemini-3.5-flash-lite` | Google, free tier | needs `GEMINI_API_KEY` in `.env`; rate-limited per minute |
| `mock:demo` | nowhere | scripted answers for the built-in examples only — for testing, not results |

## API

| endpoint | what it does |
|---|---|
| `GET  /api/health` | database, Ollama and Gemini status |
| `GET  /api/models` | the models that will actually answer |
| `POST /api/connections/test` | check connection details (any host/database/user) |
| `POST /api/schema` | every schema, table, column, type and key |
| `POST /api/ask` | the seven steps; returns the answer and the full trace |
| `POST /api/ask/stream` | the same, streamed step by step (Server-Sent Events) |
| `GET  /api/examples`, `/api/runs`, `POST /api/feedback` | UI helpers |
| `/v1/...` | the original single-database endpoints used by the evaluation harness |

```json
POST /api/ask
{
  "question": "Which product category produced the highest revenue in the warehouse data?",
  "connection": { "host": "localhost", "port": 5432, "database": "adventureworks",
                  "user": "postgres", "password": "…" },
  "model": "ollama:qwen2.5-coder:7b"
}
```

Use `"connection": {"use_default": true}` to use the database in `.env` without sending a
password.

## Running the experiments

```powershell
python -m eval.runner --suite adventureworks --arm all        # the four arms
python -m eval.runner --suite adversarial    --arm ours       # unsafe / unanswerable
python -m eval.runner --suite adventureworks --arm ours --ablate l4
python -m eval.report                                         # results table for the paper
```

## Layout

| Path | What it is |
|---|---|
| `frontend/` | the web UI — plain HTML/CSS/JS, served by FastAPI at `/`, no build step |
| `app/engine/` | the v2 engine: per-request connections, schema cache, LLM schema shrink, grounded answer |
| `app/api/v2.py` | the `/api` endpoints the UI uses (incl. SSE streaming) |
| `app/pipeline/gate/` | **the contribution**: L1–L5, repair loop, confidence scoring |
| `app/llm/` | Ollama, Gemini, OpenAI, Anthropic, scripted demo |
| `app/db/`, `app/retrieval/`, `app/pipeline/` | introspection, embedding retrieval, v1 pipeline |
| `scripts/` | `setup_all.py` and the individual loading steps |
| `eval/` | AdventureWorks, adversarial and BIRD suites, metrics, ablation runner |
| `tests/` | pytest — 79 tests; the live ones skip themselves without PostgreSQL |
