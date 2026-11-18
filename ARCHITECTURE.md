# Architecture

This document records *why* the system is built this way, so the paper and Presentation 2
can be written from it.

## 1. The claim

> A text-to-SQL system can be made meaningfully more trustworthy by putting a **layered,
> deterministic verification gate** between SQL generation and execution, and by using the
> gate's own telemetry as a **calibrated confidence score** that lets the system abstain.

Everything in the codebase exists either to make that claim testable or to make the
comparison fair.

## 2. Stage by stage

### Stage 1 — introspection (`app/db/introspect.py`)

Reads `pg_catalog` directly: tables, columns, types, nullability, primary keys, foreign
keys, comments, `pg_stats.n_distinct`, and up to four sampled values per text-ish column.

Two design decisions worth defending:

* **No curated schema description.** Everything is machine-read. A system that needs a
  human-written data dictionary cannot be deployed on a database nobody has documented,
  which is the realistic case.
* **Sampled values are part of the schema.** The single most common wrong-but-executable
  query comes from guessing a code value (`status = 'active'` when the column holds `'A'`).
  Sampling is bounded with `TABLESAMPLE SYSTEM` so it stays cheap on a 10M-row table.

The catalog is fingerprinted (`schema_version`) and cached; a schema change invalidates the
retrieval index automatically.

### Stage 2 — retrieval (`app/retrieval/`)

Answers **RQ1**. Three retrieval paths, unioned, then a foreign-key closure:

| path | what it recovers |
|---|---|
| table-first | tables whose name or purpose matches the question |
| column-first | tables whose *columns* match even when the table name is opaque |
| value lookup | the column that literally contains `'Thunder Bay'` |
| FK closure | bridge tables needed to join the chosen ones |

`retrieval_margin` — the score gap between the last table kept and the first dropped — is
recorded and fed into the confidence score. A narrow margin means retrieval was unsure,
which is precisely when the final answer deserves less trust.

Question expansion (`question.py`) is rule-based, not an LLM call: keywords, literals,
aggregation hints. It costs nothing and is deterministic, which keeps the retrieval
experiment reproducible.

Embeddings come from Ollama (`nomic-embed-text`) with a pure-Python hashed TF-IDF
fallback, so the system still runs with no model server. The fallback is weaker and the
results must say which one was used — `/health` and every result file record it.

### Stage 3 — generation (`app/pipeline/generate.py`)

The prompt contains only the retrieved sub-schema, the FK edges between those tables,
sampled values, and the question. The model is asked for `REASONING:` then `SQL:`; the
reasoning is kept for the error taxonomy and never shown as the answer. The model may
reply `NOT_ANSWERABLE`, which is treated as a legitimate abstention rather than a failure.

### Stage 4 — the verification gate (`app/pipeline/gate/`)

The contribution. Five checks, run in the order **L1 → L3 → L2 → L4 → L5**, fail-fast:

| layer | question | cost | catches |
|---|---|---|---|
| **L1 parse** | is it one valid PostgreSQL statement? | µs | truncation, stray prose, two statements, DML |
| **L3 policy** | is it a single harmless read? | µs | INSERT/UPDATE/DELETE/DDL, stacked statements, `pg_read_file`, `pg_sleep`, system catalogs |
| **L2 bind** | do all identifiers exist? | µs | hallucinated tables and columns (with a nearest-name hint) |
| **L4 plan** | what would it cost? | ms (EXPLAIN, no execution) | cartesian products, missing join predicates, planner-level type errors |
| **L5 sniff** | does the result look like an answer? | ms (LIMIT + timeout) | empty result, single NULL, whole-table answers |

L3 runs before L2 on purpose. `SELECT rolpassword FROM pg_authid` names a table that is not
in the application schema, so L2 would report "unknown table" and the repair loop would
politely ask the model to try again. Running the policy check first classifies it correctly
as an attack: refused, logged, never repaired.

L4 recognises a join whose predicate the planner pushed into a parameterised inner index
scan (`Index Cond: (productid = i.product_id)`), and ignores a cross join with a one-row
side (an aggregate), so only genuine cartesian products are rejected.

Properties that matter for the write-up:

* **No layer calls a model.** The published notion of "self-correction" is one more LLM
  call with no ground truth; every check here is a rule or the database itself.
* **Checks are on the parse tree, not on the text.** `WHERE name = 'delete from stock'` is
  allowed; `-- ; DELETE FROM …` is not. A regex filter gets both wrong.
* **Failures are structured.** Each failure carries a `repair_hint` written for the model
  ("column orders.total does not exist; nearest match orders.total_amount"), not a stack
  trace. This is what makes the bounded repair loop converge in one or two attempts.
* **Write attempts are refused, not repaired.** L1 and L3 mark those `blocked`; the
  orchestrator never asks the model to try again.
* **Layers can be switched off individually**, which is exactly the per-layer ablation the
  evaluation plan promises.

Defence in depth: even with L3 disabled, stage 5 executes as a `SELECT`-only role inside a
read-only transaction. L3 exists so the attempt is *detected and logged*, which is what the
adversarial suite measures.

### Confidence (`app/pipeline/gate/__init__.py`)

Answers **RQ3**. A weighted combination of four signals, fixed in advance and never tuned
per question:

| signal | weight | rationale |
|---|---|---|
| all layers passed | 0.55 | the primary evidence |
| no repair was needed | 0.20 | a first-attempt query is more trustworthy |
| retrieval margin | 0.15 | unsure retrieval ⇒ unsure answer |
| result shape clean | 0.10 | empty/NULL results are suspicious |

Below `GATE_CONFIDENCE_THRESHOLD` the system abstains: it returns the draft SQL and says
what it could not confirm, instead of a number it cannot support.

### Stage 5 — execution (`app/pipeline/execute.py`)

Read-only role, `default_transaction_read_only = on`, `statement_timeout`, and a hard
`LIMIT` wrapped around the query. Returns rows plus provenance: the query, the tables
touched, the runtime.

### Stage 6 — synthesis (`app/pipeline/synthesize.py`)

The model receives the question and the result rows and nothing else. Then every numeral in
its sentence is checked against the result set, with a tolerance for legitimate rounding
and for years that appear inside dates. An ungrounded number fails the sentence and the
user gets a plain table. Execution accuracy stops at the result set; this step is where
a correct query can still produce a wrong answer, and we measure it.

## 3. The v2 engine and the web UI (`app/engine/`, `app/api/v2.py`, `frontend/`)

The pipeline above (v1) answers questions about the one database in `.env`, using an
embedding index for retrieval. The v2 engine is what the web UI runs. It makes two changes,
both driven by the demo requirement that *a user can point it at any PostgreSQL database
and ask*:

1. **Per-request connections.** Every `/api/ask` carries connection details (or
   `use_default`). `DBTarget` builds a libpq conninfo with
   `-c default_transaction_read_only=on -c statement_timeout=…`, so every session is
   read-only *whatever role it logs in as* — a user-supplied superuser cannot write either.
   Pools are kept per target in a small LRU (`CONNECTION_CACHE_SIZE`). Passwords are
   `SecretStr`, are never logged, and are hashed into the pool key rather than stored.
2. **LLM schema shrinking instead of an embedding index.** A new database has no index,
   so retrieval is done by the model itself:
   * the whole schema is rendered compactly (`schema.table (~rows): col type PK -> ref.col`);
     if it exceeds `PRUNE_MAX_PROMPT_TOKENS`, a keyword ranking plus FK neighbours narrows
     it first;
   * the model returns JSON `{answerable, tables:[{table, columns, reason}]}`;
   * the selection is **validated against the catalogue**: unknown tables and columns are
     dropped and counted (hallucination rate is reported per question), near-misses are
     corrected (`difflib`, cutoff 0.88), primary keys and FK join columns are added back,
     and bridge tables on the FK path between chosen tables are added;
   * `answerable: false` with no tables is an abstention before any SQL is written;
   * an unusable reply falls back to keyword ranking, and the result says so.

   The catalogue is cached in memory and on disk, keyed by a fingerprint of the target,
   and invalidated by an md5 over `pg_attribute` — a changed schema is re-read automatically.

The engine is a generator of stage events (`connect, schema, shrink, generate, verify,
execute, answer`); `/api/ask/stream` sends them as Server-Sent Events so the UI can show
each step as it happens, and `/api/ask` drains the same generator, so both give identical
results. The final answer step reuses the grounding check from v1, applied to both the raw
and the display-rounded rows.

Models are named `provider:model` per request — `ollama:qwen2.5-coder:7b`,
`ollama:llama3.1:latest`, `gemini:gemini-2.5-flash` — and can differ per step (e.g. Gemini
shrinks, qwen writes SQL, llama phrases the answer). `mock:demo` is a scripted stand-in for
tests and must never be reported as a model result.

`python -m eval.runner --engine v2 --model <spec> --arm all` evaluates the v2 engine with any
model; for v2 the arms are full-vs-shrunk schema × gate-off-vs-on.

## 4. The four arms

| arm | schema | verification | abstention | what it isolates |
|---|---|---|---|---|
| `baseline_a` | whole database | none | no | the value of retrieval (RQ1) |
| `baseline_b` | retrieved | none | no | the value of the gate (RQ2) |
| `baseline_c` | retrieved | LLM re-checks itself | no | the gate vs. the published approach |
| `ours` | retrieved | L1–L5 | yes | the full contribution |

Baseline C is the DIN-SQL-style configuration and is the arm the contribution must beat.
Per-layer ablations (`--ablate l4`) then attribute the gain to individual checks.

## 5. Metrics

| metric | research question |
|---|---|
| execution accuracy (EX), order-insensitive result match | RQ1, RQ2 |
| EX among questions it chose to answer | RQ3 |
| abstention-aware score with penalties c ∈ {0, 1, 10} | RQ3 |
| unsafe-query block rate on the adversarial suite | RQ3 / security gap |
| schema recall and false-positive rate | RQ1 |
| model calls, prompt tokens, ms per question | RQ4 |

The penalty-based score follows TrustSQL: a correct answer and a correct abstention both
score +1, an unnecessary abstention scores 0, and a confidently wrong answer scores −c. A
system that always answers can therefore score below zero, which is the behaviour the
proposal argues is missing from the literature.

## 6. Known limitations (state these in the paper)

* The v2 schema shrink sends table and column names (and row estimates) to whichever model
  is chosen. With a cloud model (Gemini) that metadata leaves the machine; the rows
  themselves only do in the final answer step.
* PostgreSQL only. L4 depends on `EXPLAIN (FORMAT JSON)` and the read-only role is a
  PostgreSQL feature; porting to another engine means rewriting those two layers.
* Single-turn. No dialogue, no follow-up questions.
* L2 cannot fully resolve unqualified columns inside correlated subqueries; it checks them
  against the union of tables in scope, which is sound but not complete.
* The warehouse data is synthetic. It provides volume, not realistic distributions, so the
  L4 cost ceiling is calibrated to this database and would need re-calibrating elsewhere.
* Cost figures are provider- and date-specific.
* Benchmark annotation error (CIDR 2026) bounds how precisely any EX number can be
  trusted, which is why a hand-checked slice is reported separately.
