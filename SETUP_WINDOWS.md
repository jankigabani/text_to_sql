# Setup on Windows (PostgreSQL already installed)

Everything below is run from `D:\Janki\researchmethodology` in PowerShell.

## 0. What you need

| thing | check |
|---|---|
| Python 3.10+ | `py -V` |
| PostgreSQL 14+ running | `services.msc` → `postgresql-x64-18` is *Running* |
| `psql` | not needed on PATH — the scripts find `D:\Janki\PostgreSQL\18\bin\psql.exe` themselves |
| Ollama | `ollama --version` |
| ~3 GB free disk (10M rows ≈ 1.5–2 GB) | |

If you want to type `psql` yourself, add it to PATH (adjust the version number):

```powershell
$env:Path += ";D:\Janki\PostgreSQL\18\bin"
# permanently:
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";D:\Janki\PostgreSQL\18\bin", "User")
```

### If you do not know the postgres password

The installer required one and there is no default. Reset it from an **administrator**
PowerShell:

```powershell
$f = "D:\Janki\PostgreSQL\18\data\pg_hba.conf"
Copy-Item $f "$f.bak" -Force
(Get-Content $f) -replace '^(host\s+all\s+all\s+(?:127\.0\.0\.1/32|::1/128)\s+)\S+', '${1}trust' |
    Set-Content $f -Encoding ascii
Restart-Service postgresql-x64-18
& "D:\Janki\PostgreSQL\18\bin\psql.exe" -U postgres -h 127.0.0.1 -c "ALTER USER postgres WITH PASSWORD 'YourNewPassword';"
Copy-Item "$f.bak" $f -Force
Restart-Service postgresql-x64-18
```

Edit only the `host all all` lines. The `host replication all` lines below them control
database-to-database copying, not logins, so changing those does nothing for the password.

Find your own paths with:

```powershell
(Get-CimInstance Win32_Service -Filter "Name like 'postgresql%'").PathName
```

## 1. Python environment

```powershell
cd D:\Janki\researchmethodology
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run the activate script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 2. Configuration

```powershell
copy .env.example .env
notepad .env
```

Set at least:

```ini
PGPASSWORD=<your postgres password>
PG_READONLY_PASSWORD=<pick anything>
PGDATABASE=adventureworks
DEFAULT_MODEL=ollama:qwen2.5-coder:7b
OLLAMA_NUM_CTX=16384
GEMINI_API_KEY=<optional, from https://aistudio.google.com/apikey>
GEMINI_MODEL=gemini-3.5-flash-lite
```

Secrets (passwords, the Gemini key) go in `.env` only. `.env` is git-ignored;
`.env.example` is not, so keep it as a template with placeholders.

Then:

```powershell
python scripts\00_check_postgres.py
```

Fix anything it marks `[FAIL]` before going on. Common ones:

| message | fix |
|---|---|
| `cannot connect: password authentication failed` | wrong `PGPASSWORD` in `.env`; see the password reset above |
| `database "adventureworks" does not exist` | expected before step 3; the check script now warns and continues |
| `cannot connect: ... refused` | the PostgreSQL service is not running |
| `no tables found` | step 3 has not run yet |
| `no planner statistics` | run `psql -d adventureworks -c "ANALYZE"` |

## 3. Load the whole dataset (one command)

```powershell
python scripts\setup_all.py
```

It runs every step below in order and skips whatever is already done, so it is safe to
rerun. Expect 5–10 minutes the first time; most of it is step 3b.

| step | what it does | on its own |
|---|---|---|
| 3a | downloads AdventureWorks (~18 MB) and loads **68 tables in 5 schemas** | `python scripts\01_load_adventureworks.py` |
| 3b | adds a `warehouse` schema with **~10 million rows** that reference the real tables by foreign key | `python scripts\02_amplify.py --target-rows 10000000` |
| 3c | creates the read-only role and proves it cannot write | `python scripts\03_create_readonly_role.py` |
| 3d | builds the embedding index used by the `/v1` endpoints | `python scripts\04_build_schema_index.py` |

Options: `--rows 1000000` for a smaller, faster load; `--reload` to drop and load
everything again; `--psql "D:\Janki\PostgreSQL\18\bin\psql.exe"` if psql is somewhere
unusual.

It ends with a table of schemas, tables and row counts.

### See it in pgAdmin

1. In the left tree: **Servers ▸ PostgreSQL 18 ▸ Databases** → right-click ▸ **Refresh**.
2. Open **adventureworks ▸ Schemas**: `humanresources`, `person`, `production`,
   `purchasing`, `sales`, `warehouse`.
3. Open any schema ▸ **Tables** ▸ right-click a table ▸ **View/Edit Data ▸ First 100 Rows**.
4. For the diagram: right-click **adventureworks ▸ ERD For Database**.

If pgAdmin had the database open while you used `--reload`, it may show an error once;
refresh again.

## 4. Models

```powershell
ollama list
```

You already have `qwen2.5-coder:7b`, `llama3.1:latest` and `nomic-embed-text`. Both chat
models appear in the web UI's model picker; pick per question, or a different one per step
under *Options*.

| model | strengths | notes |
|---|---|---|
| `qwen2.5-coder:7b` | trained on code — usually the more accurate SQL writer | ~6 GB RAM |
| `llama3.1:8b` | general model — fluent answer sentences | ~6 GB RAM; weaker at exact column names |
| `gemini-3.5-flash-lite` | strong and fast, free tier | needs `GEMINI_API_KEY`; ~10 requests/min, one question uses 3 |

If Ollama is "not reachable", start it from the Start menu (or run `ollama serve`).

## 5. Start the app

```powershell
uvicorn app.main:app --reload
```

* **http://127.0.0.1:8000/** — the web UI
* **http://127.0.0.1:8000/docs** — the API (`/api/...` for the UI, `/v1/...` for the
  evaluation harness; the old `POST /ask` is now `POST /v1/ask`)

### Demo script for the class (5 minutes)

1. Top bar: the three green dots (database, Ollama, Gemini).
2. Left: **Test connection**, then show the schema explorer — 6 schemas, ~70 tables; search
   "product".
3. Click an **Answerable** example, model `qwen2.5-coder:7b`, **Ask**. Walk the seven
   steps as they light up: the shrink bars (74 tables → 4), the picked tables lighting up in
   the explorer, the SQL, the five checks, the rows, the grounded answer.
4. **Harmful** tab → "Delete all the products…" → it stops at step 5, red, *blocked*.
   A real model is told to write only SELECTs, so it often refuses by itself (it then
   abstains at step 4) — itself a good point to make. To show the gate catching an actual
   DELETE every time, pick the model **scripted demo**: it always attempts the attack.
5. **Not in the data** tab → the weather question → it abstains at step 3, before any SQL.
6. Switch the model to `llama3.1` or Gemini and ask the same question — compare.
7. *Custom* connection → another database (e.g. `postgres`) → the schema explorer changes.
8. *Options* → untick **Run the verification gate**, then ask the harmful "keeps the
   database busy for ten minutes" question. Without the gate the query reaches the
   database and ties it up until the 15-second timeout kills it; with the gate it was
   refused in about a millisecond, before touching the database.

## 6. Run the experiments

```powershell
# v2 engine (what the UI runs), one model at a time:
python -m eval.runner --engine v2 --model ollama:qwen2.5-coder:7b --arm all
python -m eval.runner --engine v2 --model ollama:llama3.1:latest  --arm all
python -m eval.runner --engine v2 --model gemini:gemini-3.5-flash-lite --arm ours

# v1 pipeline (embedding retrieval):
python -m eval.runner --suite adventureworks --arm all
python -m eval.runner --suite adventureworks --arm ours --ablate l4
python -m eval.report
```

Results go to `results\`. `results\summary.csv` accumulates one row per run, and
`python -m eval.report --latex` prints the table for the IEEE paper.

### The benchmark for Presentation 2 (BIRD Mini-Dev + AdventureWorks)

```powershell
python scripts\run_benchmark.py
```

* Downloads BIRD Mini-Dev once (~1 GB) and copies the databases it needs into PostgreSQL
  (`bird_california_schools`, `bird_financial`, … — they appear in pgAdmin too).
* Asks 100 BIRD questions (a seeded, stratified sample) and all 30 AdventureWorks
  questions, each three ways: **plain** LLM, **+ schema shrink**, **ours** (shrink + gate +
  repair + abstention). Same model, same questions.
* Uses `GEMINI_MODEL` by default and paces itself under the free-tier limit; about 1.5–2
  hours. `--model ollama:qwen2.5-coder:7b` runs it locally instead.
* If the daily quota runs out it stops cleanly; run the same command later to continue.
* Results: `results\benchmark_<model>.md` (tables) and `..._summary.json`.

## 7. Tests

```powershell
pytest -q
```

81 tests. The ones needing a database skip themselves if PostgreSQL is unreachable.

## Troubleshooting

**`ModuleNotFoundError: app`** — run commands from the project root with the venv active.

**`ollama unreachable`** — Ollama runs as a background service; check
<http://127.0.0.1:11434> in a browser. The system falls back to keyword retrieval and
reports it in `/health`, so the API still starts.

**Everything abstains** — open step 5 in the UI (or `attempts[].checks` in `/api/ask`). The usual cause
is a small model producing a hallucinated column (L2) on its first try; `qwen2.5-coder:7b`
or larger fixes most of it. You can also raise `GATE_MAX_REPAIRS` to 3.

**L4 blocks a query you believe is fine** — your `GATE_PLAN_COST_CEILING` is calibrated for
10M rows. Check the real cost with
`EXPLAIN (FORMAT JSON) <query>` and raise the ceiling in `.env` if needed; record the value
you used in the paper.

**`psql: FATAL: database "adventureworks" does not exist`** — step 3 did not finish. Rerun
`python scripts\setup_all.py`; it is safe to repeat.

**The UI says "Gemini error 429"** — the free tier's per-minute limit. The app waits and
retries automatically; if it still fails, wait a minute or pick a local model.

**The UI says "Gemini error 400: API key not valid"** — check `GEMINI_API_KEY` in `.env`
(no quotes, no spaces) and restart uvicorn.

**A question takes 30–60 s with a local model** — normal on a laptop CPU: three model
calls (shrink, SQL, answer). The steps light up one by one, so the class sees progress.
