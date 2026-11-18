/* Ask your database — the web UI. Plain JavaScript, no build step, no external libraries.
 *
 * Every button here calls one of the /api endpoints:
 *   GET  /api/health              status pills in the top bar
 *   GET  /api/models              the model picker
 *   GET  /api/connections/default the "Default (.env)" connection (no password)
 *   POST /api/connections/test    "Test connection"
 *   POST /api/schema              "Load schema" → schema explorer
 *   GET  /api/examples            example question chips
 *   POST /api/ask/stream          "Ask" → the seven steps, live (Server-Sent Events)
 *   GET  /api/runs                "Recent questions"
 *   POST /api/feedback            "This answer is wrong"
 */
"use strict";

// ─── the seven steps, as the class sees them ──────────────────────────────────
const STAGES = [
  { id: "connect",  label: "Connect",  desc: "open a read-only session",
    explain: "We open a connection to the database you chose. The session is read-only and has a time limit, so even a harmful query could not change anything." },
  { id: "schema",   label: "Schema",   desc: "read tables, columns, keys",
    explain: "We read the catalogue: every schema, table, column, data type, primary key and foreign key. It is cached, and re-read automatically when the database changes." },
  { id: "shrink",   label: "Shrink",   desc: "LLM keeps only what matters",
    explain: "The full schema is far too big to send with every question. The model picks only the tables and columns this question needs. We then check its picks against the real catalogue (made-up names are dropped) and add the key columns and bridge tables needed to join them." },
  { id: "generate", label: "Write SQL", desc: "LLM writes the query",
    explain: "The model writes one PostgreSQL query using only the shrunk schema. If a check below fails, the reason is fed back and it tries again (a limited number of times)." },
  { id: "verify",   label: "Verify",   desc: "five checks before it runs",
    explain: "Five deterministic checks — no AI involved. The safety check runs before the name check, so a harmful query is refused even when it names real tables. DELETE, DROP, UPDATE, INSERT, GRANT, system tables and dangerous functions are blocked, never 'repaired'." },
  { id: "execute",  label: "Run",      desc: "execute read-only",
    explain: "Only a query that passed every check is run, inside a read-only transaction, with a row limit and a timeout." },
  { id: "answer",   label: "Answer",   desc: "LLM explains the rows",
    explain: "The model turns the question, the SQL and the rows into a short sentence. Every number in that sentence is checked against the result; if any number is not in the rows, the sentence is thrown away and the table is shown instead." },
];
const STAGE_IDX = Object.fromEntries(STAGES.map((s, i) => [s.id, i]));
const LAYER_ORDER = ["l1", "l3", "l2", "l4", "l5"];
const LAYER_LABEL = {
  l1: "Valid SQL?", l2: "Tables & columns exist?", l3: "Read-only & safe?",
  l4: "Sensible query plan?", l5: "Result looks right?",
};

// ─── state ────────────────────────────────────────────────────────────────────
const state = {
  connMode: "default",
  defaultConn: null,
  schema: null,          // last /api/schema response
  examples: null,
  exGroup: "answerable",
  models: null,
  run: null,             // the run in progress
  abort: null,
  lastResult: null,
};

// ─── tiny helpers ─────────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const fmtMs = (ms) => ms == null ? "" : ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
const fmtInt = (n) => n == null ? "–" : Number(n).toLocaleString();

function toast(msg, bad = false) {
  const t = h("div", { class: "toast" + (bad ? " bad" : "") }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), bad ? 6000 : 3000);
}

const store = {
  get(k, d) { try { const v = localStorage.getItem("t2s." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("t2s." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

async function api(path, body, opts = {}) {
  const init = body === undefined
    ? { method: "GET", signal: opts.signal }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: opts.signal };
  const r = await fetch(path, init);
  let data = null;
  try { data = await r.json(); } catch { /* empty body */ }
  if (!r.ok) {
    let msg = data && data.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    throw new Error(msg || `${r.status} ${r.statusText}`);
  }
  return data;
}

// ─── SQL highlighting (just enough to read it on a projector) ─────────────────
const SQL_KW = new Set(("select from where group by order having limit offset join left right inner outer full cross on as and or not in is null " +
  "distinct case when then else end union all intersect except with recursive over partition desc asc between like ilike exists any " +
  "insert into update delete drop alter create table grant revoke truncate set values returning filter interval cast nulls first last " +
  "true false lateral using window rows range preceding following current row").split(" "));

function highlightSQL(sql) {
  const re = /(--[^\n]*|\/\*[\s\S]*?\*\/)|('(?:[^']|'')*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)(\s*\()?|([\s\S])/g;
  let out = "", m;
  while ((m = re.exec(sql))) {
    if (m[1]) out += `<span class="c">${esc(m[1])}</span>`;
    else if (m[2]) out += `<span class="s">${esc(m[2])}</span>`;
    else if (m[3]) out += `<span class="n">${esc(m[3])}</span>`;
    else if (m[4]) {
      const w = m[4];
      if (SQL_KW.has(w.toLowerCase())) out += `<span class="k">${esc(w.toUpperCase())}</span>`;
      else if (m[5]) out += `<span class="f">${esc(w)}</span>`;
      else out += esc(w);
      if (m[5]) out += esc(m[5]);
    } else out += esc(m[6]);
  }
  return out;
}

function sqlBlock(sql, struck = false) {
  return h("pre", { class: "sql" + (struck ? " struck" : ""), html: highlightSQL(sql || "") });
}

// ─── result tables ────────────────────────────────────────────────────────────
function isNum(v) { return typeof v === "number" || (typeof v === "string" && /^-?\d+(\.\d+)?$/.test(v)); }

function fmtCell(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 4 });
  if (typeof v === "string" && /^-?\d+\.\d{5,}$/.test(v)) return Number(v).toLocaleString(undefined, { maximumFractionDigits: 4 });
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function dataTable(columns, rows, max = 200) {
  if (!rows || !rows.length) return h("div", { class: "empty" }, "The query returned no rows.");
  const cols = columns && columns.length ? columns : Object.keys(rows[0]);
  return h("div", { class: "tablewrap" },
    h("table", { class: "data" },
      h("thead", {}, h("tr", {}, cols.map((c) => h("th", {}, c)))),
      h("tbody", {}, rows.slice(0, max).map((r) => h("tr", {}, cols.map((c) => {
        const v = r[c], f = fmtCell(v);
        return f === null ? h("td", { class: "null" }, "NULL") : h("td", { class: isNum(v) ? "numc" : "", title: f }, f);
      }))))));
}

// ═══ top bar ═══════════════════════════════════════════════════════════════════
async function loadHealth() {
  try {
    const hl = await api("/api/health");
    setPill("#pill-db", hl.database && hl.database.ok, hl.database && hl.database.ok
      ? `${hl.database.name} · ${hl.database.tables} tables` : "database offline",
      hl.database && (hl.database.detail || hl.database.server));
    const p = hl.models || {};
    setPill("#pill-ollama", p.ollama && p.ollama.ok, "ollama", p.ollama && p.ollama.detail);
    setPill("#pill-gemini", p.gemini && p.gemini.ok, "gemini", p.gemini && p.gemini.detail);
  } catch (e) {
    setPill("#pill-db", false, "API offline", e.message);
  }
}

function setPill(sel, ok, text, title) {
  const el = $(sel);
  el.className = "pill " + (ok ? "ok" : "bad");
  el.innerHTML = `<span class="dot"></span>${esc(text)}`;
  el.title = title || "";
}

$("#btn-big").addEventListener("click", () => {
  document.documentElement.classList.toggle("big");
  store.set("big", document.documentElement.classList.contains("big"));
});
if (store.get("big", false)) document.documentElement.classList.add("big");

// ═══ 01 connection ═════════════════════════════════════════════════════════════
function setConnMode(mode) {
  state.connMode = mode;
  $$("#conn-mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  $("#conn-default").classList.toggle("hidden", mode !== "default");
  $("#conn-custom").classList.toggle("hidden", mode !== "custom");
  store.set("connMode", mode);
}
$$("#conn-mode button").forEach((b) => b.addEventListener("click", () => setConnMode(b.dataset.mode)));

function connection() {
  if (state.connMode === "default") {
    const db = $("#default-db").value;
    const c = { use_default: true };
    if (db && state.defaultConn && db !== state.defaultConn.database) c.database = db;
    return c;
  }
  const schemas = $("#c-schemas").value.split(",").map((s) => s.trim()).filter(Boolean);
  return {
    use_default: false,
    host: $("#c-host").value.trim() || "localhost",
    port: Number($("#c-port").value) || 5432,
    database: $("#c-db").value.trim(),
    user: $("#c-user").value.trim(),
    password: $("#c-pw").value,
    sslmode: $("#c-ssl").value,
    schemas: schemas.length ? schemas : null,
    include_views: $("#c-views").value === "1",
  };
}

function connLabel() {
  const c = connection();
  if (c.use_default) {
    const d = state.defaultConn || {};
    return `${c.database || d.database}@${d.host}:${d.port}`;
  }
  return `${c.database}@${c.host}:${c.port}`;
}

// Remember the custom form (never the password).
["#c-host", "#c-port", "#c-db", "#c-user", "#c-ssl", "#c-schemas", "#c-views"].forEach((id) => {
  const el = $(id);
  const saved = store.get("form" + id, null);
  if (saved !== null) el.value = saved;
  el.addEventListener("change", () => store.set("form" + id, el.value));
});

async function loadDefaultConnection() {
  try {
    const d = await api("/api/connections/default");
    state.defaultConn = d;
    $("#default-info").innerHTML =
      `<span>host</span><span>${esc(d.host)}:${esc(d.port)}</span>` +
      `<span>user</span><span>${esc(d.user)}</span>` +
      `<span>password</span><span>${d.has_password ? "set in .env ✓" : "<b>missing</b>"}</span>`;
    const sel = $("#default-db");
    sel.innerHTML = "";
    sel.append(h("option", { value: d.database }, d.database));
  } catch (e) {
    $("#default-info").innerHTML = `<span>error</span><span>${esc(e.message)}</span>`;
  }
}

function fillDatabaseList(dbs) {
  if (!dbs || !state.defaultConn) return;
  const sel = $("#default-db");
  const cur = sel.value || state.defaultConn.database;
  sel.innerHTML = "";
  for (const db of dbs) sel.append(h("option", { value: db, selected: db === cur }, db + (db === state.defaultConn.database ? "  (.env)" : "")));
}

function connStatus(ok, html) {
  $("#conn-status").innerHTML = `<div class="status-box ${ok ? "ok" : "bad"}">${html}</div>`;
}

async function testConnection() {
  const btn = $("#btn-test");
  btn.disabled = true;
  try {
    const r = await api("/api/connections/test", { ...connection() });
    connStatus(true,
      `<b>Connected</b> in ${r.latency_ms} ms<div class="kv">` +
      `<span>server</span><span>PostgreSQL ${esc(r.server_version)}</span>` +
      `<span>database</span><span>${esc(r.database)} · ${r.tables} tables in ${r.schemas} schemas</span>` +
      `<span>user</span><span>${esc(r.user)} (read-only session)</span>` +
      `<span>other DBs</span><span>${esc((r.databases || []).join(", "))}</span></div>`);
    if (state.connMode === "default") fillDatabaseList(r.databases);
    return true;
  } catch (e) {
    connStatus(false, `<b>Could not connect.</b><br>${esc(e.message)}`);
    return false;
  } finally { btn.disabled = false; }
}

$("#btn-test").addEventListener("click", testConnection);
$("#btn-schema").addEventListener("click", () => loadSchema(false));
$("#btn-refresh-schema").addEventListener("click", () => loadSchema(true));
$("#default-db").addEventListener("change", () => loadSchema(false));

// ═══ 02 schema explorer ════════════════════════════════════════════════════════
async function loadSchema(refresh) {
  const btn = $("#btn-schema");
  btn.disabled = true;
  $("#schema-tree").innerHTML = `<div class="waiting"><span class="spinner"></span> Reading the catalogue…</div>`;
  if (!(await testConnection())) {
    btn.disabled = false;
    $("#schema-tree").innerHTML = `<div class="empty">Fix the connection first.</div>`;
    return;
  }
  try {
    const t0 = performance.now();
    const s = await api("/api/schema", { connection: connection(), refresh: !!refresh });
    state.schema = s;
    renderSchema();
    const c = s.cache || {};
    $("#schema-note").textContent =
      `${s.database} · read ${c.cached ? "from " + c.cached + " cache" : "live from the database"} in ${fmtMs(performance.now() - t0)} · ` +
      `~${fmtInt(s.stats.estimated_rows)} rows in total`;
    updateExampleNote();
  } catch (e) {
    $("#schema-tree").innerHTML = `<div class="err">${esc(e.message)}</div>`;
    connStatus(false, `<b>Could not read the schema.</b><br>${esc(e.message)}`);
  } finally { btn.disabled = false; }
}

function renderSchema(picked) {
  const s = state.schema;
  if (!s) return;
  const st = s.stats;
  $("#schema-stats").innerHTML = [["schemas", st.schemas], ["tables", st.tables], ["columns", st.columns], ["FKs", st.foreign_keys]]
    .map(([k, v]) => `<div class="stat"><b>${fmtInt(v)}</b><span>${k}</span></div>`).join("");

  const q = $("#schema-search").value.trim().toLowerCase();
  const pickedMap = picked || state.picked || {};
  const tree = $("#schema-tree");
  tree.innerHTML = "";
  let shown = 0;
  for (const sch of s.schemas) {
    const tables = sch.tables.filter((t) => !q || t.qualified.toLowerCase().includes(q) ||
      t.columns.some((c) => c.name.toLowerCase().includes(q)));
    if (!tables.length) continue;
    const anyPicked = tables.some((t) => pickedMap[t.qualified]);
    const det = h("details", { class: "sch", open: !!q || anyPicked || s.schemas.length === 1 },
      h("summary", {}, sch.name, h("span", { class: "n" }, `${tables.length} tables`)));
    for (const t of tables) {
      const pk = pickedMap[t.qualified];
      const tdet = h("details", { class: "tbl" + (pk ? " picked" : ""), open: !!pk || (!!q && t.columns.some((c) => c.name.toLowerCase().includes(q))) },
        h("summary", { title: t.comment || "" }, t.name, h("span", { class: "rows" }, t.rows != null ? `~${fmtInt(t.rows)}` : "")));
      if (t.comment) tdet.append(h("div", { class: "tcomment" }, t.comment));
      const cols = h("div", { class: "cols" });
      for (const c of t.columns) {
        const cp = pk && (pk.length === 0 || pk.includes(c.name));
        cols.append(h("div", { class: "col" + (cp && pk.length ? " picked" : ""), title: c.samples && c.samples.length ? "e.g. " + c.samples.join(", ") : "" },
          h("span", { class: "cn" }, c.name),
          h("span", { class: "ty" }, c.type + (c.nullable ? "" : " not null")),
          c.pk ? h("span", { class: "badge pk" }, "PK") : null,
          c.fk ? h("span", { class: "badge fk" }, "FK") : null,
          c.fk ? h("span", { class: "fkto" }, "→ " + c.fk) : null));
      }
      tdet.append(cols);
      det.append(tdet);
      shown++;
    }
    tree.append(det);
  }
  if (!shown) tree.innerHTML = q ? `<div class="empty">Nothing matches “${esc(q)}”.</div>`
    : `<div class="empty">This database has no tables this user can read.</div>`;
}
$("#schema-search").addEventListener("input", () => renderSchema());

// ═══ 03 ask: models, examples, options ═════════════════════════════════════════
async function loadModels(refresh = false) {
  try {
    const m = await api("/api/models" + (refresh ? "?refresh=true" : ""));
    state.models = m;
    const groups = { ollama: "On this laptop · Ollama", gemini: "Cloud · Google Gemini (free tier)", openai: "Cloud · OpenAI", anthropic: "Cloud · Anthropic", mock: "Testing" };
    const ids = new Set(m.models.map((x) => x.id));
    const saved = store.get("model", null);
    const fill = (sel, withDefault) => {
      sel.innerHTML = "";
      if (withDefault) sel.append(h("option", { value: "" }, "same as above"));
      if (!withDefault && m.default && !ids.has(m.default)) sel.append(h("option", { value: m.default }, `${m.default} (default, not reachable)`));
      for (const [prov, label] of Object.entries(groups)) {
        const items = m.models.filter((x) => x.provider === prov);
        if (!items.length) continue;
        sel.append(h("optgroup", { label }, items.map((x) => h("option", { value: x.id }, x.label))));
      }
    };
    fill($("#model"), false);
    ["#m-schema", "#m-sql", "#m-answer"].forEach((id) => fill($(id), true));
    $("#model").value = saved && (ids.has(saved) || saved === m.default) ? saved : m.default;
    if (!$("#model").value && m.models.length) $("#model").value = m.models[0].id;
  } catch (e) {
    toast("Could not list models: " + e.message, true);
  }
}
$("#model").addEventListener("change", () => store.set("model", $("#model").value));
$("#o-per-stage").addEventListener("change", (e) => $$(".per-stage").forEach((el) => el.classList.toggle("hidden", !e.target.checked)));

async function loadExamples() {
  try {
    state.examples = await api("/api/examples");
    renderExamples();
  } catch (e) { $("#ex-chips").innerHTML = `<span class="muted">${esc(e.message)}</span>`; }
}

function renderExamples() {
  const box = $("#ex-chips");
  box.innerHTML = "";
  const items = (state.examples && state.examples.groups[state.exGroup]) || [];
  for (const it of items) {
    box.append(h("button", { class: "chip", title: it.id, onclick: () => { $("#question").value = it.question; $("#question").focus(); } },
      it.difficulty ? h("span", { class: "lvl" }, it.difficulty) : null, it.question));
  }
}
$$("#ex-tabs button").forEach((b) => b.addEventListener("click", () => {
  state.exGroup = b.dataset.g;
  $$("#ex-tabs button").forEach((x) => x.classList.toggle("on", x === b));
  renderExamples();
}));

function updateExampleNote() {
  const db = state.schema ? state.schema.database : null;
  const exDb = state.examples ? state.examples.database : "adventureworks";
  $("#ex-note").textContent = db && db !== exDb
    ? `These examples are for ${exDb}; you are connected to ${db} — type your own question.`
    : `Example questions for ${exDb}`;
}

function options() {
  return {
    schema_strategy: $("#o-strategy").value,
    prune_columns: $("#o-prunecols").checked,
    max_tables: Number($("#o-maxtables").value) || 8,
    verify: $("#o-verify").checked,
    disabled_layers: $$("#o-layers input:checked").map((x) => x.value),
    max_repairs: Number($("#o-repairs").value),
    row_limit: Number($("#o-rows").value) || 200,
    refresh_schema: $("#o-refresh").checked,
  };
}

function requestBody() {
  const body = { question: $("#question").value.trim(), connection: connection(), model: $("#model").value, options: options() };
  if ($("#o-per-stage").checked) {
    body.models = { schema: $("#m-schema").value, sql: $("#m-sql").value, answer: $("#m-answer").value };
  }
  return body;
}

// ═══ 04 the pipeline ═══════════════════════════════════════════════════════════
function renderFlow() {
  const flow = $("#flow");
  flow.innerHTML = "";
  STAGES.forEach((s, i) => {
    flow.append(h("div", { class: "node", id: "node-" + s.id, onclick: () => {
      const card = $("#stage-" + s.id);
      if (card) { card.classList.remove("collapsed"); card.scrollIntoView({ behavior: "smooth", block: "start" }); }
    } },
      h("div", { class: "circle" }, String(i + 1)),
      h("div", { class: "lbl" }, s.label),
      h("div", { class: "desc" }, s.desc),
      h("div", { class: "ms" })));
  });
}

function renderHowto() {
  $("#howto").innerHTML = STAGES.map((s, i) => `<div><b>${i + 1}. ${esc(s.label)}</b>${esc(s.explain)}</div>`).join("");
}

function setNode(stage, status, ms) {
  const n = $("#node-" + stage);
  if (!n) return;
  n.className = "node " + status;
  if (ms != null) n.querySelector(".ms").textContent = fmtMs(ms);
  else if (status === "running") n.querySelector(".ms").textContent = "…";
}

function setChip(status, text) {
  const c = $("#run-chip");
  c.className = "status-chip " + status;
  c.textContent = text || status;
}

function stageCard(stage) {
  let card = $("#stage-" + stage);
  if (card) return card;
  const s = STAGES[STAGE_IDX[stage]];
  card = h("section", { class: "card stage", id: "stage-" + stage },
    h("div", { class: "hd", onclick: () => card.classList.toggle("collapsed") },
      h("span", { class: "num" }, String(STAGE_IDX[stage] + 1)),
      h("div", {}, h("h2", {}, s.label), h("div", { class: "what" }, s.desc)),
      h("div", { class: "right" }, h("span", { class: "model hidden" }), h("span", { class: "ms" }))),
    h("div", { class: "bd" }, h("p", { class: "explain" }, s.explain), h("div", { class: "content" })));
  // keep cards in pipeline order
  const container = $("#stages");
  const after = Array.from(container.children).find((c) => STAGE_IDX[c.id.replace("stage-", "")] > STAGE_IDX[stage]);
  container.insertBefore(card, after || null);
  return card;
}

function setCard(stage, status, { ms, model, content } = {}) {
  const card = stageCard(stage);
  card.classList.remove("running", "done", "failed", "error", "blocked", "skipped");
  card.classList.add(status);
  if (ms != null) card.querySelector(".hd .ms").textContent = fmtMs(ms);
  if (model) {
    const m = card.querySelector(".hd .model");
    m.textContent = model;
    m.classList.remove("hidden");
  }
  if (content !== undefined) {
    const c = card.querySelector(".content");
    c.innerHTML = "";
    if (content) c.append(content);
  }
  return card;
}

const waiting = (text) => h("div", { class: "waiting" }, h("span", { class: "spinner" }), text);

// ─── stage renderers ──────────────────────────────────────────────────────────
function renderConnect(d) {
  return h("div", { class: "kv" },
    h("span", {}, "database"), h("span", {}, `${d.database} on ${d.host}:${d.port}`),
    h("span", {}, "user"), h("span", {}, `${d.user} · read-only session, statement timeout`),
    h("span", {}, "server"), h("span", {}, d.server || ""),
    h("span", {}, "latency"), h("span", {}, `${d.latency_ms} ms`));
}

function renderSchemaStage(d) {
  return h("div", {},
    h("div", { class: "metrics" },
      metric(fmtInt(d.tables), "tables"), metric(fmtInt(d.columns), "columns"),
      metric(fmtInt(d.foreign_keys), "foreign keys"),
      metric(d.cached ? `${d.cached} cache` : "live read", "source")),
    h("p", { class: "rowinfo" }, `schema version ${d.schema_version} — the cache is thrown away automatically when this changes.`));
}

function metric(v, label) { return h("div", { class: "metric" }, h("b", {}, v), h("span", {}, label)); }

function bar(label, before, after, suffix = "") {
  const pct = before ? Math.max(1.5, (after / before) * 100) : 0;
  const track = h("div", { class: "track" }, h("i", { style: "width:100%" }));
  setTimeout(() => { track.firstChild.style.width = pct + "%"; }, 60);
  return h("div", { class: "sbar" }, h("span", { class: "lab" }, label), track,
    h("span", { class: "nums" }, `${fmtInt(before)} → ${fmtInt(after)}${suffix}`));
}

function renderShrink(d) {
  const wrap = h("div", {});
  if (d.error) wrap.append(h("div", { class: "warnbox" }, `The model's selection could not be used (${d.error}); fell back to keyword matching.`));
  if (d.strategy === "full") wrap.append(h("div", { class: "warnbox" }, "Baseline mode: the full schema is sent to the model, nothing is shrunk."));
  wrap.append(h("div", { class: "shrinkbars" },
    bar("tables", d.tables_before, d.tables_after),
    bar("columns", d.columns_before, d.columns_after),
    bar("tokens", d.full_schema_tokens, d.shrunk_schema_tokens, d.reduction_pct != null ? `  (−${d.reduction_pct}%)` : "")));
  if (d.prefiltered_to) wrap.append(h("p", { class: "rowinfo" }, `Schema was too big for one prompt: a keyword pre-filter first narrowed it to ${d.prefiltered_to} candidate tables.`));
  const hall = [...(d.hallucinated_tables || []), ...(d.hallucinated_columns || [])];
  if (hall.length) wrap.append(h("div", { class: "warnbox" }, `The model named things that do not exist — dropped: ${hall.join(", ")}`));
  const corr = Object.entries(d.corrected_tables || {});
  if (corr.length) wrap.append(h("div", { class: "warnbox" }, "Corrected near-misses: " + corr.map(([a, b]) => `${a} → ${b}`).join(", ")));
  if (d.answerable === false) wrap.append(h("div", { class: "warnbox" }, "The model says no table in this database holds what the question asks about."));
  const list = h("div", { class: "sel-list" });
  for (const s of d.selected || []) {
    list.append(h("div", { class: "sel" },
      h("div", { class: "t" }, s.table, h("span", { class: "src " + s.source }, s.source === "llm" ? "picked by model" : s.source === "bridge" ? "join bridge" : s.source)),
      s.reason ? h("div", { class: "why" }, s.reason) : null,
      h("div", { class: "cc" }, (s.columns && s.columns.length ? s.columns : ["all columns"]).map((c) => h("span", {}, c)))));
  }
  if ((d.selected || []).length) wrap.append(list);
  return wrap;
}

function renderAttempts() {
  const run = state.run;
  const box = h("div", {});
  for (const a of run.attempts) {
    const el = h("div", { class: "attempt" },
      h("div", { class: "ah" }, h("b", {}, `Attempt ${a.n}`),
        a.status === "running" ? h("span", { class: "spinner" }) : null,
        a.status === "running" ? h("span", { class: "muted" }, "the model is writing SQL…") : null,
        a.ms != null ? h("span", { class: "muted mono" }, fmtMs(a.ms)) : null,
        a.passed === true ? h("span", { class: "gbadge ok" }, "✓ passed all checks") : null,
        a.passed === false ? h("span", { class: "gbadge bad" }, a.blocked ? "✗ blocked" : "✗ failed a check") : null));
    if (a.repair_hint) el.append(h("div", { class: "repair" }, "Fed back to the model: " + a.repair_hint));
    if (a.error) el.append(h("div", { class: "err" }, a.error));
    if (a.sql) el.append(sqlBlock(a.sql, a.blocked));
    else if (a.status === "done" && !a.error) el.append(h("div", { class: "warnbox" }, "The model said this cannot be answered from the selected tables."));
    if (a.reasoning) el.append(h("p", { class: "reasoning" }, h("b", {}, "Model's reasoning: "), a.reasoning));
    box.append(el);
  }
  return box;
}

function checkRows(checks, pendingAll) {
  const byLayer = Object.fromEntries((checks || []).map((c) => [c.layer, c]));
  const failedAt = (checks || []).find((c) => !c.passed && !c.skipped);
  return h("div", { class: "checks" }, LAYER_ORDER.map((ly) => {
    const c = byLayer[ly];
    if (!c) {
      const why = pendingAll ? "checking…" : failedAt ? "not run — an earlier check failed" : "not run";
      return h("div", { class: "chk skip pending" }, h("span", { class: "ic" }, ""), h("span", { class: "ly" }, ly.toUpperCase()),
        h("span", { class: "nm" }, LAYER_LABEL[ly]), h("span", { class: "msg muted" }, why), h("span", { class: "t" }));
    }
    const cls = c.skipped ? "skip" : !c.passed ? "fail" : c.warning ? "warnc" : "pass";
    const icon = c.skipped ? "–" : !c.passed ? "✗" : c.warning ? "!" : "✓";
    return h("div", { class: "chk " + cls }, h("span", { class: "ic" }, icon), h("span", { class: "ly" }, ly.toUpperCase()),
      h("span", { class: "nm" }, c.name || LAYER_LABEL[ly]), h("span", { class: "msg" }, c.skipped ? (c.message || "switched off") : c.message),
      h("span", { class: "t" }, c.ms != null ? `${c.ms.toFixed(1)} ms` : ""));
  }));
}

function renderVerify() {
  const box = h("div", {});
  const tried = state.run.attempts.filter((a) => a.checks || a.verifying);
  tried.forEach((a, i) => {
    const last = i === tried.length - 1;
    const el = h("div", { class: "attempt" },
      h("div", { class: "ah" }, h("b", {}, `Checks on attempt ${a.n}`),
        a.verifying ? h("span", { class: "spinner" }) : null,
        a.verifyMs != null ? h("span", { class: "muted mono" }, fmtMs(a.verifyMs)) : null),
      checkRows(a.checks, a.verifying));
    if (a.warnings && a.warnings.length) el.append(h("div", { class: "warnbox", style: "margin-top:8px" }, "Warnings: " + a.warnings.join("; ")));
    if (!last && a.passed === false) el.append(h("p", { class: "rowinfo" }, "↳ the reason was sent back to the model for another try."));
    box.append(el);
  });
  return box;
}

function renderExecute(d) {
  return h("div", {},
    h("div", { class: "metrics" }, metric(fmtInt(d.row_count), d.truncated ? "rows (limit hit)" : "rows"),
      metric(fmtMs(d.db_ms), "database time"), metric(String((d.columns || []).length), "columns")),
    dataTable(d.columns, d.rows, 50),
    d.row_count > 50 ? h("p", { class: "rowinfo" }, `Showing the first 50 of ${fmtInt(d.row_count)} rows.`) : null);
}

function renderAnswer(d) {
  const box = h("div", {});
  if (d.answer) box.append(h("p", { style: "font-size:1.05rem;margin:0 0 8px" }, d.answer));
  box.append(d.grounded
    ? h("span", { class: "gbadge ok" }, "✓ every number in the answer appears in the result")
    : h("span", { class: "gbadge bad" }, "✗ rejected: numbers not found in the result — " + (d.ungrounded_numbers || []).join(", ")));
  if (d.rejected_answer) box.append(h("p", { class: "rowinfo" }, h("b", {}, "Rejected wording: "), h("s", {}, d.rejected_answer)));
  if (d.error) box.append(h("div", { class: "warnbox", style: "margin-top:8px" }, d.error));
  if (!d.llm_used && d.answer) box.append(h("p", { class: "rowinfo" }, "No model call was needed for this answer."));
  return box;
}

// ─── the stream ───────────────────────────────────────────────────────────────
function resetRun(question) {
  state.run = { question, attempts: [], started: performance.now(), reached: new Set() };
  state.picked = null;
  $("#stages").innerHTML = "";
  $("#final").classList.add("hidden");
  $("#howto-card").classList.add("hidden");
  STAGES.forEach((s) => setNode(s.id, "", null));
  $$(".node .ms").forEach((m) => { m.textContent = ""; });
  setChip("running", "running");
  if (state.schema) renderSchema({});
}

function onEvent(ev) {
  if (ev.type === "result") return onResult(ev.data);
  if (ev.type === "error") { setChip("error", "error"); toast(ev.message, true); return; }
  if (ev.type !== "stage") return;

  const { stage, status, ms, data, error, attempt, model } = ev;
  const run = state.run;
  run.reached.add(stage);
  const nodeStatus = status === "failed" ? "failed" : status;
  setNode(stage, nodeStatus, ms);

  switch (stage) {
    case "connect":
    case "schema":
    case "execute":
    case "answer": {
      const render = { connect: renderConnect, schema: renderSchemaStage, execute: renderExecute, answer: renderAnswer }[stage];
      if (status === "running") {
        const msg = { connect: "Opening a read-only connection…", schema: "Reading tables, columns and keys…",
          execute: "Running the query…", answer: "The model is writing the answer…" }[stage];
        setCard(stage, "running", { model, content: waiting(msg) });
      } else if (status === "error") setCard(stage, "error", { ms, content: h("div", { class: "err" }, error) });
      else setCard(stage, "done", { ms, content: render(data || {}) });
      break;
    }
    case "shrink":
      if (status === "running") setCard("shrink", "running", { model, content: waiting("The model is choosing the relevant tables and columns…") });
      else {
        setCard("shrink", status === "error" ? "error" : "done", { ms, content: status === "error" ? h("div", { class: "err" }, error) : renderShrink(data) });
        if (data && data.selected) {
          const picked = {};
          data.selected.forEach((s) => { picked[s.table] = s.columns || []; });
          state.picked = picked;
          renderSchema(picked);
        }
      }
      break;
    case "generate": {
      let a = run.attempts.find((x) => x.n === attempt);
      if (!a) { a = { n: attempt }; run.attempts.push(a); }
      if (status === "running") { a.status = "running"; a.model = model; }
      else if (status === "error") { a.status = "error"; a.error = error; a.ms = ms; }
      else Object.assign(a, { status: "done", ms, sql: data.sql, reasoning: data.reasoning, repair_hint: data.repair_hint });
      setCard("generate", status === "running" ? "running" : status === "error" ? "error" : "done",
        { ms: run.attempts.reduce((t, x) => t + (x.ms || 0), 0) || null, model, content: renderAttempts() });
      break;
    }
    case "verify": {
      const a = run.attempts.find((x) => x.n === attempt);
      if (!a) break;
      if (status === "running") { a.verifying = true; }
      else {
        a.verifying = false;
        a.verifyMs = ms;
        a.checks = data.checks;
        a.passed = data.passed;
        a.warnings = data.warnings;
        const failed = (data.checks || []).find((c) => !c.passed && !c.skipped);
        a.blocked = !!(failed && (failed.layer === "l3" || (failed.details && failed.details.blocked)));
        if (a.blocked) setNode("verify", "blocked", ms);
      }
      setCard("verify", status === "running" ? "running" : a.blocked ? "blocked" : status === "failed" ? "failed" : "done",
        { ms, content: renderVerify() });
      setCard("generate", $("#stage-generate").classList.contains("error") ? "error" : "done", { content: renderAttempts() });
      break;
    }
  }
}

function onResult(r) {
  state.lastResult = r;
  const run = state.run;
  // steps the run never reached
  STAGES.forEach((s) => { if (!run.reached.has(s.id)) setNode(s.id, "skipped", null); });
  const chipText = { answered: "answered", blocked: "blocked", abstained: "abstained", error: "error" }[r.status] || r.status;
  setChip(r.status, chipText + " · " + fmtMs(r.total_ms));
  renderFinal(r);
  loadRuns();
}

function renderFinal(r) {
  const el = $("#final");
  el.className = "card final " + r.status;
  el.innerHTML = "";
  const titles = { answered: "Answer", blocked: "Refused — this request is not safe", abstained: "No answer — and that is on purpose", error: "Something went wrong" };
  const confCls = r.confidence >= 0.75 ? "" : r.confidence >= 0.5 ? "mid" : "low";
  const body = h("div", { class: "bd" },
    h("p", { class: "q" }, "“" + r.question + "”"),
    r.status === "answered"
      ? h("p", { class: "a" }, r.answer)
      : h("p", { class: "a reason" }, r.abstain_reason || "No answer."),
    r.status === "answered" ? h("div", { class: "conf" },
      h("span", { class: "muted", style: "font-size:.8rem" }, "Confidence"),
      h("div", { class: "bar" }, h("i", { class: confCls, style: `width:${Math.round(r.confidence * 100)}%` })),
      h("b", { style: "font-size:.85rem" }, `${Math.round(r.confidence * 100)}% · ${r.confidence_label}`),
      r.grounded === true ? h("span", { class: "gbadge ok" }, "✓ grounded in the rows") : r.grounded === false ? h("span", { class: "gbadge bad" }, "table shown instead of a sentence") : null) : null,
    h("div", { class: "metrics" },
      metric(fmtMs(r.total_ms), "total time"),
      metric(String(r.llm_calls), "LLM calls"),
      metric(fmtInt(r.prompt_tokens_est), "prompt tokens (est.)"),
      metric(String(r.attempts.length), r.attempts.length === 1 ? "SQL attempt" : "SQL attempts"),
      r.shrink && r.shrink.tables_after != null ? metric(`${r.shrink.tables_after}/${r.shrink.tables_before}`, "tables used") : null,
      metric(r.status === "answered" || r.row_count ? fmtInt(r.row_count) : "–", "rows")),
    r.sql ? h("div", { style: "margin-top:10px" }, sqlBlock(r.sql, r.status === "blocked")) : null,
    r.status === "answered" && r.rows && r.rows.length ? h("div", { style: "margin-top:10px" }, dataTable(r.columns, r.rows, 200)) : null,
    h("div", { class: "feedback" },
      r.sql ? h("button", { class: "btn small", onclick: () => { navigator.clipboard && navigator.clipboard.writeText(r.sql); toast("SQL copied"); } }, "Copy SQL") : null,
      h("button", { class: "btn small", onclick: (e) => showFeedback(e.target, r) }, "This answer is wrong"),
      h("span", { class: "muted mono", style: "font-size:.7rem" }, `run ${r.run_id} · ${Object.values(r.models || {}).filter((v, i, a) => a.indexOf(v) === i).join(", ")}`)),
    h("details", { class: "raw", style: "margin-top:12px" }, h("summary", {}, "Raw JSON response"), h("pre", {}, JSON.stringify(r, null, 2))));
  el.append(h("div", { class: "hd" }, h("span", { class: "status-chip " + r.status }, r.status), h("h2", {}, titles[r.status] || r.status)), body);
  el.classList.remove("hidden");
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showFeedback(btn, r) {
  const box = btn.parentElement;
  if (box.querySelector("input")) return;
  const sel = h("select", { style: "max-width:180px" }, ["wrong numbers", "wrong tables", "misread the question", "should have answered", "should have refused", "other"].map((x) => h("option", {}, x)));
  const inp = h("input", { type: "text", placeholder: "what was wrong? (optional)" });
  const send = h("button", { class: "btn small dark", onclick: async () => {
    try {
      await api("/api/feedback", { run_id: r.run_id, is_wrong: true, category: sel.value, note: inp.value || null });
      toast("Thanks — recorded in feedback.jsonl");
      sel.remove(); inp.remove(); send.remove();
    } catch (e) { toast(e.message, true); }
  } }, "Send");
  box.append(sel, inp, send);
}

async function ask() {
  const body = requestBody();
  if (body.question.length < 3) { toast("Type a question first.", true); $("#question").focus(); return; }
  if (!body.connection.use_default && (!body.connection.database || !body.connection.user)) {
    toast("Fill in the database and user, or switch to the default connection.", true); return;
  }
  if (state.abort) state.abort.abort();
  const ctrl = new AbortController();
  state.abort = ctrl;
  resetRun(body.question);
  $("#btn-ask").disabled = true;
  $("#btn-stop").classList.remove("hidden");

  try {
    const resp = await fetch("/api/ask/stream", {
      method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body), signal: ctrl.signal,
    });
    if (!resp.ok) {
      let msg = `${resp.status} ${resp.statusText}`;
      try { const j = await resp.json(); msg = Array.isArray(j.detail) ? j.detail.map((d) => d.msg).join("; ") : (j.detail || msg); } catch { /* not JSON */ }
      throw new Error(msg);
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const data = block.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n");
        if (data) {
          try { onEvent(JSON.parse(data)); } catch (e) { console.error("bad event", e, data); }
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") { setChip("", "stopped"); toast("Stopped. (The server finishes the current step in the background.)"); }
    else { setChip("error", "error"); toast(e.message, true); }
  } finally {
    $("#btn-ask").disabled = false;
    $("#btn-stop").classList.add("hidden");
    state.abort = null;
  }
}

$("#btn-ask").addEventListener("click", ask);
$("#btn-stop").addEventListener("click", () => state.abort && state.abort.abort());
$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey || !e.shiftKey)) { e.preventDefault(); ask(); }
});

// ═══ history ═══════════════════════════════════════════════════════════════════
async function loadRuns() {
  try {
    const runs = await api("/api/runs?limit=25");
    const box = $("#runs");
    box.innerHTML = "";
    if (!runs.length) { box.innerHTML = `<div class="empty">No questions yet.</div>`; return; }
    for (const r of runs) {
      const models = Object.values(r.models || {}).filter((v, i, a) => a.indexOf(v) === i).join(", ");
      box.append(h("div", { class: "hrow", title: "Click to ask again", onclick: () => { $("#question").value = r.question; window.scrollTo({ top: 0, behavior: "smooth" }); } },
        h("span", { class: "status-chip " + r.status }, r.status),
        h("span", { class: "q" }, r.question),
        h("span", { class: "mm" }, models),
        h("span", { class: "mm" }, fmtMs(r.total_ms))));
    }
  } catch { /* the log is optional */ }
}
$("#btn-runs").addEventListener("click", loadRuns);

// ═══ start ═════════════════════════════════════════════════════════════════════
renderFlow();
renderHowto();
setConnMode(store.get("connMode", "default"));
(async () => {
  await Promise.all([loadDefaultConnection(), loadModels(), loadExamples()]);
  loadHealth();
  loadRuns();
  await loadSchema(false);
  updateExampleNote();
})();
