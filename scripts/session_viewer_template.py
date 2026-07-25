"""The session dashboard's self-contained HTML template, shared by the static
exporter (scripts/export_session_viewer.py) and the logging server
(scripts/session_server.py). Extracted verbatim from export_session_viewer.py
to keep that script small; see its module docstring for the page anatomy and
the color-validation notes.
"""

# Chart chrome tokens + validated series pair, declared once as CSS custom
# properties with dark mode as its own selected set (see module docstring).
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Session tracker</title>
<style>
:root {
  --surface: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
  --hairline: #e1e0d9; --baseline: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --wash: #f0efec;
  --s-total: #2a78d6; --s-table: #1baf7a;
  --good: #006300;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --page: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
    --hairline: #2c2c2a; --baseline: #383835; --ring: rgba(255,255,255,0.10);
    --wash: #383835;
    --s-total: #3987e5; --s-table: #199e70;
    --good: #0ca30c;
  }
}
* { box-sizing: border-box; margin: 0; }
body { font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page); color: var(--ink); padding: 24px; }
.shell { max-width: 980px; margin: 0 auto; }
h1 { font-size: 18px; }
.sub { color: var(--ink-muted); font-size: 13px; margin-top: 2px; }
.card { background: var(--surface); border: 1px solid var(--ring);
  border-radius: 10px; padding: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px; margin-top: 16px; }
.tile .label { font-size: 12px; color: var(--ink-muted); }
.tile .value { font-size: 26px; font-weight: 600; margin-top: 2px; }
.tile .hint { font-size: 12px; color: var(--ink-2); margin-top: 2px; }
#chartCard { margin-top: 12px; }
.legend { display: flex; gap: 18px; font-size: 12px; color: var(--ink-2); margin: 2px 0 6px; }
.legend .key { display: inline-block; width: 16px; height: 0; border-top: 2px solid;
  vertical-align: 3px; margin-right: 6px; border-radius: 1px; }
#plotWrap { position: relative; }
svg { display: block; width: 100%; height: auto; }
svg:focus { outline: 2px solid var(--ink); outline-offset: 2px; border-radius: 6px; }
#tip { position: absolute; pointer-events: none; background: var(--surface);
  border: 1px solid var(--ring); border-radius: 8px; padding: 8px 10px;
  font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); display: none;
  min-width: 170px; z-index: 2; }
#tip .when { color: var(--ink-muted); margin-bottom: 4px; }
#tip .row { display: flex; align-items: center; gap: 7px; margin-top: 2px; }
#tip .row .key { width: 12px; height: 0; border-top: 2px solid; border-radius: 1px; flex: none; }
#tip .row b { font-variant-numeric: tabular-nums; }
#tip .row span { color: var(--ink-2); }
#tip .own { margin-top: 5px; padding-top: 5px; border-top: 1px solid var(--hairline);
  color: var(--ink-2); }
table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--ink-muted); font-weight: 600; padding: 6px 10px 4px; }
th.sym { text-transform: none; } /* uppercasing greek rho yields a P-lookalike */
td { padding: 6px 10px; border-top: 1px solid var(--hairline);
  font-variant-numeric: tabular-nums; }
td.txt { font-variant-numeric: normal; color: var(--ink-2); }
th.num, td.num { text-align: right; }
.section { margin-top: 20px; }
.section h2 { font-size: 14px; margin-bottom: 6px; }
.stakeCard { margin-top: 10px; }
.stakeCard .head { font-size: 13px; color: var(--ink-2); }
.xcheck { margin-top: 6px; font-size: 12px; color: var(--ink-muted); }
.xcheck.warn { color: var(--ink); }
.stakeSel { display: block; font-size: 12px; color: var(--ink-muted); margin-top: 10px; }
.stakeSel select { margin-left: 6px; font: inherit; font-size: 12px; color: var(--ink);
  background: var(--page); border: 1px solid var(--hairline); border-radius: 6px;
  padding: 3px 6px; }
.empty { margin-top: 16px; text-align: left; }
.empty code { display: block; background: var(--wash); border-radius: 6px;
  padding: 8px 10px; margin-top: 8px; font-size: 13px; }
footer { margin-top: 14px; font-size: 12px; color: var(--ink-muted); }
#logCard { margin-top: 16px; }
#logCard .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 10px 14px; margin-top: 10px; }
#logCard label { display: block; font-size: 12px; color: var(--ink-muted); }
#logCard label.rec { color: var(--ink); font-weight: 600; }
#logCard input, #logCard select {
  width: 100%; margin-top: 3px; padding: 6px 8px; font: inherit; font-size: 13px;
  color: var(--ink); background: var(--page); border: 1px solid var(--hairline);
  border-radius: 6px; }
#logCard input:focus, #logCard select:focus { outline: 2px solid var(--ink); outline-offset: -1px; }
#logCard .auto { font-size: 12px; color: var(--ink-2); margin-top: 8px; min-height: 16px; }
#logCard .actions { display: flex; align-items: center; gap: 14px; margin-top: 12px; }
#logCard button { font: inherit; font-size: 13px; font-weight: 600; padding: 7px 16px;
  border: 0; border-radius: 6px; background: var(--ink); color: var(--surface); cursor: pointer; }
#logCard .err { font-size: 12px; color: var(--ink); }
#logCard .uploadHint { font-size: 12px; color: var(--ink-muted); }
#logCard input[type=file] { display: none; }
#logCard .uploadHint a { color: var(--ink-2); cursor: pointer; text-decoration: underline; }
</style>
</head>
<body>
<div class="shell">
  <h1>Session tracker</h1>
  <div class="sub" id="sub"></div>
  <div id="content"></div>
  <footer>table result = balance delta minus claims/deposits plus withdrawals &#183;
    rakeback earned = claimable-balance delta plus claims &#183;
    &#961; = rakeback earned per unit of rake-paid counter movement</footer>
</div>
<script>
const DATA = __DATA__;
const C = DATA.currency;
const content = document.getElementById("content");
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const money = v => (v < 0 ? "\\u2212" : "+") + C + Math.abs(v).toFixed(2);
const moneyOrDash = v => v === null ? "\\u2014" : money(v);
const dateOf = ts => ts.split(" ")[0];
const hhmm = ts => ts.split(" ")[1].slice(0, 5);

document.getElementById("sub").textContent = DATA.summary.n
  ? `${DATA.summary.n} sessions \\u00b7 ${dateOf(DATA.sessions[0].start)} \\u2192 ${dateOf(DATA.sessions[DATA.sessions.length - 1].end)}`
  : "no sessions recorded yet";

if (DATA.served) renderLogForm();
if (!DATA.summary.n) {
  if (!DATA.served) {
    const card = el("div", "card empty");
    card.appendChild(el("div", null, "Log sessions from the page by running the local server:"));
    card.appendChild(el("code", null, "python scripts/session_server.py"));
    card.appendChild(el("div", "sub", "then open http://127.0.0.1:8765 \\u2014 or use the terminal: python scripts/session.py start / end"));
    content.appendChild(card);
  }
} else {
  renderTiles();
  renderChart();
  renderTable();
  if (DATA.hasHH) renderPerStake();
  if (DATA.hasGrades) renderAnalysis();
}
if (DATA.hasPopulation) renderPopulation();  // pool census, independent of the ledger

function renderLogForm() {
  const card = el("div", "card");
  card.id = "logCard";
  card.appendChild(el("h2", null, "Log a session"));

  const srcLabel = el("label", null, "session");
  const select = el("select");
  srcLabel.appendChild(select);
  card.appendChild(srcLabel);
  const auto = el("div", "auto", "loading detected sessions\\u2026");
  card.appendChild(auto);

  const fields = [
    ["rakeback_balance_start", "rakeback balance at start", "rec"],
    ["rakeback_balance_end", "rakeback balance at end", "rec"],
    ["rake_paid_start", "rake-paid counter at start", ""],
    ["rake_paid_end", "rake-paid counter at end", ""],
    ["start_balance", "main balance at start", ""],
    ["end_balance", "main balance at end", ""],
    ["rakeback_claimed", "rakeback claimed mid-session", ""],
    ["deposits", "deposits", ""],
    ["withdrawals", "withdrawals", ""],
    ["start_time", "start time (YYYY/MM/DD HH:MM:SS)", ""],
    ["end_time", "end time", ""],
    ["notes", "notes", ""],
  ];
  const grid = el("div", "grid");
  const inputs = {};
  for (const [name, labelText, cls] of fields) {
    const label = el("label", cls || null, labelText);
    const input = el("input");
    input.type = "text";
    input.autocomplete = "off";
    if (["rakeback_claimed", "deposits", "withdrawals"].includes(name)) input.placeholder = "0";
    inputs[name] = input;
    label.appendChild(input);
    grid.appendChild(label);
  }
  card.appendChild(grid);

  const actions = el("div", "actions");
  const save = el("button", null, "Save session");
  const err = el("span", "err", "");
  const uploadHint = el("span", "uploadHint");
  const uploadLink = el("a", null, "upload a new hand-history export");
  const fileInput = el("input");
  fileInput.type = "file";
  fileInput.accept = ".txt";
  uploadHint.appendChild(uploadLink);
  uploadHint.appendChild(fileInput);
  actions.appendChild(save);
  actions.appendChild(err);
  actions.appendChild(uploadHint);
  card.appendChild(actions);
  content.appendChild(card);

  let proposals = [];
  let selected = null; // the chosen proposal, or null for manual entry

  function refreshProposals() {
    fetch("/api/detect").then(r => r.json()).then(data => {
      proposals = data.proposals || [];
      select.replaceChildren();
      const manual = el("option", null, "manual entry (type the times yourself)");
      manual.value = "-1";
      select.appendChild(manual);
      proposals.forEach((p, i) => {
        const opt = el("option", null,
          `${dateOf(p.start)} ${hhmm(p.start)}\\u2013${hhmm(p.end)} \\u00b7 ${p.hands} hands \\u00b7 net ${money(p.hhNet)}`);
        opt.value = String(i);
        select.appendChild(opt);
      });
      if (proposals.length) { select.value = "0"; applySelection(); }
      else { select.value = "-1"; applySelection(); auto.textContent = "no unlogged sessions found in data/*.txt \\u2014 upload an export or enter times manually"; }
    }).catch(() => { auto.textContent = "could not reach /api/detect"; });
  }

  function applySelection() {
    const i = parseInt(select.value);
    selected = i >= 0 ? proposals[i] : null;
    if (selected) {
      inputs.start_time.value = selected.start;
      inputs.end_time.value = selected.end;
      auto.textContent =
        `auto from hand history: table result ${money(selected.hhNet)} \\u00b7 ${selected.hands} hands` +
        ` \\u00b7 stakes ${selected.stakes}` +
        (selected.excluded ? ` \\u00b7 ${selected.excluded} hands excluded (unparsed actions)` : "");
    } else {
      auto.textContent = "manual entry: table result will come from your balance readings (enter both)";
    }
  }
  select.addEventListener("change", applySelection);

  async function uploadFile(file) {
    if (!file) return;
    auto.textContent = `uploading ${file.name}\\u2026`;
    const resp = await fetch("/api/upload", { method: "POST",
      headers: { "X-Filename": encodeURIComponent(file.name) }, body: await file.text() });
    const data = await resp.json();
    if (data.error) { err.textContent = data.error; auto.textContent = ""; return; }
    err.textContent = "";
    refreshProposals();
  }

  uploadLink.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));
  // Drag-and-drop an export anywhere on the page -- same path as the picker link.
  for (const evt of ["dragover", "dragenter"]) {
    document.body.addEventListener(evt, e => e.preventDefault());
  }
  document.body.addEventListener("drop", e => {
    e.preventDefault();
    uploadFile(e.dataTransfer.files && e.dataTransfer.files[0]);
  });

  save.addEventListener("click", async () => {
    const num = name => inputs[name].value.trim() === "" ? null : inputs[name].value.trim();
    const body = {
      start_time: inputs.start_time.value.trim(),
      end_time: inputs.end_time.value.trim(),
      start_balance: num("start_balance"),
      end_balance: num("end_balance"),
      rakeback_balance_start: num("rakeback_balance_start"),
      rakeback_balance_end: num("rakeback_balance_end"),
      rakeback_claimed: num("rakeback_claimed"),
      deposits: num("deposits"),
      withdrawals: num("withdrawals"),
      rake_paid_start: num("rake_paid_start"),
      rake_paid_end: num("rake_paid_end"),
      notes: inputs.notes.value,
      stakes_note: selected ? selected.stakes : "",
      hands_played: selected ? selected.hands : null,
      hh_table_result: selected ? selected.hhNet : null,
    };
    err.textContent = "";
    const resp = await fetch("/api/log", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await resp.json();
    if (data.error) { err.textContent = data.error; return; }
    location.reload();
  });

  refreshProposals();
}

function renderTiles() {
  const s = DATA.summary;
  const tiles = el("div", "tiles");
  const tile = (label, value, hint) => {
    const t = el("div", "card tile");
    t.appendChild(el("div", "label", label));
    t.appendChild(el("div", "value", value));
    if (hint) t.appendChild(el("div", "hint", hint));
    tiles.appendChild(t);
  };
  tile("Combined result", money(s.total), "table + rakeback");
  tile("Table result", money(s.table));
  tile("Rakeback earned", money(s.rakeback),
       s.rakebackMissing ? `unmeasured in ${s.rakebackMissing} session(s)` : undefined);
  tile("Effective rakeback rate \\u03c1",
       s.rhoMean === null ? "\\u2014" : s.rhoMean.toFixed(3),
       s.rhoMean === null
         ? "record the rake-paid counter at start/end to measure it"
         : (s.rhoSe !== null ? `\\u00b1 ${s.rhoSe.toFixed(3)} (1 SE) \\u00b7 ${s.rhoN} session(s)` : `${s.rhoN} session(s)`));
  content.appendChild(tiles);
}

function niceStep(raw) {
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (raw <= m * mag) return m * mag;
  return 10 * mag;
}

function renderChart() {
  const rows = DATA.sessions, n = rows.length;
  const card = el("div", "card section");
  card.id = "chartCard";
  card.appendChild(el("h2", null, "Bankroll curve (cumulative)"));
  const legend = el("div", "legend");
  for (const [name, color] of [["total (table + rakeback)", "var(--s-total)"], ["table only", "var(--s-table)"]]) {
    const item = el("span");
    const key = el("span", "key");
    key.style.borderTopColor = color;
    item.appendChild(key);
    item.appendChild(document.createTextNode(name));
    legend.appendChild(item);
  }
  card.appendChild(legend);

  const W = 900, H = 300, L = 60, R = 110, T = 14, B = 34;
  const xs = i => n === 1 ? (L + (W - L - R) / 2) : L + i * (W - L - R) / (n - 1);
  const vals = rows.flatMap(r => [r.cumTotal, r.cumTable]).concat([0]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi === lo) { hi += 1; lo -= 1; }
  const step = niceStep((hi - lo) / 4);
  lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const ys = v => T + (hi - v) * (H - T - B) / (hi - lo);

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("tabindex", "0");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Cumulative bankroll by session; values in the table below");
  const add = (parent, tag, attrs) => {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    parent.appendChild(node);
    return node;
  };

  for (let v = lo; v <= hi + 1e-9; v += step) {
    add(svg, "line", { x1: L, x2: W - R, y1: ys(v), y2: ys(v),
      stroke: v === 0 ? "var(--baseline)" : "var(--hairline)", "stroke-width": 1 });
    const t = add(svg, "text", { x: L - 8, y: ys(v) + 4, "text-anchor": "end",
      "font-size": 11, fill: "var(--ink-muted)", style: "font-variant-numeric: tabular-nums" });
    t.textContent = C + v.toLocaleString();
  }
  const tickEvery = Math.max(1, Math.ceil(n / 6));
  rows.forEach((r, i) => {
    if (i % tickEvery && i !== n - 1) return;
    const t = add(svg, "text", { x: xs(i), y: H - B + 18, "text-anchor": "middle",
      "font-size": 11, fill: "var(--ink-muted)" });
    t.textContent = dateOf(r.start).slice(5);
  });

  const series = [
    { key: "cumTotal", color: "var(--s-total)", label: "total" },
    { key: "cumTable", color: "var(--s-table)", label: "table only" },
  ];
  for (const s of series) {
    add(svg, "polyline", {
      points: rows.map((r, i) => `${xs(i)},${ys(r[s.key])}`).join(" "),
      fill: "none", stroke: s.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    });
    rows.forEach((r, i) => add(svg, "circle", { cx: xs(i), cy: ys(r[s.key]), r: 4,
      fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }));
  }
  // Direct end labels (secondary ink, identity from the colored key drawn beside
  // the text, never colored text) -- nudged apart if the two ends converge.
  const last = rows[n - 1];
  let eyTotal = ys(last.cumTotal), eyTable = ys(last.cumTable);
  if (Math.abs(eyTotal - eyTable) < 16) {
    const mid = (eyTotal + eyTable) / 2, sign = eyTotal <= eyTable ? 1 : -1;
    eyTotal = mid - 8 * sign; eyTable = mid + 8 * sign;
  }
  for (const [y, s] of [[eyTotal, series[0]], [eyTable, series[1]]]) {
    add(svg, "line", { x1: W - R + 8, x2: W - R + 22, y1: y, y2: y,
      stroke: s.color, "stroke-width": 2, "stroke-linecap": "round" });
    const t = add(svg, "text", { x: W - R + 27, y: y + 4, "font-size": 11, fill: "var(--ink-2)" });
    t.textContent = s.label;
  }

  const cross = add(svg, "line", { y1: T, y2: H - B, stroke: "var(--baseline)",
    "stroke-width": 1, visibility: "hidden" });
  const hit = add(svg, "rect", { x: L, y: T, width: W - L - R, height: H - T - B,
    fill: "transparent" });

  const wrap = el("div"); wrap.id = "plotWrap";
  const tip = el("div"); tip.id = "tip";
  wrap.appendChild(svg); wrap.appendChild(tip);
  card.appendChild(wrap);
  content.appendChild(card);

  let focusIdx = -1;
  function showAt(i, pointerX) {
    const r = rows[i];
    cross.setAttribute("x1", xs(i)); cross.setAttribute("x2", xs(i));
    cross.setAttribute("visibility", "visible");
    tip.replaceChildren();
    tip.appendChild(el("div", "when", `session ${i + 1} \\u00b7 ${dateOf(r.start)} ${hhmm(r.start)}\\u2013${hhmm(r.end)}`));
    for (const s of series) {
      const row = el("div", "row");
      const key = el("span", "key"); key.style.borderTopColor = s.color;
      row.appendChild(key);
      row.appendChild(el("b", null, money(r[s.key])));
      row.appendChild(el("span", null, s.label));
      tip.appendChild(row);
    }
    tip.appendChild(el("div", "own",
      `this session: table ${money(r.table)} \\u00b7 rakeback ${moneyOrDash(r.rakeback)}` +
      (r.rho !== null ? ` \\u00b7 \\u03c1 ${r.rho.toFixed(2)}` : "")));
    tip.style.display = "block";
    const rect = wrap.getBoundingClientRect();
    const px = (pointerX !== undefined ? pointerX : xs(i) * rect.width / W);
    const flip = px > rect.width * 0.62;
    tip.style.left = flip ? "" : (px + 14) + "px";
    tip.style.right = flip ? (rect.width - px + 14) + "px" : "";
    tip.style.top = "22px";
  }
  function hide() { cross.setAttribute("visibility", "hidden"); tip.style.display = "none"; focusIdx = -1; }

  hit.addEventListener("pointermove", e => {
    const rect = wrap.getBoundingClientRect();
    const vx = (e.clientX - rect.left) * W / rect.width;
    let best = 0;
    for (let i = 1; i < n; i++) if (Math.abs(xs(i) - vx) < Math.abs(xs(best) - vx)) best = i;
    showAt(best, e.clientX - rect.left);
  });
  hit.addEventListener("pointerleave", hide);
  svg.addEventListener("keydown", e => {
    if (e.key === "ArrowRight") { focusIdx = Math.min(n - 1, focusIdx + 1); showAt(focusIdx); e.preventDefault(); }
    if (e.key === "ArrowLeft") { focusIdx = focusIdx < 0 ? n - 1 : Math.max(0, focusIdx - 1); showAt(focusIdx); e.preventDefault(); }
    if (e.key === "Escape") hide();
  });
  svg.addEventListener("blur", hide);
}

function renderTable() {
  const card = el("div", "card section");
  card.appendChild(el("h2", null, "Sessions"));
  const table = el("table");
  const thead = el("thead"); const hr = el("tr");
  for (const [name, numeric] of [["date", 0], ["time", 0], ["table", 1], ["rakeback", 1],
                                 ["total", 1], ["\\u03c1", 1], ["stakes", 0], ["notes", 0]]) {
    const cls = (numeric ? "num" : "") + (name === "\\u03c1" ? " sym" : "");
    const th = el("th", cls.trim() || null, name); hr.appendChild(th);
  }
  thead.appendChild(hr); table.appendChild(thead);
  const tbody = el("tbody");
  for (const r of DATA.sessions) {
    const tr = el("tr");
    tr.appendChild(el("td", "txt", dateOf(r.start)));
    tr.appendChild(el("td", "txt", `${hhmm(r.start)}\\u2013${hhmm(r.end)}`));
    for (const v of [r.table, r.rakeback, r.total]) tr.appendChild(el("td", "num", moneyOrDash(v)));
    tr.appendChild(el("td", "num", r.rho !== null ? r.rho.toFixed(3) : ""));
    tr.appendChild(el("td", "txt", r.stakes));
    tr.appendChild(el("td", "txt", r.notes));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  card.appendChild(table);
  content.appendChild(card);
}

function renderPerStake() {
  const section = el("div", "section");
  section.appendChild(el("h2", null, "Per-stake (hand-history join)"));
  for (const r of DATA.sessions) {
    const card = el("div", "card stakeCard");
    card.appendChild(el("div", "head",
      `${dateOf(r.start)} ${hhmm(r.start)}\\u2013${hhmm(r.end)}` +
      (r.perStake ? ` \\u00b7 ${r.perStake.hands} hands` : "")));
    if (!r.perStake) {
      card.appendChild(el("div", "xcheck", "no hands in the export for this window"));
      section.appendChild(card); continue;
    }
    const table = el("table"); const hr = el("tr");
    for (const [name, numeric] of [["stake", 0], ["hands", 1], ["net", 1], ["net bb", 1],
                                   ["bb/100", 1], ["excluded (unparsed)", 1]]) {
      hr.appendChild(el("th", numeric ? "num" : null, name));
    }
    table.appendChild(hr);
    for (const st of r.perStake.stakes) {
      const tr = el("tr");
      tr.appendChild(el("td", "txt", `${st.bb}bb`));
      tr.appendChild(el("td", "num", String(st.hands)));
      tr.appendChild(el("td", "num", money(st.net)));
      tr.appendChild(el("td", "num", (st.netBb < 0 ? "\\u2212" : "+") + Math.abs(st.netBb).toFixed(1)));
      tr.appendChild(el("td", "num", st.bbPer100 !== null ? st.bbPer100.toFixed(1) : "\\u2014"));
      tr.appendChild(el("td", "num", st.excluded ? String(st.excluded) : ""));
      table.appendChild(tr);
    }
    card.appendChild(table);
    if (r.perStake.residual === null) {
      card.appendChild(el("div", "xcheck",
        "table result sourced from hand history (no balance readings to cross-check)"));
    } else {
      const bad = Math.abs(r.perStake.residual) > 1.0;
      card.appendChild(el("div", "xcheck" + (bad ? " warn" : ""),
        (bad ? "\\u26a0 " : "") +
        `cross-check: balance table result ${money(r.table)} vs HH net ${money(r.perStake.hhNet)}` +
        ` (residual ${money(r.perStake.residual)})` +
        (bad ? " \\u2014 likely causes: a reward credited straight to the main balance, " +
               "a balance read while still seated at a table, an unentered claim/deposit, " +
               "or an export missing tables" : "")));
    }
    section.appendChild(card);
  }
  content.appendChild(section);
}

function renderAnalysis() {
  const section = el("div", "section");
  section.appendChild(el("h2", null, "Push/fold analysis (GTO grade & luck)"));
  const act = a => a === "shove_or_call" ? "shove/call" : a;
  const signedBb = v => (v < 0 ? "\\u2212" : "+") + Math.abs(v).toFixed(2);
  for (const r of DATA.sessions) {
    const card = el("div", "card stakeCard");
    const a = r.analysis;
    card.appendChild(el("div", "head",
      `${dateOf(r.start)} ${hhmm(r.start)}\\u2013${hhmm(r.end)}` +
      (a ? ` \\u00b7 ${a.spots} push/fold spots \\u00b7 accuracy ${a.accuracyPct.toFixed(1)}%` +
           ` \\u00b7 EV lost ${C}${a.evLostCurrency.toFixed(2)}` : "")));
    if (!a) {
      card.appendChild(el("div", "xcheck", "no gradeable push/fold spots in this session's window"));
      section.appendChild(card); continue;
    }
    if (a.luckSpots) {
      const dir = a.luckGapCurrency >= 0 ? "ran hot" : "ran cold";
      card.appendChild(el("div", "xcheck",
        `luck check (${a.luckSpots} of ${a.spots} spots): model predicted ${money(a.predictedCurrency)}` +
        ` from your decisions \\u00b7 realized ${money(a.realizedCurrency)}` +
        ` \\u00b7 luck gap ${money(a.luckGapCurrency)} (${dir})`));
    }
    if (r.perStake) {
      card.appendChild(el("div", "xcheck",
        `graded hands net ${money(a.gradedNetCurrency)} \\u00b7 ungraded hands net ` +
        `${money(r.perStake.hhNet - a.gradedNetCurrency)} (walks/limps/non-push-fold: no model EV)`));
    }
    const table = el("table"); const hr = el("tr");
    for (const [name, numeric] of [["stake", 0], ["spots", 1], ["matched", 1],
                                   ["EV lost bb", 1], ["predicted bb", 1], ["realized bb", 1]]) {
      hr.appendChild(el("th", numeric ? "num" : null, name));
    }
    table.appendChild(hr);
    for (const st of a.stakes) {
      const tr = el("tr");
      tr.appendChild(el("td", "txt", `${st.bb}bb`));
      tr.appendChild(el("td", "num", String(st.spots)));
      tr.appendChild(el("td", "num", `${st.matched}/${st.spots}`));
      tr.appendChild(el("td", "num", st.evLostBb.toFixed(2)));
      tr.appendChild(el("td", "num", st.luckSpots ? signedBb(st.predictedBb) : "\\u2014"));
      tr.appendChild(el("td", "num", st.luckSpots ? signedBb(st.realizedBb) : "\\u2014"));
      table.appendChild(tr);
    }
    card.appendChild(table);
    if (a.mistakes.length) {
      const mt = el("table"); const mh = el("tr");
      for (const [name, numeric] of [["costliest deviations", 0], ["pos", 0], ["stack", 1],
                                     ["hero", 0], ["GTO (weight)", 0], ["loss bb", 1], ["stake", 0]]) {
        mh.appendChild(el("th", numeric ? "num" : null, name));
      }
      mt.appendChild(mh);
      for (const m of a.mistakes) {
        const tr = el("tr");
        tr.appendChild(el("td", "txt", m.hand));
        tr.appendChild(el("td", "txt", `${m.pos} (${m.model})`));
        tr.appendChild(el("td", "num", `${m.stackBb}bb`));
        tr.appendChild(el("td", "txt", act(m.hero)));
        tr.appendChild(el("td", "txt", `${act(m.gto)} (${m.weight.toFixed(2)})`));
        tr.appendChild(el("td", "num", m.lossBb.toFixed(3)));
        tr.appendChild(el("td", "txt", `${m.bb}bb`));
        mt.appendChild(tr);
      }
      card.appendChild(mt);
    }
    section.appendChild(card);
  }
  content.appendChild(section);
}

function renderPopulation() {
  const p = DATA.population;
  const section = el("div", "section");
  section.appendChild(el("h2", null, "Population vs GTO (all scanned hands)"));
  const card = el("div", "card stakeCard");
  card.appendChild(el("div", "xcheck",
    `${p.decisions.toLocaleString()} villain decisions across ${p.hands.toLocaleString()} hands` +
    ` \\u00b7 IDs rotate every hand \\u2014 population-level only`));

  const stakes = [...new Set(p.rows.flatMap(r => r.cells.filter(c => c.bb !== null).map(c => c.bb)))]
    .sort((a, b) => a - b);
  const label = el("label", "stakeSel", "stake");
  const select = el("select");
  const optAll = el("option", null, "all stakes");
  optAll.value = "";
  select.appendChild(optAll);
  for (const bb of stakes) {
    const o = el("option", null, `${bb}bb`);
    o.value = String(bb);
    select.appendChild(o);
  }
  label.appendChild(select);
  card.appendChild(label);

  const tableWrap = el("div");
  card.appendChild(tableWrap);
  function draw() {
    const bb = select.value === "" ? null : parseFloat(select.value);
    const table = el("table");
    const hr = el("tr");
    for (const [name, numeric] of [["info set", 0], ["situation", 0], ["n", 1],
                                   ["pool go % \\u00b1 SE", 1], ["GTO %", 1], ["z", 1]]) {
      hr.appendChild(el("th", numeric ? "num" : null, name));
    }
    table.appendChild(hr);
    for (const r of p.rows) {
      const cell = r.cells.find(c => c.bb === bb);
      if (!cell || !cell.n) continue;
      const tr = el("tr");
      tr.appendChild(el("td", "txt", r.key));
      tr.appendChild(el("td", "txt", r.desc));
      tr.appendChild(el("td", "num", String(cell.n)));
      tr.appendChild(el("td", "num", cell.goPct === null ? "\\u2014"
        : `${cell.goPct.toFixed(1)}%` + (cell.sePct !== null ? ` \\u00b1 ${cell.sePct.toFixed(1)}` : "")));
      tr.appendChild(el("td", "num", r.gtoPct === null ? "\\u2014" : `${r.gtoPct.toFixed(1)}%`));
      tr.appendChild(el("td", "num", cell.z === null ? "\\u2014"
        : (Math.abs(cell.z) >= 2 ? "\\u26a0 " : "") + cell.z.toFixed(1)));
      table.appendChild(tr);
    }
    tableWrap.replaceChildren(table);
  }
  select.addEventListener("change", draw);
  draw();

  const shownRows = p.rows.filter(r => r.shown);
  if (shownRows.length) {
    const st = el("table");
    const sh = el("tr");
    for (const [name, numeric] of [["shown-down shoves", 0], ["n", 1], ["inside GTO range", 1],
                                   ["top out-of-range shoves", 0]]) {
      sh.appendChild(el("th", numeric ? "num" : null, name));
    }
    st.appendChild(sh);
    for (const r of shownRows) {
      const tr = el("tr");
      tr.appendChild(el("td", "txt", r.key));
      tr.appendChild(el("td", "num", String(r.shown.n)));
      tr.appendChild(el("td", "num", `${r.shown.inGtoPct.toFixed(1)}%`));
      tr.appendChild(el("td", "txt",
        r.shown.top.map(([h, c]) => `${h}\\u00d7${c}`).join(" \\u00b7 ") || "\\u2014"));
      st.appendChild(tr);
    }
    card.appendChild(st);
    card.appendChild(el("div", "xcheck",
      "biased sample: cards appear only at showdowns \\u2014 uncalled steals and folded hands are invisible"));
  }
  section.appendChild(card);
  content.appendChild(section);
}
</script>
</body>
</html>
"""
