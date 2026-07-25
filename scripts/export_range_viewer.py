#!/usr/bin/env python3
"""Export the solved 4-max ranges as a self-contained interactive HTML viewer.

The production PNG (viz.plot_fourmax_ranges) crams all 14 heatmaps into one image --
fine as a record, useless for quick reference. This script emits two variants of the
same page from one template:

  cache/range_viewer.html   full standalone document (open locally in any browser)
  --artifact PATH           body-only fragment for publishing as a standalone HTML artifact
                            (the artifact host supplies the <!DOCTYPE>/<head>/<body>
                            skeleton, so the fragment must not) -- this is the
                            view-it-on-your-phone path.

Cell encoding: each cell is filled left-to-right by its shove/call weight -- the red
fill occupies exactly weight% of the cell width over a neutral "fold" base. That makes
mixed-strategy cells readable at a glance (a 30% call literally looks 30% full) and,
unlike the PNG's red-yellow-green interpolation, doesn't rely on a red-vs-green hue
pair (the classic colorblind-unsafe combination): the value is carried by geometry,
color just makes it pop. Red still means shove/call, matching the PNG's salient half.

Theming is token-level: palette custom properties on :root, redefined under
@media (prefers-color-scheme: dark) AND under :root[data-theme="dark"|"light"] so the
artifact viewer's manual theme toggle overrides the OS preference in both directions.

Rerun after any re-solve: python scripts/export_range_viewer.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
import tree
import viz
from game import FOURMAX_CONFIG, NINEMAX_CONFIG


# Moved to src/tree.py (pure structure -> text, needed by dashboard code that must
# not import viz/matplotlib); alias kept for existing callers of this module.
infoset_description = tree.infoset_description

# Registry so one script serves any solved config. Each entry: the GameConfig, the
# default solved-ranges pickle, and the human labels used in the page title/heading.
CONFIGS = {
    "fourmax": {"cfg": FOURMAX_CONFIG, "ranges": "cache/fourmax_ranges.pkl",
                "title": "4-max Push/Fold Ranges", "heading": "4-max push/fold"},
    "ninemax": {"cfg": NINEMAX_CONFIG, "ranges": "cache/ninemax_ranges.pkl",
                "title": "9-max Push/Fold Ranges", "heading": "9-max push/fold"},
}


def build_payload(pkl_path: str, cfg) -> dict:
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    ranges, info = saved["ranges"], saved.get("info", {})

    # Label grid straight from the solver's own 13x13 convention (viz._GRID_CANON_INDEX):
    # rows/cols A..2, upper-right suited, lower-left offsuit, diagonal pairs.
    labels = [[cards.canon_label(viz._GRID_CANON_INDEX[r, c]) for c in range(13)] for r in range(13)]

    infosets = []
    for i in tree.build_infosets(cfg):
        key = tree.infoset_key(i.seat, i.shoved_before)
        vec = ranges[key]
        shove_pct = 100 * sum(vec[j] * cards.CANON_COMBO_WEIGHT[j] for j in range(169)) / cards.NUM_COMBOS
        infosets.append({
            "key": key,
            "seat": i.seat,
            "action": "call" if i.shoved_before else "shove",
            # How many opponents already shoved before hero's decision. Drives the
            # "action faced" filter (0 = open, 1 = facing one shove, 2+ = multiway) so
            # 9-max's 510 info sets stay navigable -- the common opens + facing-one are
            # shown first, the rare deep multiway nodes filtered out by default.
            "faced": len(i.shoved_before),
            "desc": infoset_description(cfg, i),
            "shovePct": round(shove_pct, 1),
            "grid": [[round(float(w), 4) for w in row] for row in viz.range_to_grid(vec)],
        })

    solved_on = datetime.date.fromtimestamp(Path(pkl_path).stat().st_mtime).isoformat()
    meta_bits = [f"solved {solved_on}"]
    if "exploitability_hi_mc" in info:
        meta_bits.append(
            f"exploitability {info['exploitability_hi_mc']:.4f} bb "
            f"({info['exploitability_hi_mc_2x']:.4f} at 2x MC)"
        )
    if "iterations" in info:
        meta_bits.append(f"{info['iterations']} sweeps")

    # Table-selector metadata: seats in action order, which seats post blinds (for
    # labels), and the dealer seat (the one immediately before the first blind). Lets the
    # viewer render a round poker table generically for any config, not just 9-max.
    seats = list(cfg.seats)
    blind_positions = [idx for idx, s in enumerate(seats) if s in cfg.blinds]
    button = seats[min(blind_positions) - 1] if blind_positions and min(blind_positions) > 0 else None

    return {
        "labels": labels,
        "infosets": infosets,
        "meta": " · ".join(meta_bits),
        "seats": seats,
        "blinds": dict(cfg.blinds),
        "button": button,
    }


# Palette: the dataviz reference tokens validated for this viewer (shove red 3.85:1
# light / 5.39:1 dark vs surface; all label inks >= 3.9:1 on their fills). Defined
# once per mode and interpolated into :root, the dark media query, and both
# data-theme override blocks so the artifact theme toggle wins in either direction.
LIGHT_VARS = """\
  --surface: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
  --hairline: #e1e0d9; --ring: rgba(11,11,11,0.10);
  --fold: #f0efec; --shove: #e34948; --shove-ink: #ffffff;
  --tab-active: #0b0b0b; --tab-active-ink: #fcfcfb;
  --felt: #d3e5da; --felt-edge: #9cc0ab;"""

DARK_VARS = """\
  --surface: #1a1a19; --page: #0d0d0d;
  --ink: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
  --hairline: #2c2c2a; --ring: rgba(255,255,255,0.10);
  --fold: #383835; --shove: #e66767; --shove-ink: #0b0b0b;
  --tab-active: #ffffff; --tab-active-ink: #0d0d0d;
  --felt: #1d2b24; --felt-edge: #334c3e;"""

# The page content: everything inside <body>, wrapped in #rv-root rather than styling
# <body> itself, so the same markup drops cleanly into the artifact host's skeleton.
PAGE_CONTENT = """<style>
:root {
__LIGHT_VARS__
}
@media (prefers-color-scheme: dark) { :root {
__DARK_VARS__
} }
:root[data-theme="light"] {
__LIGHT_VARS__
}
:root[data-theme="dark"] {
__DARK_VARS__
}
#rv-root, #rv-root * { box-sizing: border-box; margin: 0; }
#rv-root {
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page); color: var(--ink);
  display: flex; min-height: 100vh;
}
#rv-root nav {
  width: 232px; flex: none; padding: 16px 10px 24px;
  border-right: 1px solid var(--hairline); overflow-y: auto;
}
#rv-root nav h1 { font-size: 15px; padding: 2px 8px 10px; }
#filter { display: flex; flex-wrap: wrap; gap: 4px; padding: 0 8px 8px; }
#filter button {
  flex: none; width: auto; padding: 4px 9px; margin: 0; border: 1px solid var(--hairline);
  border-radius: 999px; background: none; color: var(--ink-2); font: inherit;
  font-size: 12px; cursor: pointer;
}
#filter button.active { background: var(--tab-active); color: var(--tab-active-ink);
  border-color: var(--tab-active); }
#rv-root nav .seat { font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
  color: var(--ink-muted); text-transform: uppercase; padding: 12px 8px 4px; }
#rv-root nav button {
  display: flex; justify-content: space-between; gap: 8px; width: 100%;
  padding: 6px 8px; margin: 1px 0; border: 0; border-radius: 6px;
  background: none; color: var(--ink-2); font: inherit; font-size: 13px;
  text-align: left; cursor: pointer;
}
#rv-root nav button:hover { background: var(--fold); }
#rv-root nav button:focus-visible { outline: 2px solid var(--ink); outline-offset: 1px; }
#rv-root nav button.active { background: var(--tab-active); color: var(--tab-active-ink); }
#rv-root nav button .pct { font-variant-numeric: tabular-nums; opacity: 0.75; }
#rv-root main { flex: 1; padding: 22px 28px; min-width: 0; }
#rv-root header h2 { font-size: 18px; }
#rv-root header .key { color: var(--ink-muted); font-size: 13px; margin-top: 2px; }
#rv-root header .stat { margin-top: 6px; font-size: 14px; color: var(--ink-2); }
#rv-root header .stat b { color: var(--ink); font-size: 16px; }
/* Round-table selector: tap your seat, then tap who has already shoved. */
#picker { max-width: 720px; margin-bottom: 6px; }
#table {
  position: relative; width: 100%; max-width: 430px; aspect-ratio: 1.32 / 1;
  margin: 2px auto 4px;
}
#table .felt {
  position: absolute; inset: 7% 4%; border-radius: 50%;
  background: var(--felt); border: 2px solid var(--felt-edge);
}
.seatbtn {
  position: absolute; transform: translate(-50%, -50%);
  min-width: 54px; padding: 7px 6px; border: 1.5px solid var(--hairline);
  border-radius: 11px; background: var(--surface); color: var(--ink);
  font: 600 12px/1.15 inherit; text-align: center; cursor: pointer;
  box-shadow: 0 1px 3px var(--ring); user-select: none;
}
.seatbtn .role { display: block; font-size: 8.5px; font-weight: 600; letter-spacing: .04em;
  color: var(--ink-muted); text-transform: uppercase; margin-top: 1px; }
.seatbtn.hero { background: var(--tab-active); color: var(--tab-active-ink); border-color: var(--tab-active); }
.seatbtn.hero .role { color: var(--tab-active-ink); opacity: .8; }
.seatbtn.shoved { background: var(--shove); color: var(--shove-ink); border-color: var(--shove); }
.seatbtn.shoved .role { color: var(--shove-ink); opacity: .85; }
.seatbtn.folded { opacity: .45; }
.seatbtn.toact { opacity: .5; border-style: dashed; }
.seatbtn:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
#tableinfo {
  display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: baseline; justify-content: center;
  text-align: center; font-size: 13px; color: var(--ink-2); margin: 2px 0 4px; min-height: 20px;
}
#tableinfo b { color: var(--ink); }
#tablereset {
  border: 1px solid var(--hairline); border-radius: 999px; background: none;
  color: var(--ink-2); font: inherit; font-size: 12px; padding: 3px 10px; cursor: pointer;
}
#tablereset:hover { background: var(--fold); }
#wrap { margin-top: 14px; max-width: 720px; }
#grid {
  /* minmax(0,1fr) not bare 1fr: with aspect-ratio cells, 1fr's implicit auto
     minimum gives tracks a floor that overflows narrow viewports. */
  display: grid; grid-template-columns: 20px repeat(13, minmax(0, 1fr));
  gap: 2px; background: var(--surface); padding: 10px;
  border: 1px solid var(--ring); border-radius: 10px;
}
.cell, #grid .hdr { min-width: 0; }
#grid .hdr { display: flex; align-items: center; justify-content: center;
  font-size: 11px; color: var(--ink-muted); }
.cell {
  position: relative; aspect-ratio: 1; border-radius: 4px;
  background: var(--fold); overflow: hidden; cursor: default;
}
.cell .fill { position: absolute; inset: 0; width: calc(var(--w) * 100%);
  background: var(--shove); }
.cell span { position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  font-size: clamp(8px, 1.15vw, 12px); font-weight: 600; color: var(--ink-2); }
.cell.hi span { color: var(--shove-ink); }
.cell span small { font-weight: 500; font-size: 0.8em; opacity: 0.85; }
.cell:hover, .cell.sel { outline: 2px solid var(--ink); outline-offset: -2px; z-index: 1; }
#detail { margin-top: 10px; min-height: 22px; font-size: 14px; color: var(--ink-2); }
#detail b { color: var(--ink); }
#legend { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center;
  margin-top: 8px; font-size: 12px; color: var(--ink-muted); }
#legend .sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
  margin-right: 5px; vertical-align: -2px; border: 1px solid var(--ring); }
#rv-root footer { margin-top: 14px; font-size: 12px; color: var(--ink-muted); }
@media (max-width: 760px) {
  #rv-root { flex-direction: column; }
  /* Nav becomes a sticky, thumb-scrollable chip strip. */
  #rv-root nav {
    width: auto; display: flex; flex-wrap: nowrap; align-items: center; gap: 4px;
    overflow-x: auto; -webkit-overflow-scrolling: touch;
    position: sticky; top: 0; z-index: 2;
    padding: 8px 10px; background: var(--page);
    border-right: 0; border-bottom: 1px solid var(--hairline);
  }
  #rv-root nav h1 { display: none; }
  #rv-root nav .seat { flex: none; padding: 0 2px 0 8px; }
  #rv-root nav button {
    flex: none; width: auto; border: 1px solid var(--hairline); border-radius: 999px;
    padding: 6px 11px; white-space: nowrap;
  }
  #rv-root main { padding: 14px 12px 24px; }
  #table { max-width: 340px; aspect-ratio: 1.22 / 1; }
  .seatbtn { min-width: 46px; padding: 6px 4px; font-size: 11px; border-radius: 9px; }
  .seatbtn .role { font-size: 8px; }
  #grid { gap: 1px; padding: 6px; grid-template-columns: 15px repeat(13, minmax(0, 1fr)); }
  #grid .hdr { font-size: 9px; }
  .cell span { font-size: clamp(6.5px, 2.1vw, 11px); }
  .cell { border-radius: 3px; }
}
</style>
<div id="rv-root">
<nav id="nav"><h1>__HEADING__</h1><div id="filter"></div></nav>
<main>
  <div id="picker">
    <div id="table"><div class="felt"></div></div>
    <div id="tableinfo"></div>
  </div>
  <header>
    <h2 id="desc"></h2>
    <div class="key" id="key"></div>
    <div class="stat">Range: <b id="pct"></b> of combos</div>
  </header>
  <div id="wrap">
    <div id="grid"></div>
    <div id="detail">Tap or hover a hand for its exact weight.</div>
    <div id="legend">
      <span><span class="sw" style="background:var(--shove)"></span>shove / call (fill = weight)</span>
      <span><span class="sw" style="background:var(--fold)"></span>fold</span>
      <span>rows/cols A&#8594;2 &#183; upper right suited &#183; diagonal pairs &#183; lower left offsuit</span>
    </div>
    <footer id="meta"></footer>
  </div>
</main>
</div>
<script>
const DATA = __DATA__;
const RANKS = "AKQJT98765432";
const nav = document.getElementById("nav");
const filterBar = document.getElementById("filter");
let current = 0;
let selectedCell = null;
let tabs = [];

// --- Round-table selector: tap your seat, then tap who has already shoved --------
// The solved key for (hero, shovers) is exactly `hero|<shovers sorted alphabetically>`
// (see tree.infoset_key), so a table selection resolves to an info set by string match
// -- no scrolling the 510-item list.
const SEATS = DATA.seats, BLINDS = DATA.blinds || {}, BUTTON = DATA.button;
const keyToIndex = {};
DATA.infosets.forEach((is, i) => { keyToIndex[is.key] = i; });
const tableEl = document.getElementById("table");
const tableInfo = document.getElementById("tableinfo");
const seatBtn = new Map();
let heroSeat = null;
let shovers = new Set();

function buildTable() {
  const n = SEATS.length;
  SEATS.forEach((seat, i) => {
    const theta = -Math.PI / 2 + i * 2 * Math.PI / n;  // seat 0 (UTG) at top, clockwise
    const b = document.createElement("button");
    b.className = "seatbtn";
    b.style.left = (50 + 45 * Math.cos(theta)) + "%";
    b.style.top = (50 + 43 * Math.sin(theta)) + "%";
    let sub = seat in BLINDS ? BLINDS[seat] + "bb" : (seat === BUTTON ? "● btn" : "");
    b.innerHTML = seat + (sub ? `<span class="role">${sub}</span>` : "");
    b.onclick = () => onSeat(seat);
    tableEl.appendChild(b);
    seatBtn.set(seat, b);
  });
}

function onSeat(seat) {
  const i = SEATS.indexOf(seat);
  if (heroSeat === null) {
    heroSeat = seat;                                   // first pick sets your seat
  } else {
    const h = SEATS.indexOf(heroSeat);
    if (i < h) {                                       // earlier seat -> toggle shoved
      shovers.has(seat) ? shovers.delete(seat) : shovers.add(seat);
    } else if (i > h) {                                // later seat -> move your seat there
      heroSeat = seat;
      [...shovers].forEach(s => { if (SEATS.indexOf(s) >= i) shovers.delete(s); });
    }                                                  // tapping your own seat: no-op (use reset)
  }
  renderTable();
  resolveTable();
}

function resetTable() { heroSeat = null; shovers = new Set(); renderTable(); }

function renderTable() {
  const h = heroSeat === null ? -1 : SEATS.indexOf(heroSeat);
  seatBtn.forEach((b, seat) => {
    const i = SEATS.indexOf(seat);
    b.classList.toggle("hero", seat === heroSeat);
    b.classList.toggle("shoved", shovers.has(seat));
    b.classList.toggle("folded", h >= 0 && i < h && !shovers.has(seat));
    b.classList.toggle("toact", h >= 0 && i > h);
  });
  let msg, hint = "";
  if (heroSeat === null) {
    msg = "Tap your seat to begin.";
  } else {
    const inOrder = [...shovers].sort((a, b2) => SEATS.indexOf(a) - SEATS.indexOf(b2));
    msg = `You: <b>${heroSeat}</b> · ` + (inOrder.length ? inOrder.join(", ") + " shoved" : "folded to you");
    if (!inOrder.length && h > 0) hint = "tap anyone who already shoved";
  }
  tableInfo.innerHTML = `<span>${msg}</span>` +
    (hint ? `<span style="color:var(--ink-muted)">${hint}</span>` : "") +
    (heroSeat !== null ? ` <button id="tablereset">reset</button>` : "");
  const rb = document.getElementById("tablereset");
  if (rb) rb.onclick = resetTable;
}

function resolveTable() {
  if (heroSeat === null) return;
  const key = heroSeat + "|" + [...shovers].sort().join("|");  // alphabetical -> matches key
  if (key in keyToIndex) select(keyToIndex[key], true);
  else showWalk();  // only unmatched valid case: last seat with nobody shoved = a walk
}

function syncTableFromKey(key) {
  const parts = key.split("|");
  heroSeat = parts[0];
  shovers = new Set(parts.slice(1).filter(Boolean));
  renderTable();
}

function showWalk() {
  document.getElementById("desc").textContent =
    heroSeat + ": everyone folded — you win the walk. No decision to solve.";
  document.getElementById("key").textContent = "walk (no info set)";
  document.getElementById("pct").textContent = "—";
  document.getElementById("grid").innerHTML =
    "<div style='grid-column:1/-1;padding:28px 8px;text-align:center;color:var(--ink-muted)'>" +
    "No range — a walk isn't a decision. Add a shover, or pick another seat.</div>";
  document.getElementById("detail").textContent = "";
}

// "Action faced" filters keep the (up to 510) info sets navigable: `faced` is how many
// opponents had already shoved before hero's decision. Common play is opens + facing one.
const FILTERS = [
  { id: "common", label: "Common",   test: is => is.faced <= 1 },
  { id: "open",   label: "Opens",    test: is => is.faced === 0 },
  { id: "faced1", label: "Facing 1", test: is => is.faced === 1 },
  { id: "faced2", label: "Facing 2+", test: is => is.faced >= 2 },
  { id: "all",    label: "All",      test: () => true },
];
// Default to "all" for a small tree (4-max's 14 fit fine), "common" for a big one (9-max).
let activeFilter = DATA.infosets.length > 30 ? "common" : "all";

FILTERS.forEach(f => {
  const b = document.createElement("button");
  b.textContent = f.label; b.dataset.fid = f.id;
  b.onclick = () => { activeFilter = f.id; renderNav(); };
  filterBar.appendChild(b);
});

// Sidebar (desktop) / chip strip (mobile): tabs grouped by seat, with shove/call %.
// Rebuilt whenever the filter changes; each tab carries its absolute infoset index.
function renderNav() {
  nav.querySelectorAll(".seat, button").forEach(el => {
    if (el.parentElement !== filterBar) el.remove();
  });
  filterBar.querySelectorAll("button").forEach(b =>
    b.classList.toggle("active", b.dataset.fid === activeFilter));
  const test = FILTERS.find(f => f.id === activeFilter).test;
  let lastSeat = null;
  DATA.infosets.forEach((is, i) => {
    if (!test(is)) return;
    if (is.seat !== lastSeat) {
      const h = document.createElement("div");
      h.className = "seat"; h.textContent = is.seat;
      nav.appendChild(h); lastSeat = is.seat;
    }
    const b = document.createElement("button");
    b.innerHTML = `<span>${is.key}</span><span class="pct">${is.shovePct.toFixed(1)}%</span>`;
    b.dataset.idx = i;
    b.onclick = () => select(i);
    nav.appendChild(b);
  });
  tabs = [...nav.querySelectorAll("button")].filter(b => b.dataset.idx !== undefined);
  if (!tabs.some(b => +b.dataset.idx === current) && tabs.length) select(+tabs[0].dataset.idx);
  else markActiveTab();
}

function markActiveTab(scroll = true) {
  const active = tabs.find(t => +t.dataset.idx === current);
  tabs.forEach(t => t.classList.toggle("active", t === active));
  // Only scroll the (possibly far-down) list item into view for list/keyboard picks.
  // When you're building a spot on the table, keep the viewport put so the table
  // doesn't jump away mid-click.
  if (active && scroll) active.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function comboInfo(r, c) {
  if (r === c) return { n: 6, kind: "pair" };
  return r < c ? { n: 4, kind: "suited" } : { n: 12, kind: "offsuit" };
}

function select(i, fromTable) {
  current = i;
  selectedCell = null;
  const is = DATA.infosets[i];
  if (!fromTable) syncTableFromKey(is.key);  // list/keyboard pick -> light up the table
  markActiveTab(!fromTable);  // don't auto-scroll to the list item when built via the table
  document.getElementById("desc").textContent = is.desc;
  document.getElementById("key").textContent = "info set " + is.key;
  document.getElementById("pct").textContent = is.shovePct.toFixed(1) + "%";

  const grid = document.getElementById("grid");
  grid.innerHTML = "<div class='hdr'></div>" +
    [...RANKS].map(r => `<div class='hdr'>${r}</div>`).join("");
  for (let r = 0; r < 13; r++) {
    grid.insertAdjacentHTML("beforeend", `<div class='hdr'>${RANKS[r]}</div>`);
    for (let c = 0; c < 13; c++) {
      const w = is.grid[r][c], label = DATA.labels[r][c];
      const mixed = w > 0.02 && w < 0.98;
      const cell = document.createElement("div");
      cell.className = "cell" + (w >= 0.55 ? " hi" : "");
      cell.style.setProperty("--w", w);
      cell.innerHTML = `<div class="fill"></div><span>${label}${
        mixed ? `<small>${Math.round(w * 100)}%</small>` : ""}</span>`;
      cell.onmouseenter = () => showDetail(label, w, r, c, is.action);
      // Tap keeps the cell outlined until another is tapped -- persistent selection
      // feedback for touch, where there's no hover state.
      cell.onclick = () => {
        if (selectedCell) selectedCell.classList.remove("sel");
        cell.classList.add("sel");
        selectedCell = cell;
        showDetail(label, w, r, c, is.action);
      };
      grid.appendChild(cell);
    }
  }
}

function showDetail(label, w, r, c, action) {
  const { n, kind } = comboInfo(r, c);
  document.getElementById("detail").innerHTML =
    `<b>${label}</b> &mdash; ${action} <b>${(w * 100).toFixed(1)}%</b> &#183; ${kind} &#183; ${n} combos`;
}

document.addEventListener("keydown", e => {
  const pos = tabs.findIndex(t => +t.dataset.idx === current);
  if (e.key === "ArrowDown" || e.key === "ArrowRight") { if (tabs.length) select(+tabs[(pos + 1) % tabs.length].dataset.idx); e.preventDefault(); }
  if (e.key === "ArrowUp" || e.key === "ArrowLeft") { if (tabs.length) select(+tabs[(pos + tabs.length - 1) % tabs.length].dataset.idx); e.preventDefault(); }
  const n = parseInt(e.key); if (n >= 1 && n <= 9 && tabs[n - 1]) select(+tabs[n - 1].dataset.idx);
});

document.getElementById("meta").textContent = DATA.meta;
buildTable();
renderNav();
select(tabs.length ? +tabs[0].dataset.idx : 0);
</script>"""

LOCAL_WRAPPER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
</head>
<body>
__CONTENT__
</body>
</html>
"""


def render(payload: dict, heading: str) -> str:
    content = (
        PAGE_CONTENT
        .replace("__HEADING__", heading)
        .replace("__LIGHT_VARS__", LIGHT_VARS)
        .replace("__DARK_VARS__", DARK_VARS)
        .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="fourmax",
                        help="which solved game to view (picks the config, default ranges "
                             "pickle, and page title)")
    parser.add_argument("--ranges", default=None,
                        help="solved-ranges pickle (defaults to the chosen config's)")
    parser.add_argument("--out", default=None,
                        help="output HTML path (defaults to cache/<config>_range_viewer.html)")
    parser.add_argument("--artifact", default=None, metavar="PATH",
                        help="also write a body-only fragment (with inline <title>) for "
                             "publishing as a standalone HTML artifact")
    args = parser.parse_args()

    entry = CONFIGS[args.config]
    ranges_path = args.ranges or entry["ranges"]
    out_path = args.out or f"cache/{args.config}_range_viewer.html"
    title = entry["title"]

    payload = build_payload(ranges_path, entry["cfg"])
    content = render(payload, entry["heading"])

    html = LOCAL_WRAPPER.replace("__TITLE__", title).replace("__CONTENT__", content)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html)
    print(f"wrote {out_path} ({len(html) // 1024} KB, {len(payload['infosets'])} info sets)")

    if args.artifact:
        fragment = f"<title>{title}</title>\n" + content
        Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
        Path(args.artifact).write_text(fragment)
        print(f"wrote {args.artifact} (artifact fragment, {len(fragment) // 1024} KB)")


if __name__ == "__main__":
    main()
