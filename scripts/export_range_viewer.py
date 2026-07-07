#!/usr/bin/env python3
"""Export the solved 4-max ranges as a single self-contained interactive HTML viewer.

The production PNG (viz.plot_fourmax_ranges) crams all 14 heatmaps into one image --
fine as a record, useless for quick reference. This script emits
cache/range_viewer.html: one offline file (no CDNs, no fetches) with a tab per info
set, a large 13x13 grid, and hover/tap detail for exact per-hand weights.

Cell encoding: each cell is filled left-to-right by its shove/call weight -- the red
fill occupies exactly weight% of the cell width over a neutral "fold" base. That makes
mixed-strategy cells readable at a glance (a 30% call literally looks 30% full) and,
unlike the PNG's red-yellow-green interpolation, doesn't rely on a red-vs-green hue
pair (the classic colorblind-unsafe combination): the value is carried by geometry,
color just makes it pop. Red still means shove/call, matching the PNG's salient half.

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
from game import FOURMAX_CONFIG


def infoset_description(cfg, infoset) -> str:
    """Plain-English situation line, e.g. 'BB call vs UTG + SB shoves (BTN folded)'."""
    order = {s: i for i, s in enumerate(cfg.seats)}
    shovers = sorted(infoset.shoved_before, key=order.get)
    folded = [s for s in cfg.seats[: infoset.seat_pos] if s not in infoset.shoved_before]
    folded_note = f" ({', '.join(folded)} folded)" if folded else ""
    if not shovers:
        return f"{infoset.seat} open-shove{folded_note or ' (first to act)'}"
    return f"{infoset.seat} call vs {' + '.join(shovers)} shove{'s' if len(shovers) > 1 else ''}{folded_note}"


def build_payload(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    ranges, info = saved["ranges"], saved.get("info", {})

    # Label grid straight from the solver's own 13x13 convention (viz._GRID_CANON_INDEX):
    # rows/cols A..2, upper-right suited, lower-left offsuit, diagonal pairs.
    labels = [[cards.canon_label(viz._GRID_CANON_INDEX[r, c]) for c in range(13)] for r in range(13)]

    infosets = []
    for i in tree.build_infosets(FOURMAX_CONFIG):
        key = tree.infoset_key(i.seat, i.shoved_before)
        vec = ranges[key]
        shove_pct = 100 * sum(vec[j] * cards.CANON_COMBO_WEIGHT[j] for j in range(169)) / cards.NUM_COMBOS
        infosets.append({
            "key": key,
            "seat": i.seat,
            "action": "call" if i.shoved_before else "shove",
            "desc": infoset_description(FOURMAX_CONFIG, i),
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
    return {"labels": labels, "infosets": infosets, "meta": " · ".join(meta_bits)}


# One self-contained page. Colors are the dataviz reference palette's validated steps
# (shove red 3.85:1 light / 5.39:1 dark vs surface; all label inks >= 3.9:1 on their
# fills) declared once as CSS custom properties, with dark mode as its own selected set.
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>4-max push/fold ranges</title>
<style>
:root {
  --surface: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
  --hairline: #e1e0d9; --ring: rgba(11,11,11,0.10);
  --fold: #f0efec; --shove: #e34948; --shove-ink: #ffffff;
  --tab-active: #0b0b0b; --tab-active-ink: #fcfcfb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --page: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
    --hairline: #2c2c2a; --ring: rgba(255,255,255,0.10);
    --fold: #383835; --shove: #e66767; --shove-ink: #0b0b0b;
    --tab-active: #ffffff; --tab-active-ink: #0d0d0d;
  }
}
* { box-sizing: border-box; margin: 0; }
body {
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page); color: var(--ink);
  display: flex; min-height: 100vh;
}
nav {
  width: 232px; flex: none; padding: 16px 10px 24px;
  border-right: 1px solid var(--hairline); overflow-y: auto;
}
nav h1 { font-size: 15px; padding: 2px 8px 10px; }
nav .seat { font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
  color: var(--ink-muted); text-transform: uppercase; padding: 12px 8px 4px; }
nav button {
  display: flex; justify-content: space-between; gap: 8px; width: 100%;
  padding: 6px 8px; margin: 1px 0; border: 0; border-radius: 6px;
  background: none; color: var(--ink-2); font: inherit; font-size: 13px;
  text-align: left; cursor: pointer;
}
nav button:hover { background: var(--fold); }
nav button.active { background: var(--tab-active); color: var(--tab-active-ink); }
nav button .pct { font-variant-numeric: tabular-nums; opacity: 0.75; }
main { flex: 1; padding: 22px 28px; min-width: 0; }
header h2 { font-size: 18px; }
header .key { color: var(--ink-muted); font-size: 13px; margin-top: 2px; }
header .stat { margin-top: 6px; font-size: 14px; color: var(--ink-2); }
header .stat b { color: var(--ink); font-size: 16px; }
#wrap { margin-top: 18px; max-width: 720px; }
#grid {
  display: grid; grid-template-columns: 20px repeat(13, 1fr);
  gap: 2px; background: var(--surface); padding: 10px;
  border: 1px solid var(--ring); border-radius: 10px;
}
.hdr { display: flex; align-items: center; justify-content: center;
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
.cell:hover { outline: 2px solid var(--ink); outline-offset: -2px; z-index: 1; }
#detail { margin-top: 10px; min-height: 22px; font-size: 14px; color: var(--ink-2); }
#detail b { color: var(--ink); }
#legend { display: flex; gap: 18px; align-items: center; margin-top: 8px;
  font-size: 12px; color: var(--ink-muted); }
#legend .sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
  margin-right: 5px; vertical-align: -2px; border: 1px solid var(--ring); }
footer { margin-top: 14px; font-size: 12px; color: var(--ink-muted); }
@media (max-width: 760px) {
  body { flex-direction: column; }
  nav { width: auto; border-right: 0; border-bottom: 1px solid var(--hairline);
    display: flex; flex-wrap: wrap; gap: 2px; }
  nav h1, nav .seat { width: 100%; }
  nav button { width: auto; }
}
</style>
</head>
<body>
<nav id="nav"><h1>4-max push/fold</h1></nav>
<main>
  <header>
    <h2 id="desc"></h2>
    <div class="key" id="key"></div>
    <div class="stat">Range: <b id="pct"></b> of combos</div>
  </header>
  <div id="wrap">
    <div id="grid"></div>
    <div id="detail">Hover a hand for exact weight.</div>
    <div id="legend">
      <span><span class="sw" style="background:var(--shove)"></span>shove / call (fill = weight)</span>
      <span><span class="sw" style="background:var(--fold)"></span>fold</span>
      <span>rows/cols A&#8594;2 &#183; upper right suited &#183; diagonal pairs &#183; lower left offsuit</span>
    </div>
    <footer id="meta"></footer>
  </div>
</main>
<script>
const DATA = __DATA__;
const RANKS = "AKQJT98765432";
const nav = document.getElementById("nav");
let current = 0;

// Sidebar: tabs grouped by seat, each showing its aggregate shove/call %.
let lastSeat = null;
DATA.infosets.forEach((is, i) => {
  if (is.seat !== lastSeat) {
    const h = document.createElement("div");
    h.className = "seat"; h.textContent = is.seat;
    nav.appendChild(h); lastSeat = is.seat;
  }
  const b = document.createElement("button");
  b.innerHTML = `<span>${is.key}</span><span class="pct">${is.shovePct.toFixed(1)}%</span>`;
  b.onclick = () => select(i);
  nav.appendChild(b);
});
const tabs = [...nav.querySelectorAll("button")];

function comboInfo(r, c) {
  if (r === c) return { n: 6, kind: "pair" };
  return r < c ? { n: 4, kind: "suited" } : { n: 12, kind: "offsuit" };
}

function select(i) {
  current = i;
  const is = DATA.infosets[i];
  tabs.forEach((t, j) => t.classList.toggle("active", j === i));
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
      cell.onclick = () => showDetail(label, w, r, c, is.action);
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
  if (e.key === "ArrowDown" || e.key === "ArrowRight") { select((current + 1) % tabs.length); e.preventDefault(); }
  if (e.key === "ArrowUp" || e.key === "ArrowLeft") { select((current + tabs.length - 1) % tabs.length); e.preventDefault(); }
  const n = parseInt(e.key); if (n >= 1 && n <= 9) select(n - 1);
});

document.getElementById("meta").textContent = DATA.meta;
select(0);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranges", default="cache/fourmax_ranges.pkl")
    parser.add_argument("--out", default="cache/range_viewer.html")
    args = parser.parse_args()

    payload = build_payload(args.ranges)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html) // 1024} KB, {len(payload['infosets'])} info sets)")


if __name__ == "__main__":
    main()
