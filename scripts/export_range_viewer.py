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


# Palette: the dataviz reference tokens validated for this viewer (shove red 3.85:1
# light / 5.39:1 dark vs surface; all label inks >= 3.9:1 on their fills). Defined
# once per mode and interpolated into :root, the dark media query, and both
# data-theme override blocks so the artifact theme toggle wins in either direction.
LIGHT_VARS = """\
  --surface: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
  --hairline: #e1e0d9; --ring: rgba(11,11,11,0.10);
  --fold: #f0efec; --shove: #e34948; --shove-ink: #ffffff;
  --tab-active: #0b0b0b; --tab-active-ink: #fcfcfb;"""

DARK_VARS = """\
  --surface: #1a1a19; --page: #0d0d0d;
  --ink: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
  --hairline: #2c2c2a; --ring: rgba(255,255,255,0.10);
  --fold: #383835; --shove: #e66767; --shove-ink: #0b0b0b;
  --tab-active: #ffffff; --tab-active-ink: #0d0d0d;"""

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
#wrap { margin-top: 18px; max-width: 720px; }
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
  #grid { gap: 1px; padding: 6px; grid-template-columns: 15px repeat(13, minmax(0, 1fr)); }
  #grid .hdr { font-size: 9px; }
  .cell span { font-size: clamp(6.5px, 2.1vw, 11px); }
  .cell { border-radius: 3px; }
}
</style>
<div id="rv-root">
<nav id="nav"><h1>4-max push/fold</h1></nav>
<main>
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
let current = 0;
let selectedCell = null;

// Sidebar (desktop) / chip strip (mobile): tabs grouped by seat, with shove/call %.
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
  selectedCell = null;
  const is = DATA.infosets[i];
  tabs.forEach((t, j) => t.classList.toggle("active", j === i));
  tabs[i].scrollIntoView({ block: "nearest", inline: "nearest" });
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
  if (e.key === "ArrowDown" || e.key === "ArrowRight") { select((current + 1) % tabs.length); e.preventDefault(); }
  if (e.key === "ArrowUp" || e.key === "ArrowLeft") { select((current + tabs.length - 1) % tabs.length); e.preventDefault(); }
  const n = parseInt(e.key); if (n >= 1 && n <= 9) select(n - 1);
});

document.getElementById("meta").textContent = DATA.meta;
select(0);
</script>"""

TITLE = "4-max Push/Fold Ranges"

LOCAL_WRAPPER = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
</head>
<body>
__CONTENT__
</body>
</html>
"""


def render(payload: dict) -> str:
    content = (
        PAGE_CONTENT
        .replace("__LIGHT_VARS__", LIGHT_VARS)
        .replace("__DARK_VARS__", DARK_VARS)
        .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranges", default="cache/fourmax_ranges.pkl")
    parser.add_argument("--out", default="cache/range_viewer.html")
    parser.add_argument("--artifact", default=None, metavar="PATH",
                        help="also write a body-only fragment (with inline <title>) for "
                             "publishing as a standalone HTML artifact")
    args = parser.parse_args()

    payload = build_payload(args.ranges)
    content = render(payload)

    html = LOCAL_WRAPPER.replace("__CONTENT__", content)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html) // 1024} KB, {len(payload['infosets'])} info sets)")

    if args.artifact:
        fragment = f"<title>{TITLE}</title>\n" + content
        Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
        Path(args.artifact).write_text(fragment)
        print(f"wrote {args.artifact} (artifact fragment, {len(fragment) // 1024} KB)")


if __name__ == "__main__":
    main()
