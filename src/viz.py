"""13x13 range-grid heatmaps, in the universal push/fold-chart convention: rows and
columns run A..2 (highest rank first), upper-right triangle = suited, lower-left
triangle = offsuit, diagonal = pairs -- directly eyeball-comparable to
SnapShove/HoldemResources-style charts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cards
import tree

DESCENDING_RANKS = "AKQJT98765432"  # row/col 0 = A, 12 = 2


def _grid_canon_index() -> np.ndarray:
    """13x13 array of canonical indices (0..168), computed once and reused by every
    call to range_to_grid."""
    grid = np.zeros((13, 13), dtype=int)
    for row in range(13):
        for col in range(13):
            hi_char, lo_char = DESCENDING_RANKS[row], DESCENDING_RANKS[col]
            if row == col:
                label = hi_char * 2
            elif row < col:
                label = hi_char + lo_char + "s"
            else:
                label = lo_char + hi_char + "o"
            grid[row, col] = cards.canon_index(label)
    return grid


_GRID_CANON_INDEX = _grid_canon_index()


def range_to_grid(range_vec: np.ndarray) -> np.ndarray:
    """length-169 range vector (canonical order) -> 13x13 grid in chart convention."""
    return range_vec[_GRID_CANON_INDEX]


def plot_range_heatmap(range_vec: np.ndarray, title: str, ax=None, cmap: str = "RdYlGn_r"):
    """Cell color = shove/call weight in [0,1] (0=fold, 1=shove/call, continuous
    colormap doubles as the mixed-strategy encoding for fractional weights). Cell
    text = canonical label, e.g. 'AKs'."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    grid = range_to_grid(range_vec)
    im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0)

    for row in range(13):
        for col in range(13):
            idx = _GRID_CANON_INDEX[row, col]
            weight = grid[row, col]
            text_color = "white" if weight > 0.6 or weight < 0.25 else "black"
            ax.text(
                col,
                row,
                cards.canon_label(idx),
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.set_xticklabels(list(DESCENDING_RANKS))
    ax.set_yticklabels(list(DESCENDING_RANKS))
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_hu_ranges(ranges: dict, cfg, out_dir: str = "cache/") -> str:
    """HU-specific convenience: SB's open-shove range and BB's call-vs-SB range,
    side by side, saved as one PNG. The only HU-specific function in this module --
    range_to_grid/plot_range_heatmap are fully general and get reused unchanged for
    4-max's 14 heatmaps later."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    plot_range_heatmap(ranges["SB|"], "SB open-shove range", ax=axes[0])
    plot_range_heatmap(ranges["BB|SB"], "BB call-vs-SB range", ax=axes[1])
    fig.tight_layout()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / "hu_ranges.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_fourmax_ranges(ranges: dict, cfg, out_dir: str = "cache/") -> str:
    """All 14 four-max info sets as one grid of heatmaps, one row per seat (UTG: 1
    panel, BTN: 2, SB: 4, BB: 7 -- matching each seat's own info-set count, per the
    spec's 1+2+4+7=14 enumeration), left-aligned within a 4x7 GridSpec since 14 has
    no clean square layout. Reuses range_to_grid/plot_range_heatmap unchanged --
    they're already fully general over any range vector, as noted in plot_hu_ranges's
    own docstring."""
    infosets = tree.build_infosets(cfg)
    by_seat: dict[str, list] = {}
    for infoset in infosets:
        by_seat.setdefault(infoset.seat, []).append(infoset)
    for seat_infosets in by_seat.values():
        seat_infosets.sort(key=lambda i: len(i.shoved_before))

    seats = list(cfg.seats)
    max_cols = max(len(v) for v in by_seat.values())

    fig = plt.figure(figsize=(3.2 * max_cols, 3.2 * len(seats)))
    grid = fig.add_gridspec(len(seats), max_cols)

    for row, seat in enumerate(seats):
        for col, infoset in enumerate(by_seat[seat]):
            key = tree.infoset_key(infoset.seat, infoset.shoved_before)
            ax = fig.add_subplot(grid[row, col])
            plot_range_heatmap(ranges[key], key, ax=ax)

    fig.tight_layout()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / "fourmax_ranges.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
