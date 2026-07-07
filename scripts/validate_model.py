#!/usr/bin/env python3
"""Validates the solver's predicted EV against REAL observed outcomes in a hand
history -- averages out per-hand showdown variance over many hands to check
whether the model tracks reality. Distinct from scripts/backtest.py, which grades
Hero's OWN decisions against the GTO recommendation; this instead asks "regardless
of whether Hero played well, does the model's EV prediction for whatever Hero
actually did match what actually happened, on average?" -- a sanity check on the
model itself, not on Hero's play."""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import equity
import hand_history
import pushfold_spot
from backtest import SolveCache, StackTooSmallError, grade_spot, realized_ev_net_bb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-history", default="data/handhistory.txt")
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--hero-name", default="Hero")
    args = parser.parse_args()

    hands, _ = hand_history.parse_file(args.hand_history)
    table = equity.load_or_build_equity_table(cache_dir=args.cache_dir)
    cache = SolveCache(table)

    by_position: dict[str, dict[str, list[float]]] = {
        "SB": {"predicted": [], "realized": []},
        "BB": {"predicted": [], "realized": []},
    }
    for hand in hands:
        spot, reason = pushfold_spot.find_spot(hand, hero_name=args.hero_name)
        if spot is None:
            continue
        try:
            graded = grade_spot(spot, cache)
        except StackTooSmallError:
            continue
        solved = cache.get(spot.effective_stack_bb)
        realized = realized_ev_net_bb(hand, spot, solved.cfg)
        by_position[graded.position]["predicted"].append(graded.predicted_ev_net_bb)
        by_position[graded.position]["realized"].append(realized)

    print(f"{'position':10} {'n':>6} {'predicted avg':>15} {'realized avg':>14} {'realized SE':>13}  verdict")
    for position in ("SB", "BB"):
        predicted = by_position[position]["predicted"]
        realized = by_position[position]["realized"]
        n = len(predicted)
        if n < 2:
            print(f"{position:10} {n:6d}  (not enough spots)")
            continue
        pred_avg = sum(predicted) / n
        real_avg = sum(realized) / n
        real_var = sum((x - real_avg) ** 2 for x in realized) / (n - 1)
        real_se = math.sqrt(real_var / n)
        z = (real_avg - pred_avg) / real_se if real_se > 0 else float("nan")
        verdict = "within 2 SE, consistent" if abs(z) < 2 else "outside 2 SE -- investigate"
        print(f"{position:10} {n:6d} {pred_avg:15.4f} {real_avg:14.4f} {real_se:13.4f}  z={z:+.2f} ({verdict})")

    n_all = len(by_position["SB"]["predicted"]) + len(by_position["BB"]["predicted"])
    predicted_all = by_position["SB"]["predicted"] + by_position["BB"]["predicted"]
    realized_all = by_position["SB"]["realized"] + by_position["BB"]["realized"]
    if n_all >= 2:
        pred_avg = sum(predicted_all) / n_all
        real_avg = sum(realized_all) / n_all
        real_var = sum((x - real_avg) ** 2 for x in realized_all) / (n_all - 1)
        real_se = math.sqrt(real_var / n_all)
        z = (real_avg - pred_avg) / real_se if real_se > 0 else float("nan")
        verdict = "within 2 SE, consistent" if abs(z) < 2 else "outside 2 SE -- investigate"
        print(f"{'combined':10} {n_all:6d} {pred_avg:15.4f} {real_avg:14.4f} {real_se:13.4f}  z={z:+.2f} ({verdict})")


if __name__ == "__main__":
    main()
