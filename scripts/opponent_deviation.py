#!/usr/bin/env python3
"""Measures how much your REAL opponents deviate from the solved chart's own
shove/call frequencies -- doesn't need their hole cards, only their action
labels, so it isolates whether your actual opponent pool is looser or tighter
than GTO, independent of how well Hero herself plays (see scripts/backtest.py
for that side)."""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
import equity
import hand_history
import solver
import tree
from game import HU_CONFIG
from pushfold_spot import classify_bb_response_when_hero_is_sb, classify_sb_open_when_hero_is_bb


def _binomial_se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n) if n > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-history", default="data/handhistory.txt")
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--hero-name", default="Hero")
    args = parser.parse_args()

    hands, _ = hand_history.parse_file(args.hand_history)

    sb_open_counts: dict[str, int] = {"shove": 0, "fold": 0}
    bb_call_counts: dict[str, int] = {"call": 0, "fold": 0}
    for hand in hands:
        sb_result = classify_sb_open_when_hero_is_bb(hand, hero_name=args.hero_name)
        if sb_result is not None:
            sb_open_counts[sb_result] += 1
        bb_result = classify_bb_response_when_hero_is_sb(hand, hero_name=args.hero_name)
        if bb_result is not None:
            bb_call_counts[bb_result] += 1

    table = equity.load_or_build_equity_table(cache_dir=args.cache_dir)
    ranges, info = solver.solve(HU_CONFIG, table=table)
    gto_sb_shove_freq = cards.range_average_weight(ranges["SB|"])
    gto_bb_call_freq = cards.range_average_weight(ranges["BB|SB"])

    n_sb = sb_open_counts["shove"] + sb_open_counts["fold"]
    n_bb = bb_call_counts["call"] + bb_call_counts["fold"]

    print(f"Opponent SB-open decisions observed (Hero was BB): {n_sb}")
    print(f"  shove: {sb_open_counts['shove']}   fold: {sb_open_counts['fold']}")
    if n_sb > 0:
        obs = sb_open_counts["shove"] / n_sb
        se = _binomial_se(obs, n_sb)
        z = (obs - gto_sb_shove_freq) / se if se > 0 else float("nan")
        print(f"  observed shove freq: {obs:.1%} +/- {se:.1%} (1 SE)   GTO shove freq: {gto_sb_shove_freq:.1%}")
        print(f"  z = {z:+.2f} ({'consistent with GTO' if abs(z) < 2 else 'opponents deviate from GTO'})")

    print()
    print(f"Opponent BB-call decisions observed (Hero was SB, shoved): {n_bb}")
    print(f"  call:  {bb_call_counts['call']}   fold: {bb_call_counts['fold']}")
    if n_bb > 0:
        obs = bb_call_counts["call"] / n_bb
        se = _binomial_se(obs, n_bb)
        z = (obs - gto_bb_call_freq) / se if se > 0 else float("nan")
        print(f"  observed call freq:  {obs:.1%} +/- {se:.1%} (1 SE)   GTO call freq:  {gto_bb_call_freq:.1%}")
        print(f"  z = {z:+.2f} ({'consistent with GTO' if abs(z) < 2 else 'opponents deviate from GTO'})")


if __name__ == "__main__":
    main()
