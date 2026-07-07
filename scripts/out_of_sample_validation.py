#!/usr/bin/env python3
"""Out-of-sample robustness check: splits the hand history chronologically into a
first half and second half, then independently re-runs the same checks as
validate_model.py (realized vs predicted EV by position) and
opponent_deviation.py (opponent shove/call frequency vs GTO, plus the calling-
RANGE-composition check -- actual showdown win rate vs model-implied equity for
contested SB shoves) on each half separately.

The point isn't a new finding -- it's asking whether the SINGLE finding from the
full-sample backtest (opponents call Hero's shoves with a weaker-than-GTO range)
is a stable pattern across time, or an artifact concentrated in one part of the
sample (a handful of hands/players/a since-changed opponent pool)."""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
import equity
import hand_history
import pushfold_spot
import solver
import tree
from backtest import SolveCache, StackTooSmallError, grade_spot, realized_ev_net_bb
from game import HU_CONFIG
from pushfold_spot import classify_bb_response_when_hero_is_sb, classify_sb_open_when_hero_is_bb


def _binomial_se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n) if n > 0 else float("nan")


def analyze_half(hands, label, cache, table, ranges, hero_name="Hero") -> None:
    print(f"\n{'=' * 20} {label} ({len(hands)} hands, "
          f"{hands[0].timestamp} .. {hands[-1].timestamp}) {'=' * 20}")

    by_position = {"SB": {"predicted": [], "realized": []}, "BB": {"predicted": [], "realized": []}}
    sb_open_counts = {"shove": 0, "fold": 0}
    bb_call_counts = {"call": 0, "fold": 0}
    contested_sb_shoves = []  # (hero_idx, won: bool)

    for hand in hands:
        spot, reason = pushfold_spot.find_spot(hand, hero_name=hero_name)
        if spot is not None:
            try:
                graded = grade_spot(spot, cache)
            except StackTooSmallError:
                spot = None
            else:
                solved = cache.get(spot.effective_stack_bb)
                realized = realized_ev_net_bb(hand, spot, solved.cfg)
                by_position[graded.position]["predicted"].append(graded.predicted_ev_net_bb)
                by_position[graded.position]["realized"].append(realized)

                if spot.info_set_key == "SB|" and spot.hero_action == "shove_or_call":
                    if classify_bb_response_when_hero_is_sb(hand, hero_name=hero_name) == "call":
                        hero_idx = cards.canon_index(cards.hero_canon_label(*spot.hero_hand))
                        won = hand.collected_by.get(hero_name, 0.0) > 0
                        contested_sb_shoves.append((hero_idx, won))

        sb_result = classify_sb_open_when_hero_is_bb(hand, hero_name=hero_name)
        if sb_result is not None:
            sb_open_counts[sb_result] += 1
        bb_result = classify_bb_response_when_hero_is_sb(hand, hero_name=hero_name)
        if bb_result is not None:
            bb_call_counts[bb_result] += 1

    print("\n-- realized vs predicted EV (validate_model.py logic) --")
    for position in ("SB", "BB"):
        predicted = by_position[position]["predicted"]
        realized = by_position[position]["realized"]
        n = len(predicted)
        if n < 2:
            print(f"{position}: not enough spots ({n})")
            continue
        pred_avg = sum(predicted) / n
        real_avg = sum(realized) / n
        real_var = sum((x - real_avg) ** 2 for x in realized) / (n - 1)
        real_se = math.sqrt(real_var / n)
        z = (real_avg - pred_avg) / real_se if real_se > 0 else float("nan")
        print(f"{position}: n={n} predicted={pred_avg:.4f}bb realized={real_avg:.4f}bb "
              f"SE={real_se:.4f} z={z:+.2f}")

    print("\n-- opponent shove/call frequency vs GTO (opponent_deviation.py logic) --")
    n_sb = sb_open_counts["shove"] + sb_open_counts["fold"]
    n_bb = bb_call_counts["call"] + bb_call_counts["fold"]
    gto_sb_shove_freq = cards.range_average_weight(ranges["SB|"])
    gto_bb_call_freq = cards.range_average_weight(ranges["BB|SB"])
    if n_sb > 0:
        obs = sb_open_counts["shove"] / n_sb
        se = _binomial_se(obs, n_sb)
        z = (obs - gto_sb_shove_freq) / se if se > 0 else float("nan")
        print(f"SB-open shove freq: n={n_sb} observed={obs:.1%} GTO={gto_sb_shove_freq:.1%} z={z:+.2f}")
    if n_bb > 0:
        obs = bb_call_counts["call"] / n_bb
        se = _binomial_se(obs, n_bb)
        z = (obs - gto_bb_call_freq) / se if se > 0 else float("nan")
        print(f"BB-call freq:       n={n_bb} observed={obs:.1%} GTO={gto_bb_call_freq:.1%} z={z:+.2f}")

    print("\n-- calling-range composition (actual win rate vs model equity, contested SB shoves) --")
    n_contested = len(contested_sb_shoves)
    if n_contested > 0:
        wins = sum(1 for _, won in contested_sb_shoves if won)
        win_rate = wins / n_contested
        win_se = _binomial_se(win_rate, n_contested)
        model_eq_sum = sum(
            equity.hand_vs_range_equity(idx, ranges["BB|SB"], table) for idx, _ in contested_sb_shoves
        )
        model_eq_avg = model_eq_sum / n_contested
        z = (win_rate - model_eq_avg) / win_se if win_se > 0 else float("nan")
        print(f"n={n_contested} actual_win_rate={win_rate:.1%} +/- {win_se:.1%} "
              f"model_equity={model_eq_avg:.1%} z={z:+.2f}")
    else:
        print("no contested SB shoves in this half")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-history", default="data/handhistory.txt")
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--hero-name", default="Hero")
    args = parser.parse_args()

    hands, _ = hand_history.parse_file(args.hand_history)
    hands.sort(key=lambda h: hand_history.parse_timestamp(h.timestamp))

    mid = len(hands) // 2
    first_half, second_half = hands[:mid], hands[mid:]

    table = equity.load_or_build_equity_table(cache_dir=args.cache_dir)
    cache = SolveCache(table)
    ranges, _info = solver.solve(HU_CONFIG, table=table)

    analyze_half(first_half, "FIRST HALF", cache, table, ranges, hero_name=args.hero_name)
    analyze_half(second_half, "SECOND HALF", cache, table, ranges, hero_name=args.hero_name)


if __name__ == "__main__":
    main()
