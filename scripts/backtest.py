#!/usr/bin/env python3
"""Entrypoint: hand-history .txt -> parse -> detect gradable HU push/fold spots ->
solve each distinct effective-stack depth -> grade Hero's decisions -> summary +
CSV. See src/pushfold_spot.py for exactly which hands count as gradable and why
the rest are skipped (antes unmodeled, 4-max multiway shoves out of scope, etc)."""
import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import equity
import fourmax_spot
import hand_history
import pushfold_spot
from backtest import GradedSpot, SolveCache, StackTooSmallError, grade_spot, grade_spot_against, load_or_solve_fourmax_ranges

# HU skip reasons that a 4-max model COULD grade, if --include-fourmax is passed --
# everything else (antes, non-all-in action, an ungradable walk, a garbage stack
# depth) is out of scope for either model, not worth a second attempt.
FOURMAX_RETRY_REASONS = {"hero_not_blind_poster", "shove_multiway", "call_multiway"}

CSV_FIELDS = [
    "hand_id",
    "timestamp",
    "model",
    "position",
    "effective_stack_bb",
    "hero_hand",
    "hero_action",
    "gto_weight",
    "gto_recommended_action",
    "ev_loss_bb",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-history", default="data/handhistory.txt")
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--out-csv", default="cache/backtest_results.csv")
    parser.add_argument("--hero-name", default="Hero")
    parser.add_argument(
        "--include-fourmax",
        action="store_true",
        help="Also grade multiway/non-blind-seat spots against a 4-max solve. Solving "
        "FOURMAX_CONFIG from scratch takes tens of minutes to a few hours (see README's "
        "performance section) -- the result is cached to cache/fourmax_ranges_<depth>bb.pkl "
        "so this cost is paid at most once.",
    )
    args = parser.parse_args()

    hands, parse_failures = hand_history.parse_file(args.hand_history)
    print(f"parsed {len(hands)} hands ({len(parse_failures)} failed to parse)")

    table = equity.load_or_build_equity_table(cache_dir=args.cache_dir)
    cache = SolveCache(table)

    fourmax_cfg = fourmax_ranges = fourmax_evaluator = None
    if args.include_fourmax:
        fourmax_evaluator = equity.get_evaluator()
        print("\nsolving (or loading cached) 4-max ranges -- this may take a while...")
        fourmax_cfg, fourmax_ranges = load_or_solve_fourmax_ranges(table, fourmax_evaluator, cache_dir=args.cache_dir)
        print("4-max ranges ready.")

    skip_counts: collections.Counter = collections.Counter()
    graded: list[GradedSpot] = []
    for hand in hands:
        spot, reason = pushfold_spot.find_spot(hand, hero_name=args.hero_name)
        if spot is not None:
            try:
                graded.append(grade_spot(spot, cache))
            except StackTooSmallError:
                skip_counts["stack_too_small_for_config"] += 1
            continue

        if args.include_fourmax and reason in FOURMAX_RETRY_REASONS:
            fm_spot, fm_reason = fourmax_spot.find_fourmax_spot(hand, hero_name=args.hero_name)
            if fm_spot is not None:
                graded.append(grade_spot_against(fm_spot, fourmax_cfg, fourmax_ranges, table, fourmax_evaluator, model="4MAX"))
                continue
            reason = f"{reason}_then_fourmax_{fm_reason}"

        skip_counts[reason] += 1

    print(f"\ngradable spots: {len(graded)}")
    print("skip reasons:")
    for reason, count in sorted(skip_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {count}")

    if graded:
        model_counts = collections.Counter(g.model for g in graded)
        position_counts = collections.Counter(g.position for g in graded)
        matched = sum(1 for g in graded if g.ev_loss_bb < 1e-6)
        total_loss = sum(g.ev_loss_bb for g in graded)
        avg_loss = total_loss / len(graded)

        print(f"\nby model: {dict(model_counts)}")
        print(f"by position: {dict(position_counts)}")
        print(f"matched GTO exactly: {matched}/{len(graded)} ({100 * matched / len(graded):.1f}%)")
        print(f"total EV loss: {total_loss:.2f}bb  average EV loss: {avg_loss:.4f}bb")

        worst = sorted(graded, key=lambda g: -g.ev_loss_bb)[:10]
        print("\nworst deviations:")
        for g in worst:
            print(
                f"  hand {g.hand_id} [{g.model}/{g.position}] {g.hero_hand} @ {g.effective_stack_bb:.1f}bb: "
                f"hero={g.hero_action} gto={g.gto_recommended_action} "
                f"(weight={g.gto_weight:.2f}) loss={g.ev_loss_bb:.3f}bb"
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for g in graded:
            writer.writerow(
                {
                    "hand_id": g.hand_id,
                    "timestamp": g.timestamp,
                    "model": g.model,
                    "position": g.position,
                    "effective_stack_bb": g.effective_stack_bb,
                    "hero_hand": g.hero_hand,
                    "hero_action": g.hero_action,
                    "gto_weight": g.gto_weight,
                    "gto_recommended_action": g.gto_recommended_action,
                    "ev_loss_bb": g.ev_loss_bb,
                }
            )
    print(f"\nCSV written to {out_path}")


if __name__ == "__main__":
    main()
