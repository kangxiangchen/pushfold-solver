#!/usr/bin/env python3
"""Bankroll sizing via risk-of-ruin, using REALIZED per-hand results from the
backtest (not the theoretical model) -- segmented by stake ($1/$2 vs $2/$5, the
two 8bb push/fold table formats in the data) and blended for a mixed-stakes
session.

Risk-of-ruin: for a per-hand result process with mean mu and variance sigma^2
(in the SAME currency units as the bankroll), the classic Brownian-motion
approximation (Chen & Ankenman, "The Mathematics of Poker") gives

    RoR(B) = exp(-2 * mu * B / sigma^2)   for mu > 0

Solving for the bankroll B that holds risk-of-ruin at a chosen target r:

    B = -sigma^2 * ln(r) / (2 * mu)

This assumes i.i.d. per-hand results and treats the bankroll process as a
continuous random walk with drift -- standard, well-known simplifications for
this kind of sizing question, not something specific to this script. If mu <= 0
(no measured edge), no finite bankroll avoids eventual ruin -- the formula
doesn't apply and isn't evaluated.

The "blended mix" pools raw per-hand dollar results across both stakes exactly
as they actually occurred in the data (i.e. using the HISTORICAL proportion of
hands played at each stake as the implicit mix weight) -- a different target
mix would need reweighting, not attempted here.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import equity
import hand_history
import pushfold_spot
from backtest import SolveCache, StackTooSmallError, grade_spot, realized_ev_net_bb

RISK_TARGETS = [0.05, 0.01, 0.005, 0.001]


def bankroll_for_risk(mean: float, var: float, risk: float) -> float:
    if mean <= 0:
        return float("inf")
    return -var * math.log(risk) / (2 * mean)


def summarize(label: str, results_dollars: list[float], buyins: dict[str, float]) -> None:
    n = len(results_dollars)
    print(f"\n{label}: n={n}")
    if n < 2:
        print("  not enough hands")
        return

    mean = sum(results_dollars) / n
    var = sum((x - mean) ** 2 for x in results_dollars) / (n - 1)
    std = math.sqrt(var)
    per_100 = mean * 100
    se = std / math.sqrt(n)
    z = mean / se if se > 0 else float("nan")

    print(f"  mean = ${mean:.4f}/hand   std = ${std:.4f}/hand   (${per_100:.2f} per 100 hands)")
    print(f"  SE = ${se:.4f}   z = {z:+.2f}   ({'statistically distinguishable from 0' if abs(z) >= 1.96 else 'NOT statistically distinguishable from 0 at 95%'})")
    if mean <= 0:
        print("  mean <= 0 -- no finite bankroll avoids eventual ruin at this edge; fix the leak before sizing.")
        return

    print("  required bankroll by target risk-of-ruin:")
    for risk in RISK_TARGETS:
        B = bankroll_for_risk(mean, var, risk)
        buyin_strs = ", ".join(f"{B / amt:.1f}x {name}" for name, amt in buyins.items())
        print(f"    {risk:>6.1%} RoR: ${B:>10,.2f}   ({buyin_strs})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-history", default="data/handhistory.txt")
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--hero-name", default="Hero")
    args = parser.parse_args()

    hands, _ = hand_history.parse_file(args.hand_history)
    table = equity.load_or_build_equity_table(cache_dir=args.cache_dir)
    cache = SolveCache(table)

    results_by_stake: dict[float, list[float]] = {2.0: [], 5.0: []}
    all_results: list[float] = []

    for hand in hands:
        spot, reason = pushfold_spot.find_spot(hand, hero_name=args.hero_name)
        if spot is None:
            continue
        try:
            solved = cache.get(spot.effective_stack_bb)
        except StackTooSmallError:
            continue
        realized_bb = realized_ev_net_bb(hand, spot, solved.cfg)
        realized_dollars = realized_bb * hand.bb_amount
        if hand.bb_amount in results_by_stake:
            results_by_stake[hand.bb_amount].append(realized_dollars)
        all_results.append(realized_dollars)

    buyin_1_2 = 8 * 2.0
    buyin_2_5 = 8 * 5.0

    summarize("$1/$2 (8bb buy-in = $16)", results_by_stake[2.0], {"$1/$2 buy-ins": buyin_1_2})
    summarize("$2/$5 (8bb buy-in = $40)", results_by_stake[5.0], {"$2/$5 buy-ins": buyin_2_5})
    summarize(
        "Blended mix (historical proportion of hands actually played)",
        all_results,
        {"$1/$2 buy-ins": buyin_1_2, "$2/$5 buy-ins": buyin_2_5},
    )

    n1, n2 = len(results_by_stake[2.0]), len(results_by_stake[5.0])
    print(f"\n(historical mix: {n1} hands at $1/$2, {n2} hands at $2/$5 "
          f"= {100 * n1 / (n1 + n2):.0f}% / {100 * n2 / (n1 + n2):.0f}%)")


if __name__ == "__main__":
    main()
