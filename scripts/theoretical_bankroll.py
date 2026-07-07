#!/usr/bin/env python3
"""Bankroll sizing from the SOLVED MODEL alone -- zero dependency on real hand
history. Monte Carlo simulates synthetic hands directly from the solved HU
ranges + the exact 169x169 equity table (hero's hand, hero's mixed action,
opponent's hand, opponent's mixed action, win/lose all drawn from the model's
own distributions), giving an exact-in-the-limit theoretical mean AND variance
of per-hand ev_net -- the same risk-of-ruin sizing as scripts/bankroll_analysis.py,
but computed from theory instead of a (possibly noisy, possibly stake-specific)
empirical sample.

Important, and NOT a bug: at TRUE equilibrium (both players playing solved GTO)
this game is a small guaranteed loser for both seats (see the rake-free-vs-real
sanity check earlier in this project) -- so the theoretical baseline mean is
negative, and no finite bankroll avoids ruin at pure GTO-vs-GTO. Any real,
positive edge comes entirely from real opponents deviating from equilibrium
(confirmed empirically: they call your shoves with a weaker-than-optimal range),
which this script cannot know about by construction. So rather than pretend to
report ONE bankroll number, this reports a SENSITIVITY TABLE: the model's exact
variance, paired with a range of plausible edges (including the pure-equilibrium
baseline and the rough magnitudes suggested by the earlier empirical backtest),
so the bankroll figure can be read off honestly once/if the true edge size is
pinned down with more data.

SB and BB need genuinely different simulation logic, not a single generic
routine: shoving as SB means the OPPONENT'S calling decision is still live (draw
BB's hand fresh, then check BB's own range for a live continue/fold decision),
whereas calling as BB means the opponent (SB) has ALREADY shoved by definition of
the "BB|SB" info set -- so SB's hand must be drawn CONDITIONAL on having shoved
(weighted by combo_count * SB's own shove range, not a fresh uniform draw), and
there's no further continue/fold check on SB's side at all.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
import equity
import ev
import solver
import tree
from game import HU_CONFIG

RISK_TARGETS = [0.05, 0.01, 0.005, 0.001]
ASSUMED_EDGES_BB = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]  # bb/hand -- illustrative sensitivity range


def bankroll_for_risk(mean: float, var: float, risk: float) -> float:
    if mean <= 0:
        return float("inf")
    return -var * math.log(risk) / (2 * mean)


def _combo_probs() -> np.ndarray:
    weights = np.asarray(cards.CANON_COMBO_WEIGHT, dtype=np.float64)
    return weights / weights.sum()


def _win_lose_amounts(cfg) -> tuple[float, float]:
    win_amt = float(ev.showdown_ev_net_vec(cfg, 0.0, np.array([1.0]))[0])
    lose_amt = float(ev.showdown_ev_net_vec(cfg, 0.0, np.array([0.0]))[0])
    return win_amt, lose_amt


def simulate_sb(rng, n, cfg, table, sb_range, bb_range, fold_payoff, steal_ev) -> np.ndarray:
    """SB's decision: shove or fold. If shove, the OPPONENT's (BB's) hand is a
    fresh draw, and BB has a genuinely live call/fold decision depending on it."""
    combo_p = _combo_probs()
    win_amt, lose_amt = _win_lose_amounts(cfg)

    hero_idx = rng.choice(cards.NUM_CANON_HANDS, size=n, p=combo_p)
    hero_shoves = rng.random(n) < sb_range[hero_idx]

    outcomes = np.full(n, fold_payoff, dtype=np.float64)
    shoving = np.flatnonzero(hero_shoves)
    if shoving.size:
        bb_idx = rng.choice(cards.NUM_CANON_HANDS, size=shoving.size, p=combo_p)
        bb_calls = rng.random(shoving.size) < bb_range[bb_idx]

        contested = np.flatnonzero(bb_calls)
        uncontested = np.flatnonzero(~bb_calls)
        outcomes[shoving[uncontested]] = steal_ev
        if contested.size:
            eq = table[hero_idx[shoving[contested]], bb_idx[contested]]
            hero_wins = rng.random(contested.size) < eq
            outcomes[shoving[contested]] = np.where(hero_wins, win_amt, lose_amt)

    return outcomes


def simulate_bb(rng, n, cfg, table, bb_range, sb_range, fold_payoff) -> np.ndarray:
    """BB's decision: call or fold, facing an ALREADY-shoved SB. SB's hand must
    be drawn conditional on having shoved (weighted by SB's own shove range),
    not a fresh draw re-checked against SB's range -- SB already committed."""
    combo_p = _combo_probs()
    win_amt, lose_amt = _win_lose_amounts(cfg)

    sb_shove_weights = np.asarray(cards.CANON_COMBO_WEIGHT, dtype=np.float64) * sb_range
    sb_shove_p = sb_shove_weights / sb_shove_weights.sum()

    hero_idx = rng.choice(cards.NUM_CANON_HANDS, size=n, p=combo_p)
    hero_calls = rng.random(n) < bb_range[hero_idx]

    outcomes = np.full(n, fold_payoff, dtype=np.float64)
    calling = np.flatnonzero(hero_calls)
    if calling.size:
        sb_idx = rng.choice(cards.NUM_CANON_HANDS, size=calling.size, p=sb_shove_p)
        eq = table[hero_idx[calling], sb_idx]
        hero_wins = rng.random(calling.size) < eq
        outcomes[calling] = np.where(hero_wins, win_amt, lose_amt)

    return outcomes


def summarize(label: str, outcomes_bb: np.ndarray, bb_amount: float) -> None:
    outcomes_dollars = outcomes_bb * bb_amount
    n = len(outcomes_dollars)
    mean = outcomes_dollars.mean()
    var = outcomes_dollars.var(ddof=1)
    std = math.sqrt(var)

    print(f"\n{label} (theoretical, n={n:,} simulated hands, ${bb_amount:.0f}/bb):")
    print(f"  mean = ${mean:.4f}/hand ({mean / bb_amount:.4f}bb/hand)   std = ${std:.4f}/hand")
    print("  bankroll sensitivity: given this variance, at various ASSUMED edges (bb/hand):")
    for edge_bb in ASSUMED_EDGES_BB:
        edge_dollars = edge_bb * bb_amount
        label_edge = "pure GTO equilibrium" if edge_bb == 0.0 else f"+{edge_bb:.2f}bb/hand assumed edge"
        row = f"    {label_edge:32}"
        if edge_bb == 0.0 and mean <= 0:
            row += " -- mean <= 0, no finite bankroll avoids ruin"
            print(row)
            continue
        b5 = bankroll_for_risk(edge_dollars, var, 0.05)
        b1 = bankroll_for_risk(edge_dollars, var, 0.01)
        buyin = 8 * bb_amount
        row += f" 5% RoR=${b5:,.0f} ({b5 / buyin:.0f}x buyin)   1% RoR=${b1:,.0f} ({b1 / buyin:.0f}x buyin)"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--n-simulated", type=int, default=500_000)
    parser.add_argument("--bb-amount", type=float, default=2.0, help="currency value of 1bb, e.g. 2 for $1/$2, 5 for $2/$5")
    args = parser.parse_args()

    table = equity.load_or_build_equity_table(cache_dir=args.cache_dir)
    ranges, info = solver.solve(HU_CONFIG, table=table)
    print(f"solved HU_CONFIG at {HU_CONFIG.stack_bb}bb: converged={info['converged']}")

    infosets = tree.infosets_by_key(tree.build_infosets(HU_CONFIG))
    sb_infoset, bb_infoset = infosets["SB|"], infosets["BB|SB"]

    rng = np.random.default_rng(0)

    sb_fold_payoff = ev.fold_baseline_ev_net(HU_CONFIG, sb_infoset, ranges)
    sb_steal_ev = ev.steal_ev_net(HU_CONFIG, HU_CONFIG.blind_of("BB"))
    sb_outcomes = simulate_sb(
        rng, args.n_simulated, HU_CONFIG, table, ranges["SB|"], ranges["BB|SB"], sb_fold_payoff, sb_steal_ev
    )
    summarize("SB (open shove-or-fold)", sb_outcomes, args.bb_amount)

    bb_fold_payoff = ev.fold_baseline_ev_net(HU_CONFIG, bb_infoset, ranges)
    bb_outcomes = simulate_bb(rng, args.n_simulated, HU_CONFIG, table, ranges["BB|SB"], ranges["SB|"], bb_fold_payoff)
    summarize("BB (call-or-fold facing a shove)", bb_outcomes, args.bb_amount)

    combined = np.concatenate([sb_outcomes, bb_outcomes])
    summarize("Combined (SB+BB, equal mix)", combined, args.bb_amount)


if __name__ == "__main__":
    main()
