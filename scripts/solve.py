#!/usr/bin/env python3
"""Entrypoint for the HU solve: config -> solve -> charts. HU is a fast, deterministic,
seconds-long job (exact equity table, no sampling).

For 4-max, use scripts/solve_fourmax.py instead: it's a multi-hour-to-multi-day
STOCHASTIC solve (Monte Carlo multiway showdowns) that needs Polyak-Ruppert averaging, a
floored step size, checkpointing, and a high-precision final exploitability measurement --
none of which this simple entrypoint sets up. Running 4-max through here with the default
config would silently reproduce the "looks converged but is still ~0.1bb exploitable"
freeze this project specifically diagnosed and fixed (see README's convergence section)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
import equity
import solver
import tree
import viz
from game import HU_CONFIG

CONFIGS = {
    "hu": lambda: HU_CONFIG,
    "hu-rake-free": lambda: HU_CONFIG.rake_free(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=list(CONFIGS), default="hu")
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--out-dir", default="cache/")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]()

    ev_backend = equity.get_evaluator()
    equity.smoke_check(ev_backend)
    table = equity.load_or_build_equity_table(cache_dir=args.cache_dir)

    ranges, info = solver.solve(cfg, table=table, evaluator=ev_backend)
    print(f"config={args.config}")
    print(f"converged={info['converged']} iterations={info['iterations']} "
          f"max_delta={info['max_delta']:.2e} exploitability={info['exploitability']:.2e}")

    for infoset in tree.build_infosets(cfg):
        key = tree.infoset_key(infoset.seat, infoset.shoved_before)
        shove_pct = 100 * (ranges[key] * cards.CANON_COMBO_WEIGHT).sum() / cards.NUM_COMBOS
        print(f"{key}: {shove_pct:.1f}% of combos shove/call")

    out_path = viz.plot_hu_ranges(ranges, cfg, out_dir=args.out_dir)
    print(f"Heatmap saved to {out_path}")


if __name__ == "__main__":
    main()
