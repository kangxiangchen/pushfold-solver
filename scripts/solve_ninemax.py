#!/usr/bin/env python3
"""Production 9-max solver run: resumable, checkpointed, observable.

Same stochastic-approximation machinery as scripts/solve_fourmax.py (Monte Carlo
multiway showdowns, Polyak-Ruppert tail averaging -- see src/solver.py's module
docstring and the README's convergence section), but for the full-ring 9-max game
(NINEMAX_CONFIG: 9 seats, 5bb, 2.5%-of-pot rake capped at 0.5bb, 80% rakeback, no rake
on uncontested pots). 9-max is a MUCH larger solve than 4-max -- 510 info sets vs 14,
with up to ~2,300 realization leaves per sweep -- so a within-sweep multiway-equity memo
(solver passes a fresh cache each sweep) is the main thing keeping it tractable.

Tractability (a 9-max sweep is ~2,000 multiway equity evaluations, vs 16 for 4-max):
  - --workers fans each sweep's 510 independent best responses across CPU cores (the
    main speedup; defaults to all cores).
  - --mc-deep caps the MC samples on deep (4+-way) pots, which are the priciest to
    sample and, at tight 5bb ranges, rare -- so a coarse equity for them barely moves
    the EV while cutting the bulk of the cost.

Recommended recipe (default flags already enable both optimizations):
  1. Coarse:  caffeinate -i python scripts/solve_ninemax.py --mc-iterations 600 \
                  --max-iterations 4000
  2. Refine:  caffeinate -i python scripts/solve_ninemax.py --mc-iterations 6000 \
                  --max-iterations 1500 --warm-start-from cache/ninemax_ranges.pkl

If it dies (or the machine sleeps despite caffeinate), rerun the SAME command -- it
resumes from cache/ninemax_checkpoint.pkl automatically. Unlike the 4-max runner this
one writes NO matplotlib PNG (510 panels don't fit one figure); use
scripts/export_range_viewer.py --config ninemax to view the solved ranges.
"""
import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
import equity
import solver
import tree
from game import NINEMAX_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--alpha-decay", type=float, default=0.1)
    parser.add_argument("--alpha-min", type=float, default=0.05,
                        help="floor on the decayed step size -- must be >0 for a stochastic "
                             "(multiway) solve or the ranges freeze before converging.")
    parser.add_argument("--mc-iterations", type=int, default=600,
                        help="Monte Carlo samples per multiway (3+-way) showdown, per sweep.")
    parser.add_argument("--max-iterations", type=int, default=6000)
    parser.add_argument("--burn-in-frac", type=float, default=0.5,
                        help="start Polyak tail-averaging after this fraction of max-iterations.")
    parser.add_argument("--final-mc", type=int, default=4000,
                        help="MC samples for the one-time high-precision final exploitability "
                             "(measured serially over all 510 info sets, so kept modest).")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="processes to fan each sweep's per-info-set best responses across "
                             "(the main 9-max speedup; 1 = single-process).")
    parser.add_argument("--mc-deep", type=int, default=80,
                        help="capped MC samples for deep (>= --deep-threshold live opponents) "
                             "pots -- rare at 5bb, so a coarse equity is cheap and barely moves "
                             "the EV. Pass 0 to disable the cap (exact mc for every pot).")
    parser.add_argument("--deep-threshold", type=int, default=4,
                        help="a pot counts as 'deep' at or above this many live opponents.")
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--out-dir", default="cache/")
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--warm-start-from", default=None,
                        help="pickle of a prior solve ({'ranges': RangeMap} or a bare RangeMap) "
                             "to seed from -- the efficient path for high-mc refinement.")
    args = parser.parse_args()

    checkpoint_path = str(Path(args.cache_dir) / "ninemax_checkpoint.pkl")
    log_path = str(Path(args.out_dir) / "ninemax_solve.log")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )

    table = equity.load_or_build_equity_table(cache_dir=args.cache_dir)
    ev_backend = equity.get_evaluator()
    equity.smoke_check(ev_backend)

    solver_cfg = solver.SolverConfig(
        alpha=args.alpha,
        alpha_decay=args.alpha_decay,
        alpha_min=args.alpha_min,
        mc_iterations=args.mc_iterations,
        max_iterations=args.max_iterations,
        average_burn_in_frac=args.burn_in_frac,
        return_averaged=True,
        final_exploitability_mc=args.final_mc,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        n_workers=args.workers,
        mc_iterations_deep=(args.mc_deep or None),
        deep_opponent_threshold=args.deep_threshold,
    )

    warm_start = None
    if args.warm_start_from is not None:
        with open(args.warm_start_from, "rb") as f:
            loaded = pickle.load(f)
        warm_start = loaded["ranges"] if isinstance(loaded, dict) and "ranges" in loaded else loaded
        logging.info("warm-starting from %s", args.warm_start_from)

    infosets = tree.build_infosets(NINEMAX_CONFIG)
    logging.info("starting 9-max production solve: info_sets=%d max_iterations=%d mc=%d "
                 "mc_deep=%s deep_threshold=%d workers=%d alpha_min=%.3f warm_start=%s",
                 len(infosets), args.max_iterations, args.mc_iterations, args.mc_deep or "off",
                 args.deep_threshold, args.workers, args.alpha_min,
                 args.warm_start_from is not None)
    t0 = time.time()
    ranges, info = solver.solve(NINEMAX_CONFIG, solver_cfg, table=table, evaluator=ev_backend,
                                warm_start=warm_start)
    elapsed = time.time() - t0
    logging.info("solve finished in %.1f min", elapsed / 60)

    out_pkl = str(Path(args.out_dir) / "ninemax_ranges.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({"ranges": ranges, "info": info, "elapsed_sec": elapsed}, f)

    logging.info("iterations=%d averaged=%s avg_count=%d",
                 info["iterations"], info["averaged"], info["avg_count"])
    logging.info("exploitability: hi_mc=%.4f  2x_mc=%.4f  (noise floor ~ their gap)",
                 info["exploitability_hi_mc"], info["exploitability_hi_mc_2x"])

    # 510 info sets is too many to log one-per-line usefully; report the open-shove
    # ranges (folded to hero) -- the most-used chart -- and leave the rest to the viewer.
    logging.info("--- open-shove ranges (folded to hero) ---")
    for infoset in infosets:
        if infoset.shoved_before:
            continue
        key = tree.infoset_key(infoset.seat, infoset.shoved_before)
        shove_pct = 100 * (ranges[key] * cards.CANON_COMBO_WEIGHT).sum() / cards.NUM_COMBOS
        logging.info("%-6s open  %.1f%% of combos shove", infoset.seat, shove_pct)

    logging.info("ranges pickled to %s ; view with: "
                 "python scripts/export_range_viewer.py --config ninemax", out_pkl)


if __name__ == "__main__":
    main()
