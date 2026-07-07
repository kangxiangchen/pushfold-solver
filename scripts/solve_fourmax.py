#!/usr/bin/env python3
"""Production 4-max solver run: resumable, checkpointed, observable.

The 4-max solve is a multi-hour-to-multi-day stochastic-approximation run (Monte Carlo
multiway showdowns, Polyak-Ruppert tail averaging -- see src/solver.py's module
docstring and the README's convergence section). Unlike the HU solve it is NOT a
seconds-long job, so this runner exists separately from scripts/solve.py to add the
things a long run needs: periodic disk checkpoints (crash/sleep-safe, auto-resumed),
progress logging to a file, and a high-precision final exploitability report.

Launch it caffeinated so the machine doesn't sleep mid-run, e.g.:
    caffeinate -i python scripts/solve_fourmax.py --max-iterations 40000

If it dies (or the machine sleeps despite caffeinate), just rerun the SAME command --
it resumes from the checkpoint at cache/fourmax_checkpoint.pkl automatically.
"""
import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
import equity
import solver
import tree
import viz
from game import FOURMAX_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Convergence-method knobs. Defaults are the values validated in the Gate C tuning
    # sweep (see the README's performance/convergence section); override to experiment.
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--alpha-decay", type=float, default=0.1)
    parser.add_argument("--alpha-min", type=float, default=0.05,
                        help="floor on the decayed step size -- MUST be >0 for 4-max or the "
                             "ranges freeze before converging (see README).")
    parser.add_argument("--mc-iterations", type=int, default=400,
                        help="Monte Carlo samples per multiway (3+-way) showdown, per sweep.")
    parser.add_argument("--max-iterations", type=int, default=40000)
    parser.add_argument("--burn-in-frac", type=float, default=0.5,
                        help="start Polyak tail-averaging after this fraction of max-iterations.")
    parser.add_argument("--final-mc", type=int, default=40000,
                        help="MC samples for the one-time high-precision final exploitability.")
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--out-dir", default="cache/")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--warm-start-from", default=None,
                        help="pickle of a prior solve ({'ranges': RangeMap} or a bare RangeMap) "
                             "to seed from -- the efficient path for high-mc refinement.")
    args = parser.parse_args()

    checkpoint_path = str(Path(args.cache_dir) / "fourmax_checkpoint.pkl")
    log_path = str(Path(args.out_dir) / "fourmax_solve.log")
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
    )

    warm_start = None
    if args.warm_start_from is not None:
        with open(args.warm_start_from, "rb") as f:
            loaded = pickle.load(f)
        warm_start = loaded["ranges"] if isinstance(loaded, dict) and "ranges" in loaded else loaded
        logging.info("warm-starting from %s", args.warm_start_from)

    logging.info("starting 4-max production solve: max_iterations=%d mc=%d alpha_min=%.3f warm_start=%s",
                 args.max_iterations, args.mc_iterations, args.alpha_min, args.warm_start_from is not None)
    t0 = time.time()
    ranges, info = solver.solve(FOURMAX_CONFIG, solver_cfg, table=table, evaluator=ev_backend,
                                warm_start=warm_start)
    elapsed = time.time() - t0
    logging.info("solve finished in %.1f min", elapsed / 60)

    # Persist the solved ranges + full info dict alongside the elapsed wall clock.
    out_pkl = str(Path(args.out_dir) / "fourmax_ranges.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({"ranges": ranges, "info": info, "elapsed_sec": elapsed}, f)

    logging.info("iterations=%d averaged=%s avg_count=%d", info["iterations"], info["averaged"], info["avg_count"])
    logging.info("exploitability: hi_mc=%.4f  2x_mc=%.4f  (noise floor ~ their gap)",
                 info["exploitability_hi_mc"], info["exploitability_hi_mc_2x"])
    for infoset in tree.build_infosets(FOURMAX_CONFIG):
        key = tree.infoset_key(infoset.seat, infoset.shoved_before)
        shove_pct = 100 * (ranges[key] * cards.CANON_COMBO_WEIGHT).sum() / cards.NUM_COMBOS
        logging.info("%-14s %.1f%% of combos shove/call", key, shove_pct)

    out_png = viz.plot_fourmax_ranges(ranges, FOURMAX_CONFIG, out_dir=args.out_dir)
    logging.info("heatmap saved to %s ; ranges pickled to %s", out_png, out_pkl)


if __name__ == "__main__":
    main()
