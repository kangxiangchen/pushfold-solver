#!/usr/bin/env python3
"""One-time (~1-4 hour, 8-core-parallelized) build of the full 169x169 exact heads-up
equity table. Deliberately separate from solve.py/pytest so nobody triggers a
multi-hour job by accident -- run this once, then load_or_build_equity_table() just
loads the cached .npy from then on.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
import equity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--processes", type=int, default=None, help="default: os.cpu_count()")
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if a cache file already exists"
    )
    args = parser.parse_args()

    cache_path = equity._cache_path(args.cache_dir)
    if cache_path.exists() and not args.force:
        print(f"Cache already exists at {cache_path} (use --force to rebuild). Nothing to do.")
        return

    ev = equity.get_evaluator()
    equity.smoke_check(ev)
    print(f"Evaluator OK ({type(ev).__name__}). Building full 169x169 equity table...")

    start = time.time()
    table = equity.build_equity_table(cache_dir=args.cache_dir, processes=args.processes)
    elapsed = time.time() - start

    print(f"Done in {elapsed / 60:.1f} min. Saved to {cache_path}")
    aa_kk = table[cards.canon_index("AA"), cards.canon_index("KK")]
    print(f"Spot check: AA vs KK equity = {aa_kk:.4f} (expect ~0.82)")


if __name__ == "__main__":
    main()
