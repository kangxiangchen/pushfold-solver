"""Damped best-response iteration.

Nash equilibria exist for this game, but with 3+ players uniqueness is NOT guaranteed
-- naive best-response can even cycle. Damping (blending toward the best response
instead of jumping straight to it) is what makes iteration converge in practice. This
module finds *an* equilibrium, not provably *the* one; see README for why that's a
feature to discuss, not a flaw to hide.

The outer damped-update loop (`solve()` below) is fully seat-count-agnostic: it just
iterates whatever list `tree.build_infosets(cfg)` produces (2 info sets for HU, 14 for
4-max) and asks `ev.shove_ev_net_vec`/`ev.fold_baseline_ev_net` for each one's EV,
using the CURRENT ranges snapshot. Those two ev.py functions already handle every
info-set shape (any number of already-shoved opponents, any number of seats still to
act) via `ev._enumerate_realization_leaves` -- so nothing in this file needs to know
or care whether it's solving HU or 4-max. The only 4-max-specific behavior here is
constructing a real hand evaluator (needed once any info set's leaf enumeration hits
2+ live opponents, i.e. genuine multiway pots -- impossible in HU) and picking a
Monte Carlo sample count for those multiway leaves.

HU vs 4-max are two genuinely different numerical regimes, and `solve()` supports both:

  HU is DETERMINISTIC -- its equity comes from the exact 169x169 table, no sampling.
  Best response is exact every sweep, a decaying alpha drives it to a clean fixed point,
  and the final iterate IS the answer. This is the original, validated behavior and is
  preserved byte-for-byte by the default SolverConfig (alpha_min=0, return_averaged=False).

  4-max is STOCHASTIC -- multiway showdowns are Monte Carlo, so every best response is
  noisy, and the DOMINANT consequence is a quantal-response BIAS: each sweep's best response
  is a hard shove/fold threshold on a noisy MC EV, so the expected update is toward a smoothed
  (logit-like) response -- an equilibrium ~0.1bb off the true Nash. This bias is not variance;
  it shrinks only as the per-sweep MC noise does, i.e. by RAISING mc_iterations. That -- not the
  step schedule or the estimator -- is the primary lever (measured: exploitability 0.16 -> 0.045
  -> 0.020 at mc 500 -> 2000 -> 8000; see README's convergence section for the full story,
  including the wrong first guess that alpha_min/averaging were the fix -- they weren't).

  The remaining non-default knobs play supporting roles:
    - return_averaged (Polyak-Ruppert tail averaging): cleans up residual VARIANCE in the
      returned estimate. Useful but minor here, since the residual is mostly bias, not variance.
    - alpha_min: a step-size floor. Left at 0 for the production run -- flooring alpha actually
      HURT (it holds a wide, off-center orbit and breaks the averaging theory, which needs
      step -> 0). Kept as a knob but not part of the working recipe.
    - warm_start: seed from a prior approximate solve so a few hundred high-mc refinement sweeps
      suffice instead of a cold start -- the efficient path to the high-mc equilibrium.
  Additionally the exploitability METRIC is itself MC-noisy and upward-biased (it takes a max
  over a noisy quantity, Jensen), so it's measured once at high precision at the end (and again
  at 2x that) rather than gated on at a low-precision per-sweep value.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import cards
import ev
import equity
import tree
from game import GameConfig
from tree import InfoSet, RangeMap

logger = logging.getLogger(__name__)

_TIE_TOLERANCE = 1e-9


@dataclass
class SolverConfig:
    alpha: float = 0.4  # INITIAL damping factor, spec's 0.3-0.5 range, midpoint default
    alpha_decay: float = 0.1  # alpha_t = alpha / (1 + alpha_decay * (t-1)) -- a diminishing
    # step size (same idea as classical fictitious play) is needed to fully damp
    # oscillation at a mixed equilibrium (see solve() docstring). Empirically swept:
    # decay=1.0 (pure 1/t) shrinks alpha so fast it freezes before reaching the true
    # equilibrium neighborhood (exploitability stalled ~0.05); decay=0.02 barely shrinks
    # at all within a reasonable iteration budget (left a visible 2-cycle). decay=0.1
    # reached exploitability ~1e-5 in ~19k iterations, comfortably inside max_iterations.
    alpha_min: float = 0.0  # floor on the decayed alpha: alpha_t = max(alpha_min, decayed).
    # 0.0 (default) reproduces the pure-decay schedule exactly. NOTE: a positive floor was an
    # early (wrong) guess at fixing the 4-max solve -- it made exploitability WORSE (~0.21bb),
    # because a constant floor holds a wide, off-center orbit and breaks Polyak-Ruppert
    # averaging (which needs step -> 0). The real 4-max lever is per-sweep mc_iterations (see
    # module docstring). Left as a knob, but the production recipe keeps this at 0.
    epsilon: float = 1e-4  # max per-weight range-change convergence threshold
    exploitability_epsilon: float = 1e-3  # REQUIRED secondary gate, not just a diagnostic --
    # see solve() docstring: with a decaying alpha, max_delta alone can fall below epsilon
    # merely because the step size itself has shrunk, even while the ranges are still far
    # from a real best-response fixed point. Only trust "converged" when both are small.
    # NOTE: this per-sweep gate only makes sense for the noise-free HU solve. For 4-max the
    # per-sweep exploitability is MC-noisy and upward-biased, so the loop never gates on it;
    # a high-precision measurement (final_exploitability_mc) is taken once at the end.
    max_iterations: int = 50000  # safety cap, with margin above the ~19k iterations the
    # swept default schedule above needed empirically on a representative test table

    mc_iterations: int = 150  # samples per multiway_equity_mc call for 3+-way pots (HU
    # never uses this -- it has no info set with 2+ live opponents). Higher = sharper
    # per-sweep fixed point but linearly slower; tail-averaging absorbs the rest of the
    # noise, so this trades off against sweep count (see README's performance section).
    mc_iterations_refine: int | None = None  # DEPRECATED annealing knob (kept for
    # back-compat): once max_delta first drops below mc_refine_threshold, switch to this
    # many MC samples. Left off by default -- on the first real 4-max run it dominated
    # ~17.5h of wall clock for little benefit once alpha had already frozen. Prefer a
    # positive alpha_min + return_averaged instead.
    mc_refine_threshold: float = 5e-3

    # --- Polyak-Ruppert tail averaging (the estimator fix for the stochastic 4-max case) ---
    return_averaged: bool = False  # when True, return the AVERAGE of the post-burn-in
    # iterates instead of the final (noisy) iterate. Default False preserves the exact HU
    # behavior. Turn on for any MC solve: it recovers true mixed-strategy frequencies from
    # the 0/1 best-response flips, and even converts a residual limit cycle into its
    # (correct) midpoint. See solve() docstring.
    average_burn_in_frac: float = 0.5  # begin accumulating the tail average only after this
    # fraction of max_iterations has elapsed, so early transient iterates don't pollute it.

    # --- honest, high-precision final exploitability (see solve() / README) ---
    final_exploitability_mc: int = 20000  # MC samples for the ONE-TIME end-of-solve
    # exploitability measurement on the (averaged) ranges. Far higher than per-sweep
    # mc_iterations because the metric is upward-biased by ~1/sqrt(mc); it's also measured
    # at 2x this to expose the residual noise floor rather than trusting a single number.

    # --- checkpointing / observability for multi-day single-process runs ---
    checkpoint_path: str | None = None  # if set, pickle solver state here every
    # checkpoint_every sweeps (atomic temp-file+rename); resumed from on restart.
    checkpoint_every: int = 500
    log_every: int = 200  # emit a progress log line (iteration, alpha, max_delta,
    # avg_range_drift) every this many sweeps; 0 disables logging.


def shove_ev_net(
    cfg: GameConfig,
    infoset: InfoSet,
    ranges: RangeMap,
    table: np.ndarray,
    evaluator: equity.Evaluator | None = None,
    mc_iterations: int = 150,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """(ev_shove_vector, fold_payoff) for every one of the 169 hands at this info
    set, given the CURRENT ranges. Shared by best_response and the exploitability
    diagnostic so both dispatch identically. Both halves are fully general over any
    info-set shape -- see ev.shove_ev_net_vec / ev.fold_baseline_ev_net."""
    ev_shove = ev.shove_ev_net_vec(
        cfg, infoset, ranges, table, evaluator=evaluator, mc_iterations=mc_iterations, rng=rng
    )
    fold_payoff = ev.fold_baseline_ev_net(cfg, infoset, ranges)
    return ev_shove, fold_payoff


def best_response(
    cfg: GameConfig,
    infoset: InfoSet,
    ranges: RangeMap,
    table: np.ndarray,
    evaluator: equity.Evaluator | None = None,
    mc_iterations: int = 150,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pure best response per hand: 1.0 (shove/call) if EV beats folding, 0.0 if it
    doesn't, 0.5 only at an exact tie. In practice, damping (not this tolerance) is
    what reveals genuinely mixed hands -- see module docstring."""
    ev_shove, fold_payoff = shove_ev_net(
        cfg, infoset, ranges, table, evaluator=evaluator, mc_iterations=mc_iterations, rng=rng
    )
    return np.where(
        ev_shove > fold_payoff + _TIE_TOLERANCE,
        1.0,
        np.where(ev_shove < fold_payoff - _TIE_TOLERANCE, 0.0, 0.5),
    )


def damped_update(old: np.ndarray, br: np.ndarray, alpha: float) -> np.ndarray:
    return (1 - alpha) * old + alpha * br


def compute_exploitability(
    cfg: GameConfig,
    ranges: RangeMap,
    table: np.ndarray,
    infosets: list[InfoSet],
    evaluator: equity.Evaluator | None = None,
    mc_iterations: int = 150,
    rng: np.random.Generator | None = None,
) -> float:
    """Combo-weighted EV gain, summed over info sets, from switching each seat's
    CURRENT (possibly mixed) range to a pure best response against the others'
    current ranges. Zero at a true fixed point; reported as a diagnostic, not gated
    on, per spec ("optionally also require... exploitability < eps")."""
    total_gain = 0.0
    weight = np.asarray(cards.CANON_COMBO_WEIGHT, dtype=np.float64)
    for infoset in infosets:
        key = tree.infoset_key(infoset.seat, infoset.shoved_before)
        ev_shove, fold_payoff = shove_ev_net(
            cfg, infoset, ranges, table, evaluator=evaluator, mc_iterations=mc_iterations, rng=rng
        )
        current_range = ranges[key]
        value_current = current_range * ev_shove + (1 - current_range) * fold_payoff
        value_best_response = np.maximum(ev_shove, fold_payoff)
        gap = value_best_response - value_current
        total_gain += float(np.sum(gap * weight) / cards.NUM_COMBOS)
    return total_gain


def _alpha_at(solver_cfg: SolverConfig, iteration: int) -> float:
    """Decayed damping factor at a 1-indexed iteration, floored at alpha_min. With
    alpha_min=0 (the default) this is the original pure-decay schedule exactly."""
    decayed = solver_cfg.alpha / (1 + solver_cfg.alpha_decay * (iteration - 1))
    return max(solver_cfg.alpha_min, decayed)


def _save_checkpoint(path: str, state: dict) -> None:
    """Atomic pickle write (temp file + rename) so a crash mid-write can't corrupt the
    checkpoint a multi-day run depends on to resume."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    tmp.replace(p)


def _load_checkpoint(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def solve(
    cfg: GameConfig,
    solver_cfg: SolverConfig | None = None,
    table: np.ndarray | None = None,
    evaluator: equity.Evaluator | None = None,
    warm_start: RangeMap | None = None,
) -> tuple[RangeMap, dict]:
    """Damped best-response fixed point, for any GameConfig -- HU (2 info sets, never
    needs a real evaluator since no HU info set ever has 2+ live opponents) or 4-max
    (14 info sets, some of which do). A real `equity.Evaluator` is only constructed
    when `len(cfg.seats) > 2` and none was supplied, so HU callers (including every
    existing test, which passes a synthetic table) are completely unaffected by this
    generalization -- they never touch eval7/treys at all.

    With the default SolverConfig this is the original deterministic-HU behavior
    verbatim (pure-decay alpha, return the final iterate, gate on per-sweep
    exploitability). The non-default knobs (alpha_min, return_averaged,
    final_exploitability_mc, checkpointing) turn it into the stochastic-approximation
    scheme the noisy 4-max solve needs -- see the module docstring for the why of each.

    `warm_start` seeds the initial ranges from a prior (approximate) solution instead of
    the all-shove default -- standard practice for equilibrium refinement, and the
    efficient path for the 4-max production solve: a modest number of high-mc sweeps
    from an already-decent starting point drives the quantal-response bias down far
    faster than a cold start (see README's convergence section). It does NOT affect
    correctness -- the result's quality is judged by its own honestly-measured
    exploitability, independent of the seed. A resumed checkpoint takes precedence.
    """
    solver_cfg = solver_cfg or SolverConfig()
    if table is None:
        table = equity.load_or_build_equity_table()
    if evaluator is None and len(cfg.seats) > 2:
        evaluator = equity.get_evaluator()

    infosets = tree.build_infosets(cfg)
    keys = [tree.infoset_key(i.seat, i.shoved_before) for i in infosets]

    # Tail averaging accumulates a running SUM of post-burn-in iterates; the returned
    # estimate is that sum / count (Polyak-Ruppert). burn_in_iter is the first iteration
    # whose ranges are folded into the average.
    burn_in_iter = int(solver_cfg.average_burn_in_frac * solver_cfg.max_iterations)

    if warm_start is not None:
        ranges = {k: np.asarray(warm_start[k], dtype=np.float64).copy() for k in keys}
    else:
        ranges = tree.init_ranges(infosets)
    range_sum: RangeMap = {k: np.zeros(cards.NUM_CANON_HANDS) for k in keys}
    avg_count = 0
    start_iter = 1
    prev_avg: RangeMap | None = None
    avg_range_drift = float("inf")

    # Resume from a checkpoint if one exists (multi-day runs are crash/sleep-fragile).
    if solver_cfg.checkpoint_path is not None:
        ckpt = _load_checkpoint(solver_cfg.checkpoint_path)
        if ckpt is not None:
            ranges = ckpt["ranges"]
            range_sum = ckpt["range_sum"]
            avg_count = ckpt["avg_count"]
            start_iter = ckpt["iteration"] + 1
            logger.info("resumed 4-max solve from %s at iteration %d",
                        solver_cfg.checkpoint_path, ckpt["iteration"])

    max_delta = float("inf")
    iteration = start_iter - 1
    for iteration in range(start_iter, solver_cfg.max_iterations + 1):
        alpha_t = _alpha_at(solver_cfg, iteration)

        new_ranges: RangeMap = {}
        max_delta = 0.0
        for infoset, key in zip(infosets, keys):
            br = best_response(
                cfg, infoset, ranges, table, evaluator=evaluator,
                mc_iterations=solver_cfg.mc_iterations,
            )
            updated = damped_update(ranges[key], br, alpha_t)
            new_ranges[key] = updated
            max_delta = max(max_delta, float(np.abs(updated - ranges[key]).max()))
        ranges = new_ranges

        if solver_cfg.return_averaged and iteration > burn_in_iter:
            for key in keys:
                range_sum[key] += ranges[key]
            avg_count += 1

        # Deterministic-HU early stop: only meaningful when NOT averaging (the noisy
        # 4-max solve never gates on the per-sweep exploitability -- see docstring).
        if not solver_cfg.return_averaged and max_delta < solver_cfg.epsilon:
            expl = compute_exploitability(
                cfg, ranges, table, infosets, evaluator=evaluator,
                mc_iterations=solver_cfg.mc_iterations,
            )
            if expl < solver_cfg.exploitability_epsilon:
                break

        if solver_cfg.log_every and iteration % solver_cfg.log_every == 0:
            if avg_count > 0:
                cur_avg = {k: range_sum[k] / avg_count for k in keys}
                if prev_avg is not None:
                    avg_range_drift = max(
                        float(np.abs(cur_avg[k] - prev_avg[k]).max()) for k in keys
                    )
                prev_avg = cur_avg
            logger.info("iter=%d alpha=%.5f max_delta=%.2e avg_drift=%.2e",
                        iteration, alpha_t, max_delta, avg_range_drift)

        if solver_cfg.checkpoint_path is not None and iteration % solver_cfg.checkpoint_every == 0:
            _save_checkpoint(solver_cfg.checkpoint_path, {
                "ranges": ranges, "range_sum": range_sum,
                "avg_count": avg_count, "iteration": iteration,
            })

    # The returned estimate: the tail average (Polyak-Ruppert) if enabled, else the
    # final iterate. Averaging is the correct estimator under MC noise -- see docstring.
    if solver_cfg.return_averaged and avg_count > 0:
        ranges = {k: range_sum[k] / avg_count for k in keys}

    # Honest, high-precision final exploitability, measured ONCE on the returned ranges.
    # Also measured at 2x mc to expose the residual MC noise floor (the metric is
    # upward-biased by ~1/sqrt(mc)), so the reported gap isn't mistaken for a single
    # trustworthy point estimate. For deterministic HU both reads are identical/exact.
    hi_mc = solver_cfg.final_exploitability_mc
    exploitability_hi_mc = compute_exploitability(
        cfg, ranges, table, infosets, evaluator=evaluator, mc_iterations=hi_mc,
    )
    exploitability_hi_mc_2x = compute_exploitability(
        cfg, ranges, table, infosets, evaluator=evaluator, mc_iterations=2 * hi_mc,
    )

    info = {
        "iterations": iteration,
        "max_delta": max_delta,
        "final_alpha": _alpha_at(solver_cfg, iteration),
        "averaged": solver_cfg.return_averaged and avg_count > 0,
        "avg_count": avg_count,
        "avg_range_drift": avg_range_drift,
        # Primary honest gap + its noise floor (gap - gap_2x indicates residual bias):
        "exploitability_hi_mc": exploitability_hi_mc,
        "exploitability_hi_mc_2x": exploitability_hi_mc_2x,
        # Back-compat: existing HU tests/callers read info["exploitability"].
        "exploitability": exploitability_hi_mc,
        "converged": max_delta < solver_cfg.epsilon
        and exploitability_hi_mc < solver_cfg.exploitability_epsilon,
    }
    return ranges, info
