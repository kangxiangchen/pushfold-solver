"""Depth-bucketed solving and EV-loss grading of PushFoldSpots against the HU
solver's equilibrium ranges.

Solving is cheap (a fast vectorized numpy fixed point over 169-length arrays, see
solver.py's docstring), and the 169x169 equity table is already built and cached on
disk, so a fresh solve per distinct effective-stack depth encountered in a hand
history is fine -- no disk persistence of solved ranges is needed for this size of
workload.
"""

from __future__ import annotations

import dataclasses
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import cards
import ev
import solver
import tree
from game import FOURMAX_CONFIG, HU_CONFIG, GameConfig
from hand_history import ParsedHand
from pushfold_spot import PushFoldSpot, classify_bb_response_when_hero_is_sb


class StackTooSmallError(ValueError):
    """Raised when a spot's effective_stack_bb doesn't exceed the game's blinds --
    a degenerate/garbage parse, never expected from a real push/fold hand."""


def bucket_depth(effective_stack_bb: float) -> float:
    """Round to the nearest 0.1bb -- currency-derived stacks already cluster near
    round numbers, and this granularity is far finer than the EV-loss magnitudes
    this tool exists to detect, so bucketing many similar depths into one solve
    costs no measurable accuracy."""
    return round(effective_stack_bb * 10) / 10


@dataclass(frozen=True)
class _SolvedDepth:
    cfg: GameConfig
    ranges: dict


class SolveCache:
    """In-memory only -- see module docstring for why no disk persistence is needed.
    HU-only (always solves dataclasses.replace(HU_CONFIG, stack_bb=depth)) -- 4-max
    grading uses a separately solved (cfg, ranges) pair instead, since it only ever
    needs one depth (see fourmax_spot.py's module docstring) and solving it is far
    too expensive (tens of minutes to hours) to do lazily inside this cache."""

    def __init__(self, table: np.ndarray):
        self.table = table
        self._cache: dict[float, _SolvedDepth] = {}

    def get(self, effective_stack_bb: float) -> _SolvedDepth:
        depth = bucket_depth(effective_stack_bb)
        if depth <= max(HU_CONFIG.blinds.values()):
            raise StackTooSmallError(f"effective_stack_bb={effective_stack_bb} bucketed to {depth}, too small")
        if depth not in self._cache:
            cfg = dataclasses.replace(HU_CONFIG, stack_bb=depth)
            ranges, _info = solver.solve(cfg, table=self.table)
            self._cache[depth] = _SolvedDepth(cfg, ranges)
        return self._cache[depth]


@dataclass(frozen=True)
class GradedSpot:
    hand_id: str
    timestamp: str
    position: str  # "SB" or "BB"
    effective_stack_bb: float
    hero_hand: str  # canonical label, e.g. "A8o"
    hero_action: str
    gto_weight: float
    gto_recommended_action: str  # "shove_or_call" or "fold"
    ev_loss_bb: float
    predicted_ev_net_bb: float  # the model's ev_net for the action hero actually took
    model: str = "HU"  # "HU" or "4MAX" -- which solved GameConfig this was graded against


def grade_spot_against(
    spot: PushFoldSpot,
    cfg: GameConfig,
    ranges: dict,
    table: np.ndarray,
    evaluator=None,
    model: str = "HU",
    ev_memo: dict | None = None,
) -> GradedSpot:
    """Config-agnostic grading: works for any GameConfig/RangeMap pair, HU's 2 info
    sets or 4-max's 14 alike, since ev.shove_ev_net_vec/fold_baseline_ev_net are
    themselves already fully general (see ev.py's module docstring) -- there's no
    more "SB vs BB" branching needed, just a lookup of spot.info_set_key's InfoSet.
    `evaluator` is required only if grading ever hits a 2+-live-opponent leaf (i.e.
    4-max multiway); HU spots never need it.

    `ev_memo` (optional, caller-owned dict) caches the (ev_go vector, ev_fold) pair
    per (model, stack depth, info set) -- neither depends on hero's hand, so grading
    thousands of spots against the same ranges costs at most one vector computation
    per info set instead of one per spot. That's what makes bulk 4-max grading
    tractable: each multiway shove_ev_net_vec is a Monte Carlo evaluation, minutes
    across a big export when recomputed per spot, seconds when memoized. Sharing one
    MC draw per info set also makes grades within a run consistent with each other."""
    hero_label = cards.hero_canon_label(*spot.hero_hand)
    hero_idx = cards.canon_index(hero_label)

    memo_key = (model, cfg.stack_bb, spot.info_set_key)
    if ev_memo is not None and memo_key in ev_memo:
        ev_go_vec, ev_fold = ev_memo[memo_key]
    else:
        infoset = tree.infosets_by_key(tree.build_infosets(cfg))[spot.info_set_key]
        ev_go_vec = ev.shove_ev_net_vec(cfg, infoset, ranges, table, evaluator=evaluator)
        ev_fold = ev.fold_baseline_ev_net(cfg, infoset, ranges)
        if ev_memo is not None:
            ev_memo[memo_key] = (ev_go_vec, ev_fold)

    ev_go = float(ev_go_vec[hero_idx])
    gto_weight = float(ranges[spot.info_set_key][hero_idx])
    position = spot.info_set_key.split("|", 1)[0]

    gto_recommended_action = "shove_or_call" if ev_go > ev_fold else "fold"
    hero_ev = ev_go if spot.hero_action == "shove_or_call" else ev_fold
    ev_loss_bb = max(ev_go, ev_fold) - hero_ev

    return GradedSpot(
        hand_id=spot.hand_id,
        timestamp=spot.timestamp,
        position=position,
        effective_stack_bb=spot.effective_stack_bb,
        hero_hand=hero_label,
        hero_action=spot.hero_action,
        gto_weight=gto_weight,
        gto_recommended_action=gto_recommended_action,
        ev_loss_bb=ev_loss_bb,
        predicted_ev_net_bb=hero_ev,
        model=model,
    )


def grade_spot(spot: PushFoldSpot, cache: SolveCache) -> GradedSpot:
    solved = cache.get(spot.effective_stack_bb)
    return grade_spot_against(spot, solved.cfg, solved.ranges, cache.table)


def fourmax_ranges_cache_path(cache_dir: str, stack_bb: float) -> Path:
    return Path(cache_dir) / f"fourmax_ranges_{stack_bb:.1f}bb.pkl"


def load_or_solve_fourmax_ranges(
    table: np.ndarray,
    evaluator,
    stack_bb: float = 8.0,
    cache_dir: str = "cache/",
    solver_cfg: solver.SolverConfig | None = None,
) -> tuple[GameConfig, dict]:
    """Loads a previously solved FOURMAX_CONFIG RangeMap from disk if present,
    else solves it fresh (tens of minutes to hours -- see README's performance
    section) and caches the result to disk for next time. Only ONE depth is ever
    needed in practice, not SolveCache's per-hand bucketing: every real multiway
    push/fold hand this backtest has been run against has an effective_stack_bb
    of exactly 8.0bb (see fourmax_spot.py's module docstring), so a single
    solve-once-and-persist is enough to grade all of them."""
    path = fourmax_ranges_cache_path(cache_dir, stack_bb)
    cfg = dataclasses.replace(FOURMAX_CONFIG, stack_bb=stack_bb)
    if path.exists():
        with open(path, "rb") as f:
            return cfg, pickle.load(f)

    ranges, _info = solver.solve(cfg, solver_cfg, table=table, evaluator=evaluator)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ranges, f)
    return cfg, ranges


def realized_ev_net_bb(hand: ParsedHand, spot: PushFoldSpot, cfg: GameConfig) -> float:
    """The REAL observed ev_net for this specific hand's outcome -- distinct from
    grade_spot's predicted_ev_net_bb, which is an expectation over the opponent's
    whole equilibrium range. Averaged over many hands, this should converge toward
    the model's prediction if the model is accurate (law of large numbers) --
    that comparison is exactly what scripts/validate_model.py checks.

    Hero's contribution uses the MODEL's own convention (the solved cfg's
    stack_bb/blind_of), not a re-derivation from the hand's raw RETURN/ALLIN
    deltas -- CoinPoker's own uncalled-bet-return accounting has client-specific
    quirks that don't cleanly reduce to "return everything beyond the blinds", so
    anchoring to the model's convention (which is what's actually being validated)
    is both simpler and more robust than trying to reverse-engineer it. Hero's
    actual winnings (collected_by), by contrast, are unambiguous ground truth and
    used directly.

    Rakeback is deterministic here, not an expectation: for the clean 2-seat HU
    spots this backtest grades, dead_money_bb is always 0 (see pushfold_spot's
    documented approximation), so total_pot_bb depends only on the actual
    resolution (fold, uncontested shove, or contested showdown), never on whose
    hand wins -- exactly mirroring ev.showdown_ev_net_vec's, ev.steal_ev_net's,
    and ev.fold_baseline_ev_net's own rakeback terms for these HU-only cases.

    Critically, a SB shove doesn't always reach a real showdown -- BB may just
    fold to it, in which case Hero's bet was never called and contributes 0 to
    the formed pot (an uncalled bet is never part of it, same as Fixture B in
    tests/test_ev.py), NOT the full stack_bb a naive "hero shoved -> contributed
    stack_bb" assumption would give. classify_bb_response_when_hero_is_sb tells
    us which of the two actually happened. A BB call, by contrast, only ever
    exists in this dataset as the terminal preflop action, so it's always a
    genuine 2-way showdown -- no equivalent check is needed on that side.
    """
    hero_name = hand.hero_name or "Hero"
    hero_collected_bb = hand.collected_by.get(hero_name, 0.0) / hand.bb_amount

    position = "SB" if spot.info_set_key == "SB|" else "BB"
    if spot.hero_action == "fold":
        contribution_bb = cfg.blind_of(position)
        total_pot_bb = contribution_bb
    elif position == "SB" and classify_bb_response_when_hero_is_sb(hand, hero_name=hero_name) != "call":
        # Hero's shove was folded to (or the resolution is ambiguous) -- uncontested,
        # Hero's own bet was never called, so it contributes 0 (ev.steal_ev_net).
        contribution_bb = 0.0
        total_pot_bb = cfg.blind_of("BB")
    else:
        contribution_bb = cfg.stack_bb
        total_pot_bb = 2 * cfg.stack_bb

    ledger_net_bb = hero_collected_bb - contribution_bb
    rakeback_bb = cfg.rakeback_pool_bb * (contribution_bb / total_pot_bb) if total_pot_bb > 0 else 0.0
    return ledger_net_bb + rakeback_bb
