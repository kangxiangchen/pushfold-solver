"""Population census: how the OPPONENT pool actually plays each push/fold info set,
measured from villain action labels alone (no hole cards needed), compared against
the solved chart's own frequencies. Hero's decisions are excluded from every count
-- the pool must not be contaminated by hero's own play (scripts/backtest.py grades
that side) -- but hero's shoves still advance the shoved-set context that later
villains' info sets condition on.

Population-level is the ONLY level possible with this data: CoinPoker rotates the
anonymized villain IDs every hand (measured: 32k+ distinct IDs with one decision
each across 16.6k hands), so per-player profiling is structurally impossible and
`VillainDecision.actor` exists solely for the same-hand shown-cards join.

Shown-card composition (shown_composition) carries a SELECTION BIAS that callers
must surface wherever it is rendered: hole cards are revealed only at showdowns,
so uncalled steals are never observed, players may still muck at showdown, and
folded hands are invisible. It is a biased, qualitative check on assumptions like
exploit.py's top-of-range fill -- never an unbiased range estimate.

This walk is a deliberate sibling of fourmax_spot.classify_fourmax_realization,
not a generalization of it: that walk is hero-relative and all-or-nothing (one
unmappable action voids the hand), while the census records one observation per
VILLAIN actor and keeps the clean PREFIX before any unmappable action -- a
villain's fold at UTG| is a valid observation even if a later player limps,
because its context depends only on the (pure shove/fold) actions before it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import cards
import tree
from fourmax_spot import _assign_fourmax_roles
from game import FOURMAX_CONFIG
from hand_history import ParsedHand
from pushfold_spot import MAX_PUSHFOLD_STACK_BB, _is_full_shove

INFOSET_ORDER: tuple[str, ...] = tuple(
    tree.infoset_key(i.seat, i.shoved_before) for i in tree.build_infosets(FOURMAX_CONFIG)
)
VALID_KEYS: frozenset[str] = frozenset(INFOSET_ORDER)

# A shown hand counts as "inside the GTO range" when its solved weight is at least
# this -- solved vectors are near-saturated 0/1, so the exact threshold hardly moves
# the count; 0.5 splits genuinely mixed hands evenly.
IN_RANGE_THRESHOLD = 0.5


@dataclass(frozen=True)
class VillainDecision:
    hand_id: str
    bb_amount: float
    effective_stack_bb: float
    info_set_key: str  # tree.infoset_key(role, shoved_before at the decision)
    actor: str  # per-hand villain ID (rotates every hand; only for the shown join)
    action: str  # "go" (full shove or closing call) | "fold"
    shown: tuple[str, str] | None  # hand.shown_cards.get(actor)


def villain_decisions(hand: ParsedHand, hero_name: str = "Hero") -> list[VillainDecision]:
    """One observation per villain shove/fold decision in this hand; [] when the
    hand is out of scope (antes, missing blinds, unassignable roles, too deep, or
    any unparseable action verb -- which could hide a real action anywhere)."""
    if hand.ante_posters or hand.sb_poster is None or hand.bb_poster is None:
        return []
    if hand.unclassified_action_names:
        return []
    roles = _assign_fourmax_roles(hand)
    if roles is None:
        return []
    stack_of = {s.name: s.starting_stack for s in hand.seats}
    if any(name not in stack_of for name in roles):
        return []
    effective_stack_bb = min(stack_of[name] for name in roles) / hand.bb_amount
    if effective_stack_bb > MAX_PUSHFOLD_STACK_BB:
        return []

    observations: list[VillainDecision] = []
    shoved: set[str] = set()

    def record(name: str, role: str, action: str) -> bool:
        key = tree.infoset_key(role, frozenset(shoved))
        if key not in VALID_KEYS:
            return False
        observations.append(VillainDecision(
            hand_id=hand.hand_id,
            bb_amount=hand.bb_amount,
            effective_stack_bb=effective_stack_bb,
            info_set_key=key,
            actor=name,
            action=action,
            shown=hand.shown_cards.get(name),
        ))
        return True

    for a in hand.preflop_actions:
        if a.action == "return":
            continue
        role = roles.get(a.name)
        if role is None:
            break
        if a.action in ("checks", "bets", "post"):
            break
        is_shove = _is_full_shove(a, stack_of)
        is_closing_call = a.action == "calls" and bool(shoved)
        if (a.action == "raises" and not is_shove) or (a.action == "calls" and not shoved):
            break

        if a.action == "folds":
            if a.name != hero_name and not record(a.name, role, "fold"):
                break
            continue
        if is_shove or is_closing_call:
            if a.name != hero_name and not record(a.name, role, "go"):
                break
            shoved.add(role)
            continue

    return observations


def census(hands: Iterable[ParsedHand], hero_name: str = "Hero") -> list[VillainDecision]:
    """All villain decisions across `hands`, deduped by hand_id (first occurrence
    wins -- overlapping exports must not double-count the pool)."""
    seen: set[str] = set()
    decisions: list[VillainDecision] = []
    for hand in hands:
        if hand.hand_id in seen:
            continue
        seen.add(hand.hand_id)
        decisions.extend(villain_decisions(hand, hero_name=hero_name))
    return decisions


def _freq_and_se(go: int, n: int) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = go / n
    se = math.sqrt(p * (1 - p) / n)
    return p, (se if se > 0 else None)


@dataclass(frozen=True)
class StakePoolStats:
    bb_amount: float
    n: int
    go: int

    @property
    def pool_freq(self) -> float | None:
        return _freq_and_se(self.go, self.n)[0]

    @property
    def se(self) -> float | None:
        return _freq_and_se(self.go, self.n)[1]


@dataclass(frozen=True)
class InfosetPoolStats:
    key: str
    n: int
    go: int
    per_stake: tuple[StakePoolStats, ...]  # sorted by bb_amount

    @property
    def pool_freq(self) -> float | None:
        return _freq_and_se(self.go, self.n)[0]

    @property
    def se(self) -> float | None:
        return _freq_and_se(self.go, self.n)[1]


def summarize(decisions: list[VillainDecision]) -> list[InfosetPoolStats]:
    """One entry per info set in tree order -- n=0 rows included so renderers get a
    stable 14-row table."""
    counts: dict[str, dict[float, list[int]]] = {}  # key -> bb -> [n, go]
    for d in decisions:
        cell = counts.setdefault(d.info_set_key, {}).setdefault(d.bb_amount, [0, 0])
        cell[0] += 1
        if d.action == "go":
            cell[1] += 1
    stats: list[InfosetPoolStats] = []
    for key in INFOSET_ORDER:
        per_bb = counts.get(key, {})
        per_stake = tuple(
            StakePoolStats(bb_amount=bb, n=cell[0], go=cell[1])
            for bb, cell in sorted(per_bb.items())
        )
        stats.append(InfosetPoolStats(
            key=key,
            n=sum(st.n for st in per_stake),
            go=sum(st.go for st in per_stake),
            per_stake=per_stake,
        ))
    return stats


@dataclass(frozen=True)
class InfosetDeviation:
    stats: InfosetPoolStats
    gto_freq: float | None  # None when the solved ranges are unavailable
    z: float | None  # (pool - gto) / se; None when either input is undefined


def compare_to_gto(stats: list[InfosetPoolStats], ranges: Mapping | None) -> list[InfosetDeviation]:
    deviations: list[InfosetDeviation] = []
    for s in stats:
        gto = float(cards.range_average_weight(ranges[s.key])) if ranges and s.key in ranges else None
        z = None
        if gto is not None and s.pool_freq is not None and s.se is not None:
            z = (s.pool_freq - gto) / s.se
        deviations.append(InfosetDeviation(stats=s, gto_freq=gto, z=z))
    return deviations


@dataclass(frozen=True)
class ShownComposition:
    """What the pool's REVEALED shoves/calls looked like at one info set, vs the
    solved range -- see the module docstring's selection-bias warning."""

    key: str
    shown_go: int  # "go" decisions whose cards were revealed
    in_gto: int  # of those, hands the solved range plays >= IN_RANGE_THRESHOLD
    mean_gto_weight: float | None  # threshold-free summary of the same sample
    top_out_of_range: tuple[tuple[str, int], ...]  # (canon label, count), most common first


def shown_composition(decisions: list[VillainDecision], ranges: Mapping, top_n: int = 3) -> list[ShownComposition]:
    """Only info sets with at least one revealed 'go' decision appear (tree order)."""
    by_key: dict[str, list[tuple[str, float]]] = {}  # key -> [(canon label, gto weight)]
    for d in decisions:
        if d.action != "go" or d.shown is None or d.info_set_key not in ranges:
            continue
        label = cards.hero_canon_label(*d.shown)
        weight = float(ranges[d.info_set_key][cards.canon_index(label)])
        by_key.setdefault(d.info_set_key, []).append((label, weight))

    comps: list[ShownComposition] = []
    for key in INFOSET_ORDER:
        sample = by_key.get(key)
        if not sample:
            continue
        out_counts: dict[str, int] = {}
        for label, weight in sample:
            if weight < IN_RANGE_THRESHOLD:
                out_counts[label] = out_counts.get(label, 0) + 1
        top = tuple(sorted(out_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n])
        comps.append(ShownComposition(
            key=key,
            shown_go=len(sample),
            in_gto=sum(1 for _, w in sample if w >= IN_RANGE_THRESHOLD),
            mean_gto_weight=sum(w for _, w in sample) / len(sample),
            top_out_of_range=top,
        ))
    return comps
