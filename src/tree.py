"""Information-set tree: seat-count-agnostic construction.

For N seats acting in order, the seat at 0-indexed position k faces exactly 2^k
possible "who has shoved among the k earlier seats" situations, EXCEPT the last
seat, which faces 2^k - 1 (the all-folded/empty subset is a walk with no decision --
everyone folded, this seat wins uncontested, nothing to solve). This one function
produces BOTH the HU tree and the 4-max tree purely by varying cfg.seats -- no
seat-count branching beyond "is this the last seat".

Verified: 4-max = 2^0+2^1+2^2+(2^3-1) = 1+2+4+7 = 14 info sets (matches the spec's
14-info-set enumeration). HU (2 seats) = 2^0+(2^1-1) = 1+1 = 2 info sets (matches the
spec's "clean two-range fixed point").
"""

from dataclasses import dataclass
from itertools import combinations

from game import GameConfig

RangeMap = dict  # str -> np.ndarray[169]; see init_ranges. Kept untyped here to avoid
# an unnecessary numpy import in a module that's otherwise pure structure/combinatorics.


@dataclass(frozen=True)
class InfoSet:
    seat: str
    seat_pos: int  # 0-indexed position in cfg.seats
    shoved_before: frozenset  # subset of earlier seats (by action order) who already shoved
    facing_decision: str  # "open" if shoved_before is empty, else "call"
    dead_money_bb: float  # blinds forfeited by seats that folded strictly before this one
    is_last_seat: bool  # True iff this is cfg.seats[-1] (BB in both HU and 4-max)


def infoset_key(seat: str, shoved_before: frozenset) -> str:
    """Deterministic, human-readable dict key, e.g. 'BB|SB' (BB facing SB having
    shoved) or 'SB|' (SB's open, nobody shoved before them)."""
    return f"{seat}|{'|'.join(sorted(shoved_before))}"


def _all_subsets(items: tuple) -> list[frozenset]:
    result = []
    for size in range(len(items) + 1):
        for combo in combinations(items, size):
            result.append(frozenset(combo))
    return result


def build_infosets(cfg: GameConfig) -> list[InfoSet]:
    """General construction: iterate each seat's position k, enumerate subsets of the
    k earlier seats (all subsets if not the last seat; non-empty subsets only if it
    is), and compute dead money forfeited by whichever earlier seats are NOT in the
    shoved-before subset (those seats necessarily folded, since shove/fold are the
    only two actions)."""
    infosets: list[InfoSet] = []
    n = len(cfg.seats)
    for k, seat in enumerate(cfg.seats):
        earlier_seats = cfg.seats[:k]
        is_last = k == n - 1
        subsets = _all_subsets(earlier_seats)
        if is_last:
            subsets = [s for s in subsets if len(s) > 0]  # exclude the all-folded walk
        for shoved_before in subsets:
            folded_before = [s for s in earlier_seats if s not in shoved_before]
            dead_money_bb = sum(cfg.blind_of(s) for s in folded_before)
            infosets.append(
                InfoSet(
                    seat=seat,
                    seat_pos=k,
                    shoved_before=shoved_before,
                    facing_decision="open" if not shoved_before else "call",
                    dead_money_bb=dead_money_bb,
                    is_last_seat=is_last,
                )
            )
    return infosets


def infosets_by_key(infosets: list[InfoSet]) -> dict:
    return {infoset_key(i.seat, i.shoved_before): i for i in infosets}


def init_ranges(infosets: list[InfoSet], init_weight: float = 1.0):
    """All-shove/all-call init ('all-shove is a fine start' per spec) -- every one of
    the 169 hands starts at `init_weight` at every info set."""
    import numpy as np

    return {
        infoset_key(i.seat, i.shoved_before): np.full(169, init_weight, dtype=np.float64)
        for i in infosets
    }


def next_seat(cfg: GameConfig, infoset: InfoSet) -> str | None:
    """The seat that acts immediately after this info set's seat, or None if this
    info set's seat is last (BB). HU has at most one next seat ever; solver.py's
    current best_response relies on that (see its docstring for the 4-max caveat)."""
    if infoset.is_last_seat:
        return None
    return cfg.seats[infoset.seat_pos + 1]


def infoset_description(cfg, infoset: InfoSet) -> str:
    """Plain-English situation line, e.g. 'BB call vs UTG + SB shoves (BTN folded)'.
    Lives here (pure structure -> text) so dashboard code can describe info sets
    without importing viz/matplotlib; scripts/export_range_viewer.py aliases it."""
    order = {s: i for i, s in enumerate(cfg.seats)}
    shovers = sorted(infoset.shoved_before, key=order.get)
    folded = [s for s in cfg.seats[: infoset.seat_pos] if s not in infoset.shoved_before]
    folded_note = f" ({', '.join(folded)} folded)" if folded else ""
    if not shovers:
        return f"{infoset.seat} open-shove{folded_note or ' (first to act)'}"
    return f"{infoset.seat} call vs {' + '.join(shovers)} shove{'s' if len(shovers) > 1 else ''}{folded_note}"
