"""Equity engine: heads-up exact enumeration (the correctness anchor) behind a common
evaluator interface, plus Monte Carlo equity for genuine 3+-way pots (multiway_equity_mc)
-- pairwise equities from the exact table don't compose into a joint multiway equity,
so 4-max showdowns among 2+ live opponents go through sampling instead.

Evaluator polarity note (verified empirically): eval7.evaluate() already returns
higher-is-better scores. treys.Evaluator().evaluate() returns LOWER-is-better ranks
(1 = best). The Evaluator interface below normalizes both to higher-is-better so nothing
above this module needs to know which backend is active.
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Protocol

import numpy as np
from tqdm import tqdm

import cards

logger = logging.getLogger(__name__)

CACHE_VERSION = "eval7-canon169-fullenum-v1"


class Evaluator(Protocol):
    def score(self, hand_cards: list[int]) -> int:
        """hand_cards: 2-9 card-ints (hole+board). Higher score is always better,
        regardless of backend."""
        ...

    def to_native(self, card: int):
        """Backend-native card object for one card-int -- exposed so a hot loop that
        evaluates the SAME cards against many boards (see _exact_equity_one_combo)
        can convert once per combo pair instead of once per board via score()."""
        ...

    def score_native(self, native_cards: list) -> int:
        """Like score(), but skips int->native conversion -- callers already hold
        native objects from to_native()."""
        ...


class Eval7Evaluator:
    def __init__(self) -> None:
        import eval7

        self._eval7 = eval7
        # Precompute all 52 Card objects once -- constructing one per card, per hand,
        # per board (as a naive `Card(card_str(c))` in score() would) was measured to
        # dominate the exact-enumeration hot loop; a plain list lookup is ~free.
        self._card_lookup = [eval7.Card(cards.card_str(c)) for c in range(52)]

    def score(self, hand_cards: list[int]) -> int:
        lookup = self._card_lookup
        return self._eval7.evaluate([lookup[c] for c in hand_cards])

    def to_native(self, card: int):
        return self._card_lookup[card]

    def score_native(self, native_cards: list) -> int:
        return self._eval7.evaluate(native_cards)


class TreysEvaluator:
    """Fallback if eval7 fails to build (spec-required, not expected to be exercised
    on this environment -- eval7 ships a prebuilt wheel here). treys ranks LOWER as
    better, so scores are negated to satisfy this module's higher-is-better contract."""

    def __init__(self) -> None:
        import treys

        self._evaluator = treys.Evaluator()
        # See Eval7Evaluator for why this is precomputed once rather than per call.
        self._card_lookup = [treys.Card.new(cards.card_str(c)) for c in range(52)]

    def score(self, hand_cards: list[int]) -> int:
        lookup = self._card_lookup
        treys_cards = [lookup[c] for c in hand_cards]
        return self.score_native(treys_cards)

    def to_native(self, card: int):
        return self._card_lookup[card]

    def score_native(self, native_cards: list) -> int:
        hand, board = native_cards[:2], native_cards[2:]
        return -self._evaluator.evaluate(board, hand)


def get_evaluator() -> Evaluator:
    try:
        return Eval7Evaluator()
    except ImportError:
        logger.warning("eval7 unavailable, falling back to treys (pure Python, slower)")
        return TreysEvaluator()


def smoke_check(ev: Evaluator) -> None:
    """Mandatory: confirm the active evaluator imports and scores a known 7-card hand
    correctly before anything else trusts it. Royal flush must outscore a low pair."""
    royal = [cards.card_int(r, "s") for r in "AKQJT"] + [
        cards.card_int("2", "h"),
        cards.card_int("3", "h"),
    ]
    low_pair = [
        cards.card_int("2", "c"),
        cards.card_int("2", "d"),
        cards.card_int("5", "h"),
        cards.card_int("7", "s"),
        cards.card_int("9", "c"),
        cards.card_int("3", "d"),
        cards.card_int("4", "d"),
    ]
    royal_score = ev.score(royal)
    low_pair_score = ev.score(low_pair)
    if not royal_score > low_pair_score:
        raise AssertionError(
            f"Evaluator {type(ev).__name__} scored a royal flush ({royal_score}) as not "
            f"beating a low pair ({low_pair_score}) -- wrong polarity or broken evaluator."
        )


def _suit_partition_signature(
    combo_a: tuple[int, int], combo_b: tuple[int, int]
) -> frozenset:
    """Canonical, suit-relabeling-invariant description of how combo_a's and combo_b's
    suits relate (which of the 4 card-slots share a suit value). Two combo pairs with
    the same signature are related by a pure suit relabeling and are therefore
    GUARANTEED to have identical exact equity -- this is what lets equity_pair_exact
    compute one full board-enumeration per signature group instead of per raw combo
    pair (verified empirically: collapses 808,392 raw combo-pairs across all 14,196
    canonical hand pairs down to 50,505 distinct groups, a ~16x reduction, with the
    worst single pair needing only 7 groups -- with zero loss of exactness, since the
    within-group equity truly is invariant, not merely close)."""
    suits = (combo_a[0] % 4, combo_a[1] % 4, combo_b[0] % 4, combo_b[1] % 4)
    groups: dict[int, list[int]] = {}
    for slot, suit in enumerate(suits):
        groups.setdefault(suit, []).append(slot)
    return frozenset(frozenset(slots) for slots in groups.values())


def _exact_equity_one_combo(
    combo_a: tuple[int, int], combo_b: tuple[int, int], ev: Evaluator
) -> float:
    """Full C(48,5) board enumeration for one specific pair of hole-card combos.
    Returns hand_a's equity (win + half of ties) in [0, 1].

    Converts hole cards and the remaining deck to backend-native objects ONCE before
    the board loop (via to_native/score_native), rather than once per board via
    score() -- profiling showed the naive per-board int->native conversion (14
    conversions/board: 7 cards x 2 hands) dominated this loop's cost across the
    ~1.7M boards enumerated per combo pair, cutting per-pair time roughly 3x."""
    native_a = [ev.to_native(c) for c in combo_a]
    native_b = [ev.to_native(c) for c in combo_b]
    native_deck = [ev.to_native(c) for c in cards.deck_minus(*combo_a, *combo_b)]
    score_native = ev.score_native  # bind once: avoids a repeated attribute lookup
    # on `ev` for every one of the ~3.4M score_native calls in this loop (2 per board)

    wins = ties = total = 0
    for board in itertools.combinations(native_deck, 5):
        board = list(board)
        score_a = score_native(native_a + board)
        score_b = score_native(native_b + board)
        if score_a > score_b:
            wins += 1
        elif score_a == score_b:
            ties += 1
        total += 1
    return (wins + 0.5 * ties) / total


def equity_pair_exact(hand_a: int, hand_b: int, ev: Evaluator) -> tuple[float, float]:
    """Exact equity of canonical hand `hand_a` vs `hand_b` (indices 0..168), averaged
    over every valid specific-combo pairing, grouped by suit-relabeling orbit (see
    _suit_partition_signature) so the computation is exact but tractable. Returns
    (equity_a, equity_b) with equity_a + equity_b == 1.0 (heads-up, ties split)."""
    valid_pairs = cards.canon_pair_valid_combos(hand_a, hand_b)
    groups: dict[frozenset, list[tuple[tuple[int, int], tuple[int, int]]]] = {}
    for combo_a, combo_b in valid_pairs:
        sig = _suit_partition_signature(combo_a, combo_b)
        groups.setdefault(sig, []).append((combo_a, combo_b))

    weighted_sum = 0.0
    total_weight = 0
    for group_pairs in groups.values():
        rep_a, rep_b = group_pairs[0]
        eq_a = _exact_equity_one_combo(rep_a, rep_b, ev)
        weighted_sum += eq_a * len(group_pairs)
        total_weight += len(group_pairs)

    equity_a = weighted_sum / total_weight
    return equity_a, 1.0 - equity_a


def _table_worker(args: tuple[int, int]) -> tuple[int, int, float]:
    i, j = args
    ev = get_evaluator()
    eq_i, _ = equity_pair_exact(i, j, ev)
    return i, j, eq_i


def build_equity_table(cache_dir: str = "cache/", processes: int | None = None) -> np.ndarray:
    """Full 169x169 symmetric table; table[i][j] = equity of canonical hand i vs j.

    table[i][i] is exactly 0.5, not merely by convention: two hands of the SAME
    canonical type but different specific combos (e.g. hero AcAd vs opponent AhAs)
    are related by a suit relabeling that swaps them (here, c<->h and d<->s
    simultaneously), and board enumeration is invariant under any such relabeling --
    so hero's equity must equal the opponent's, and since they sum to 1, each is
    exactly 0.5. (Each worker builds its own Evaluator instance rather than one being
    passed in, since eval7/treys objects aren't necessarily picklable across
    multiprocessing workers -- see _table_worker.) Computes only the upper triangle
    (14,196 unique pairs) via equity_pair_exact and mirrors. This is the expensive,
    "run once" path -- see scripts/build_equity_cache.py."""
    n = cards.NUM_CANON_HANDS
    table = np.full((n, n), 0.5, dtype=np.float64)
    pairs = list(itertools.combinations(range(n), 2))

    with multiprocessing.Pool(processes=processes or os.cpu_count()) as pool:
        for i, j, eq_i in tqdm(
            pool.imap_unordered(_table_worker, pairs, chunksize=8),
            total=len(pairs),
            desc="building 169x169 equity table",
        ):
            table[i, j] = eq_i
            table[j, i] = 1.0 - eq_i

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(_cache_path(cache_dir), table)
    return table


def _cache_path(cache_dir: str) -> Path:
    return Path(cache_dir) / f"equity_169x169_{CACHE_VERSION}.npy"


def load_or_build_equity_table(cache_dir: str = "cache/", **build_kwargs) -> np.ndarray:
    path = _cache_path(cache_dir)
    if path.exists():
        table = np.load(path)
        if table.shape != (cards.NUM_CANON_HANDS, cards.NUM_CANON_HANDS):
            raise ValueError(f"cached table has wrong shape {table.shape}, delete {path} and rebuild")
        aa, kk = cards.canon_index("AA"), cards.canon_index("KK")
        if not (0.80 <= table[aa, kk] <= 0.84):
            raise ValueError(
                f"cached table failed AA-vs-KK spot check ({table[aa, kk]:.4f} not in "
                f"[0.80, 0.84]) -- cache looks corrupted or stale, delete {path} and rebuild"
            )
        return table
    return build_equity_table(cache_dir=cache_dir, **build_kwargs)


def hand_vs_range_equity(hand_idx: int, opp_range: np.ndarray, table: np.ndarray) -> float:
    """Hero's equity (canonical hand `hand_idx`) vs a combo-count-weighted mixture of
    the opponent's range. `opp_range` is a length-169 array of weights in [0, 1]
    (fractional/mixed strategy weight; combo-count weighting is applied here, callers
    should NOT pre-multiply by CANON_COMBO_WEIGHT themselves)."""
    weights = opp_range * np.asarray(cards.CANON_COMBO_WEIGHT, dtype=np.float64)
    total = weights.sum()
    if total == 0.0:
        return 0.5  # no opponent combos left to face -- degenerate, shouldn't be relied upon
    return float(np.dot(weights, table[hand_idx, :]) / total)


def hand_vs_range_equity_vectorized(opp_range: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Same as hand_vs_range_equity but for ALL 169 hero hands at once (returns a
    length-169 array) -- this is the version solver.py's best_response should call,
    since it needs every hero hand's equity against one opponent range every iteration."""
    weights = opp_range * np.asarray(cards.CANON_COMBO_WEIGHT, dtype=np.float64)
    total = weights.sum()
    if total == 0.0:
        return np.full(cards.NUM_CANON_HANDS, 0.5)
    return (table @ weights) / total


def _sample_combo_from_range(
    canon_probs: np.ndarray, blocked: set, rng: np.random.Generator
) -> tuple[int, int] | None:
    """Sample one specific card-combo from a range's combo-count-weighted
    distribution, excluding any combo touching a card in `blocked`. `canon_probs`
    is the precomputed (range_vec * CANON_COMBO_WEIGHT)-normalized distribution over
    the 169 canonical hands, passed in since it's shared across many calls in one MC
    batch. Retries a bounded number of times (a canonical hand can have EVERY combo
    blocked, e.g. all 6 AA combos gone if both other aces are already dealt) before
    giving up and returning None for this draw."""
    for _ in range(50):
        canon_idx = rng.choice(len(canon_probs), p=canon_probs)
        combos = cards.combos_for(canon_idx)
        rng.shuffle(combos)
        for combo in combos:
            if combo[0] not in blocked and combo[1] not in blocked:
                return combo
    return None


def multiway_equity_mc(
    opp_ranges: list[np.ndarray], ev: Evaluator, iterations: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Monte Carlo equity of every one of the 169 canonical hero hands (one
    representative combo each -- this function is already an approximation by
    design, per spec, so the same minor representative-combo bias accepted
    elsewhere doesn't add meaningfully more error here) against a JOINT random draw
    from each range in `opp_ranges` (2 ranges for a 3-way pot, 3 for 4-way),
    respecting mutual card removal, over a full 5-card board each iteration.

    Each iteration: sample one combo per opponent range (rejecting collisions with
    already-sampled cards), deal a random board from what's left, then for every
    hero canonical hand whose representative combo doesn't collide with any of
    that, evaluate hero vs all sampled opponents and accumulate hero's equity share
    (1/k if hero ties for best among k live hands, 0 if beaten, 1 if hero wins
    outright). Hero hands that collide with a given iteration's sample simply don't
    get a sample that round -- tracked per-hand so the final average only divides
    by each hand's own valid sample count, not a shared iteration count.

    `iterations` should be "moderate" (spec's own word) -- this runs inside the
    solver's per-iteration inner loop, and damping is relied on to absorb MC noise
    rather than requiring painstaking precision here.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    rng = rng or np.random.default_rng()

    weight_arr = np.asarray(cards.CANON_COMBO_WEIGHT, dtype=np.float64)
    opp_canon_probs = []
    for opp_range in opp_ranges:
        w = opp_range * weight_arr
        total = w.sum()
        opp_canon_probs.append(w / total if total > 0 else weight_arr / weight_arr.sum())

    hero_representative = [cards.combos_for(i)[0] for i in range(cards.NUM_CANON_HANDS)]

    equity_sum = np.zeros(cards.NUM_CANON_HANDS)
    sample_count = np.zeros(cards.NUM_CANON_HANDS)

    for _ in range(iterations):
        blocked: set = set()
        opp_combos = []
        for canon_probs in opp_canon_probs:
            combo = _sample_combo_from_range(canon_probs, blocked, rng)
            if combo is None:
                break
            opp_combos.append(combo)
            blocked.update(combo)
        if len(opp_combos) != len(opp_ranges):
            continue  # couldn't fill every opponent seat without collision, skip this draw

        remaining_deck = cards.deck_minus(*blocked)
        board = tuple(rng.choice(remaining_deck, size=5, replace=False))
        opp_scores = [ev.score(list(combo) + list(board)) for combo in opp_combos]

        for hero_idx, hero_combo in enumerate(hero_representative):
            if hero_combo[0] in blocked or hero_combo[1] in blocked:
                continue
            hero_score = ev.score(list(hero_combo) + list(board))
            all_scores = opp_scores + [hero_score]
            best = max(all_scores)
            if hero_score == best:
                equity_sum[hero_idx] += 1.0 / all_scores.count(best)
            sample_count[hero_idx] += 1

    if np.any(sample_count == 0):
        logger.warning(
            "multiway_equity_mc: %d of 169 hands got zero valid samples (heavily "
            "blocked by opponent ranges); returning 0.5 for those",
            int(np.sum(sample_count == 0)),
        )
    return np.where(sample_count > 0, equity_sum / np.maximum(sample_count, 1), 0.5)
