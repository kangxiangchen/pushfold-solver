"""Card representation, 169-hand canonicalization, combo weights, and removal.

Card encoding mirrors eval7's own rank/suit numbering (rank 0=2 .. 12=A, suit
0=c,1=d,2=h,3=s -- confirmed empirically against `eval7.Card`), represented here as a
plain int 0..51 (`rank*4 + suit`) for fast set/removal operations. Conversion to
`eval7.Card` happens only at the equity-evaluation boundary (see equity.py); nothing in
this module imports eval7, so it has no evaluator dependency at all.
"""

import itertools

RANKS = "23456789TJQKA"  # index 0..12
SUITS = "cdhs"  # index 0..3, matches eval7's internal suit numbering


def card_int(rank_char: str, suit_char: str) -> int:
    return RANKS.index(rank_char.upper()) * 4 + SUITS.index(suit_char.lower())


def card_str(card: int) -> str:
    rank, suit = divmod(card, 4)
    return RANKS[rank] + SUITS[suit]


def _build_canon_labels() -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Fixed 169-hand canonical order. Pairs get weight 6 (C(4,2) suit combos),
    suited 4, offsuit 12 -- 13*6 + 78*4 + 78*12 = 78 + 312 + 936 = 1326."""
    labels: list[str] = []
    weights: list[int] = []
    for hi in range(13):
        for lo in range(hi + 1):
            if lo == hi:
                labels.append(RANKS[hi] * 2)
                weights.append(6)
            else:
                labels.append(RANKS[hi] + RANKS[lo] + "s")
                weights.append(4)
                labels.append(RANKS[hi] + RANKS[lo] + "o")
                weights.append(12)
    return tuple(labels), tuple(weights)


CANON_LABELS, CANON_COMBO_WEIGHT = _build_canon_labels()
NUM_CANON_HANDS = len(CANON_LABELS)
NUM_COMBOS = sum(CANON_COMBO_WEIGHT)
assert NUM_CANON_HANDS == 169, NUM_CANON_HANDS
assert NUM_COMBOS == 1326, NUM_COMBOS

_LABEL_TO_INDEX = {label: i for i, label in enumerate(CANON_LABELS)}


def canon_index(hand_str: str) -> int:
    """Parse 'AA', 'AKs', 'AKo' (either card order) into a canonical index 0..168."""
    hand_str = hand_str.strip()
    if len(hand_str) == 2:
        r1, r2 = hand_str[0].upper(), hand_str[1].upper()
        if r1 != r2:
            raise ValueError(f"2-char hand string must be a pair, got {hand_str!r}")
        label = r1 + r2
    elif len(hand_str) == 3:
        r1, r2, suit_char = hand_str[0].upper(), hand_str[1].upper(), hand_str[2].lower()
        if suit_char not in ("s", "o"):
            raise ValueError(f"suffix must be 's' or 'o', got {hand_str!r}")
        if r1 not in RANKS or r2 not in RANKS:
            raise ValueError(f"invalid rank in {hand_str!r}")
        hi, lo = (r1, r2) if RANKS.index(r1) >= RANKS.index(r2) else (r2, r1)
        if hi == lo:
            raise ValueError(f"pairs have no suited/offsuit variant, got {hand_str!r}")
        label = hi + lo + suit_char
    else:
        raise ValueError(f"invalid hand string {hand_str!r}")
    try:
        return _LABEL_TO_INDEX[label]
    except KeyError:
        raise ValueError(f"invalid hand string {hand_str!r} (normalized to {label!r})")


def canon_label(index: int) -> str:
    return CANON_LABELS[index]


def hero_canon_label(card1: str, card2: str) -> str:
    """Two raw dealt cards like 'As', '8h' -> the canonical 'AA'/'AKs'/'AKo' label
    consumed by canon_index. Order-independent; pairs collapse regardless of suits."""
    r1, s1 = card1[0].upper(), card1[1].lower()
    r2, s2 = card2[0].upper(), card2[1].lower()
    if r1 == r2:
        return r1 + r2
    hi, lo = (r1, r2) if RANKS.index(r1) > RANKS.index(r2) else (r2, r1)
    return hi + lo + ("s" if s1 == s2 else "o")


def combos_for(index: int) -> list[tuple[int, int]]:
    """Concrete card-int combos realizing canonical hand `index`. len() == CANON_COMBO_WEIGHT[index]."""
    label = CANON_LABELS[index]
    if len(label) == 2:
        r = RANKS.index(label[0])
        cards = [r * 4 + s for s in range(4)]
        return list(itertools.combinations(cards, 2))
    hi_r, lo_r, suited = RANKS.index(label[0]), RANKS.index(label[1]), label[2] == "s"
    if suited:
        return [(hi_r * 4 + s, lo_r * 4 + s) for s in range(4)]
    return [
        (hi_r * 4 + s_hi, lo_r * 4 + s_lo)
        for s_hi in range(4)
        for s_lo in range(4)
        if s_hi != s_lo
    ]


def all_1326_combos() -> list[tuple[int, int, int]]:
    """(card_a, card_b, canon_index) for all 1326 combos."""
    return [(a, b, idx) for idx in range(NUM_CANON_HANDS) for a, b in combos_for(idx)]


def range_average_weight(range_vec) -> float:
    """Combo-count-weighted average of a length-169 range vector, e.g. the
    probability a random combo from this range has weight 1.0 (all-shove) vs 0.0
    (all-fold) -- used to get P(opponent continues) from a range of per-hand weights."""
    weights = [range_vec[i] * CANON_COMBO_WEIGHT[i] for i in range(NUM_CANON_HANDS)]
    return sum(weights) / NUM_COMBOS


def deck_minus(*used_cards: int) -> list[int]:
    used = set(used_cards)
    return [c for c in range(52) if c not in used]


def combo_blocked(combo_a: tuple[int, int], combo_b: tuple[int, int]) -> bool:
    """True if the two 2-card combos share a card (illegal simultaneous deal)."""
    return bool(set(combo_a) & set(combo_b))


def canon_pair_valid_combos(
    index_a: int, index_b: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """All (combo_a, combo_b) pairs across two canonical hands that don't collide on a
    shared card. Needed because rank-sharing pairs (e.g. AA vs AKs share the ace) have
    materially fewer valid pairings than unrelated hands -- equity must be averaged over
    these valid pairings, not computed from one arbitrarily-chosen representative combo,
    or exactly these collision-heavy pairs get silently mis-weighted."""
    combos_a = combos_for(index_a)
    combos_b = combos_for(index_b)
    return [
        (ca, cb) for ca in combos_a for cb in combos_b if not combo_blocked(ca, cb)
    ]
