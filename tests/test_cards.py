import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards


def test_169_hands_and_1326_combos():
    assert cards.NUM_CANON_HANDS == 169
    assert cards.NUM_COMBOS == 1326
    assert sum(cards.CANON_COMBO_WEIGHT) == 1326
    assert len(set(cards.CANON_LABELS)) == 169  # no duplicate labels


def test_combo_weights_by_type():
    for label, weight in zip(cards.CANON_LABELS, cards.CANON_COMBO_WEIGHT):
        if len(label) == 2:
            assert weight == 6, label
        elif label.endswith("s"):
            assert weight == 4, label
        else:
            assert weight == 12, label


def test_canon_index_roundtrip_all_labels():
    for i, label in enumerate(cards.CANON_LABELS):
        assert cards.canon_index(label) == i
        assert cards.canon_label(i) == label


def test_canon_index_accepts_either_card_order():
    assert cards.canon_index("AKs") == cards.canon_index("KAs")
    assert cards.canon_index("AKo") == cards.canon_index("KAo")
    assert cards.canon_index("AKs") != cards.canon_index("AKo")


def test_canon_index_rejects_invalid():
    for bad in ["AAs", "XY", "AKq", "A"]:
        try:
            cards.canon_index(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for {bad!r}"


def test_combos_for_lengths_match_weights():
    for i in range(cards.NUM_CANON_HANDS):
        combos = cards.combos_for(i)
        assert len(combos) == cards.CANON_COMBO_WEIGHT[i], cards.canon_label(i)
        # every combo is a pair of distinct valid card ints
        for a, b in combos:
            assert 0 <= a < 52 and 0 <= b < 52 and a != b


def test_all_1326_combos_no_duplicates_within_a_hand():
    seen_per_hand: dict[int, set] = {}
    for a, b, idx in cards.all_1326_combos():
        seen_per_hand.setdefault(idx, set()).add(frozenset((a, b)))
    for idx, combo_set in seen_per_hand.items():
        assert len(combo_set) == cards.CANON_COMBO_WEIGHT[idx]
    assert len(cards.all_1326_combos()) == 1326


def test_card_int_str_roundtrip():
    for rank in cards.RANKS:
        for suit in cards.SUITS:
            c = cards.card_int(rank, suit)
            assert cards.card_str(c) == rank.upper() + suit
            assert 0 <= c < 52


def test_deck_minus_removes_exactly_the_given_cards():
    used = (cards.card_int("A", "s"), cards.card_int("K", "s"))
    remaining = cards.deck_minus(*used)
    assert len(remaining) == 50
    assert used[0] not in remaining and used[1] not in remaining


def test_combo_blocked():
    ac = cards.card_int("A", "c")
    ad = cards.card_int("A", "d")
    kh = cards.card_int("K", "h")
    ks = cards.card_int("K", "s")
    assert cards.combo_blocked((ac, kh), (ac, ks))  # shares Ac
    assert not cards.combo_blocked((ac, kh), (ad, ks))  # disjoint


def test_canon_pair_valid_combos_collision_case_AA_vs_AKs():
    aa = cards.canon_index("AA")
    aks = cards.canon_index("AKs")
    valid = cards.canon_pair_valid_combos(aa, aks)
    # Naive cartesian product would be 6*4=24; each AKs combo uses one specific ace,
    # which collides with exactly 3 of AA's 6 combos (the 3 pairs containing that ace),
    # leaving 3 valid AA combos per AKs combo -> 4*3=12 valid pairs, not 24.
    assert len(valid) == 12
    for combo_a, combo_b in valid:
        assert not cards.combo_blocked(combo_a, combo_b)


def test_canon_pair_valid_combos_unrelated_hands_no_collisions_possible():
    # KK and 22 share no ranks, so no combo pair can possibly collide -- full 6*6=36.
    kk = cards.canon_index("KK")
    two2 = cards.canon_index("22")
    valid = cards.canon_pair_valid_combos(kk, two2)
    assert len(valid) == 36


def test_canon_pair_valid_combos_same_hand_excludes_self_overlap():
    aa = cards.canon_index("AA")
    valid = cards.canon_pair_valid_combos(aa, aa)
    # 4 suits split into two disjoint pairs: 3 ways to partition {c,d,h,s} into two
    # unordered pairs, each yielding 2 ordered (combo_a, combo_b) assignments -> 6.
    assert len(valid) == 6
    for combo_a, combo_b in valid:
        assert not cards.combo_blocked(combo_a, combo_b)
