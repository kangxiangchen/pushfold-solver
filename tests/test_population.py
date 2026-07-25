import math
import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ first, then src/ on top: src/ must win name collisions (scripts/backtest.py
# shadows src/backtest.py and would self-import if it resolved first).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
from fourmax_spot import classify_fourmax_realization
from hand_history import parse_hand
from population import (
    INFOSET_ORDER,
    InfosetPoolStats,
    census,
    compare_to_gto,
    shown_composition,
    summarize,
    villain_decisions,
)
from test_fourmax_spot import (
    BB_FACES_MULTIWAY_SHOVE,
    BTN_SHOVES_AFTER_UTG_FOLDS,
    THREE_HANDED_BB_CALLS_BTN_SHOVE,
    UTG_FOLDS,
)
from test_hand_history import SAMPLE_2_GRADABLE_BB_CALL, SAMPLE_4_ANTE_SKIP


def keys_and_actions(decisions):
    return [(d.info_set_key, d.action) for d in decisions]


# ---------------------------------------------------------------------------------
# villain_decisions: the census walk
# ---------------------------------------------------------------------------------


def test_villain_decisions_walk_hand_excludes_hero():
    obs = villain_decisions(parse_hand(UTG_FOLDS))
    # Hero's UTG fold is excluded; BB never acts in the all-fold walk.
    assert keys_and_actions(obs) == [("BTN|", "fold"), ("SB|", "fold")]
    assert all(o.bb_amount == 2.0 and o.effective_stack_bb == 8.0 for o in obs)


def test_villain_decisions_multiway_keys_and_closing_call_is_go():
    obs = villain_decisions(parse_hand(BB_FACES_MULTIWAY_SHOVE))
    # Utgp shoves, Btnp's closing CALL is a "go", Sbp folds facing both;
    # hero's own BB call is excluded.
    assert keys_and_actions(obs) == [
        ("UTG|", "go"),
        ("BTN|UTG", "go"),
        ("SB|BTN|UTG", "fold"),
    ]


def test_hero_shove_still_advances_context():
    obs = villain_decisions(parse_hand(BTN_SHOVES_AFTER_UTG_FOLDS))
    # Hero's BTN shove is not recorded but later villains face BTN in their keys.
    assert keys_and_actions(obs) == [
        ("UTG|", "fold"),
        ("SB|BTN", "fold"),
        ("BB|BTN", "fold"),
    ]


def test_prefix_kept_before_unmappable_action():
    text = BTN_SHOVES_AFTER_UTG_FOLDS.replace(
        "Sbp: folds\nBbp: folds", "Sbp: raises ₮2 to ₮4\nBbp: folds"
    )
    obs = villain_decisions(parse_hand(text))
    # Utgp's clean fold survives; the non-all-in raise ends the walk there.
    assert keys_and_actions(obs) == [("UTG|", "fold")]


def test_skips_ante_deep_and_unclassified_hands():
    assert villain_decisions(parse_hand(SAMPLE_4_ANTE_SKIP)) == []
    deep = UTG_FOLDS.replace("₮16 in chips", "₮50 in chips")  # 25bb > MAX_PUSHFOLD_STACK_BB
    assert villain_decisions(parse_hand(deep)) == []
    unclassified = UTG_FOLDS.replace("Btnp: folds", "Btnp: timebanks ₮1\nBtnp: folds")
    assert villain_decisions(parse_hand(unclassified)) == []


def test_census_dedups_by_hand_id():
    hand = parse_hand(UTG_FOLDS)
    assert len(census([hand, hand])) == len(villain_decisions(hand))


def test_shown_join():
    obs = villain_decisions(parse_hand(SAMPLE_2_GRADABLE_BB_CALL))
    by_key = {o.info_set_key: o for o in obs}
    assert by_key["BTN|"].action == "go"
    assert by_key["BTN|"].shown == ("6h", "Ad")  # 14cee93d's showdown cards
    assert by_key["SB|BTN"].shown is None  # the folder never shows


def test_census_go_actors_match_realization_live_others():
    # Cross-check against the (independently tested) hero-relative full-hand walk:
    # the roles observed shoving in the census must equal realization.live_others.
    for sample in (UTG_FOLDS, BTN_SHOVES_AFTER_UTG_FOLDS, BB_FACES_MULTIWAY_SHOVE,
                   THREE_HANDED_BB_CALLS_BTN_SHOVE, SAMPLE_2_GRADABLE_BB_CALL):
        hand = parse_hand(sample)
        realization = classify_fourmax_realization(hand)
        assert realization is not None, sample[:40]
        go_roles = {d.info_set_key.split("|", 1)[0] for d in villain_decisions(hand) if d.action == "go"}
        assert go_roles == set(realization.live_others), sample[:40]


# ---------------------------------------------------------------------------------
# summarize / compare_to_gto
# ---------------------------------------------------------------------------------


def make_dec(key="UTG|", action="fold", bb=2.0, hand_id="h", shown=None):
    from population import VillainDecision

    return VillainDecision(hand_id=hand_id, bb_amount=bb, effective_stack_bb=8.0,
                           info_set_key=key, actor="v", action=action, shown=shown)


def test_summarize_hand_verifiable_freq_se_and_per_stake():
    decisions = [
        make_dec(action="go", bb=2.0),
        make_dec(action="fold", bb=2.0),
        make_dec(action="fold", bb=2.0),
        make_dec(action="fold", bb=5.0),
    ]
    stats = summarize(decisions)
    assert [s.key for s in stats] == list(INFOSET_ORDER)  # all 14 keys, tree order
    utg = next(s for s in stats if s.key == "UTG|")
    assert (utg.n, utg.go) == (4, 1)
    assert utg.pool_freq == pytest.approx(0.25)
    assert utg.se == pytest.approx(math.sqrt(0.25 * 0.75 / 4))
    assert [(st.bb_amount, st.n, st.go) for st in utg.per_stake] == [(2.0, 3, 1), (5.0, 1, 0)]
    empty = next(s for s in stats if s.key == "BB|BTN|SB|UTG")
    assert (empty.n, empty.pool_freq, empty.se) == (0, None, None)


def test_compare_to_gto_z_and_degrade():
    stats = summarize([make_dec(action="go"), make_dec(), make_dec(), make_dec()])
    flat_ranges = {key: np.full(169, 0.5) for key in INFOSET_ORDER}
    deviations = compare_to_gto(stats, flat_ranges)
    utg = next(d for d in deviations if d.stats.key == "UTG|")
    assert utg.gto_freq == pytest.approx(0.5)
    assert utg.z == pytest.approx((0.25 - 0.5) / math.sqrt(0.25 * 0.75 / 4))

    degraded = compare_to_gto(stats, None)
    assert all(d.gto_freq is None and d.z is None for d in degraded)


def test_shown_composition_in_out_and_top_list():
    aa = cards.canon_index("AA")
    ranges = {key: np.zeros(169) for key in INFOSET_ORDER}
    ranges["UTG|"][aa] = 1.0
    decisions = [
        make_dec(action="go", shown=("As", "Ad")),   # AA -- inside the GTO range
        make_dec(action="go", shown=("Ah", "6d")),   # A6o -- outside
        make_dec(action="go", shown=None),           # unrevealed: not in the sample
        make_dec(action="fold", shown=("Kc", "Kh")), # folds never join composition
    ]
    comps = shown_composition(decisions, ranges)
    assert len(comps) == 1
    c = comps[0]
    assert (c.key, c.shown_go, c.in_gto) == ("UTG|", 2, 1)
    assert c.mean_gto_weight == pytest.approx(0.5)
    assert c.top_out_of_range == (("A6o", 1),)
