import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from game import HU_CONFIG, FOURMAX_CONFIG
import tree


def test_hu_tree_has_2_infosets():
    infosets = tree.build_infosets(HU_CONFIG)
    assert len(infosets) == 2


def test_fourmax_tree_has_14_infosets():
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    assert len(infosets) == 14


def test_fourmax_per_seat_counts_match_spec():
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    by_seat = {}
    for i in infosets:
        by_seat.setdefault(i.seat, []).append(i)
    assert len(by_seat["UTG"]) == 1
    assert len(by_seat["BTN"]) == 2
    assert len(by_seat["SB"]) == 4
    assert len(by_seat["BB"]) == 7


def test_hu_sb_open_infoset():
    infosets = tree.build_infosets(HU_CONFIG)
    sb = [i for i in infosets if i.seat == "SB"]
    assert len(sb) == 1
    assert sb[0].shoved_before == frozenset()
    assert sb[0].facing_decision == "open"
    assert sb[0].dead_money_bb == 0.0
    assert sb[0].is_last_seat is False


def test_hu_bb_call_infoset():
    infosets = tree.build_infosets(HU_CONFIG)
    bb = [i for i in infosets if i.seat == "BB"]
    assert len(bb) == 1
    assert bb[0].shoved_before == frozenset({"SB"})
    assert bb[0].facing_decision == "call"
    assert bb[0].dead_money_bb == 0.0  # SB is live (shoved), not dead money
    assert bb[0].is_last_seat is True


def test_fourmax_bb_facing_only_utg_has_dead_money_from_folded_btn_and_sb():
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    node = next(i for i in infosets if i.seat == "BB" and i.shoved_before == frozenset({"UTG"}))
    # BTN folded (blind 0) and SB folded (blind 0.5) before BB acts -> dead money 0.5
    assert node.dead_money_bb == 0.5


def test_fourmax_bb_facing_utg_and_sb_no_dead_money_from_them():
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    node = next(
        i for i in infosets if i.seat == "BB" and i.shoved_before == frozenset({"UTG", "SB"})
    )
    # only BTN folded (blind 0) -> dead money 0; SB is live, not dead money
    assert node.dead_money_bb == 0.0


def test_fourmax_bb_facing_all_three_zero_dead_money():
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    node = next(
        i
        for i in infosets
        if i.seat == "BB" and i.shoved_before == frozenset({"UTG", "BTN", "SB"})
    )
    assert node.dead_money_bb == 0.0


def test_fourmax_sb_facing_utg_folded_effectively_has_zero_dead_money_since_utg_posts_none():
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    # SB's open (nobody shoved among UTG, BTN): both folded, but both post 0 blind.
    node = next(i for i in infosets if i.seat == "SB" and i.shoved_before == frozenset())
    assert node.dead_money_bb == 0.0
    assert node.facing_decision == "open"


def test_no_walk_infoset_for_any_seat_all_folded_case_excluded_only_for_last_seat():
    hu_infosets = tree.build_infosets(HU_CONFIG)
    # BB's only infoset must be non-empty shoved_before (walk excluded)
    assert all(i.shoved_before for i in hu_infosets if i.seat == "BB")
    fourmax_infosets = tree.build_infosets(FOURMAX_CONFIG)
    assert all(i.shoved_before for i in fourmax_infosets if i.seat == "BB")
    # but non-last seats DO include the empty/open case
    assert any(not i.shoved_before for i in fourmax_infosets if i.seat == "SB")


def test_infoset_keys_are_unique():
    for cfg in (HU_CONFIG, FOURMAX_CONFIG):
        infosets = tree.build_infosets(cfg)
        keys = [tree.infoset_key(i.seat, i.shoved_before) for i in infosets]
        assert len(keys) == len(set(keys)), f"duplicate keys for {cfg.seats}"


def test_infosets_by_key_roundtrip():
    infosets = tree.build_infosets(HU_CONFIG)
    by_key = tree.infosets_by_key(infosets)
    assert set(by_key.keys()) == {"SB|", "BB|SB"}


def test_init_ranges_all_shove_start():
    infosets = tree.build_infosets(HU_CONFIG)
    ranges = tree.init_ranges(infosets)
    assert set(ranges.keys()) == {"SB|", "BB|SB"}
    for arr in ranges.values():
        assert len(arr) == 169
        assert (arr == 1.0).all()


def test_next_seat():
    infosets = tree.build_infosets(HU_CONFIG)
    sb = next(i for i in infosets if i.seat == "SB")
    bb = next(i for i in infosets if i.seat == "BB")
    assert tree.next_seat(HU_CONFIG, sb) == "BB"
    assert tree.next_seat(HU_CONFIG, bb) is None
