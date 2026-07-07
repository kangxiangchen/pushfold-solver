import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hand_history import parse_hand
from pushfold_spot import PushFoldSpot, find_spot
from test_hand_history import (
    SAMPLE_1_HERO_NOT_BLIND,
    SAMPLE_2_GRADABLE_BB_CALL,
    SAMPLE_3_SHOVE_MULTIWAY,
    SAMPLE_4_ANTE_SKIP,
)


def test_sample_1_skipped_hero_not_blind_poster():
    spot, reason = find_spot(parse_hand(SAMPLE_1_HERO_NOT_BLIND))
    assert spot is None
    assert reason == "hero_not_blind_poster"


def test_sample_2_graded_as_bb_call_spot():
    spot, reason = find_spot(parse_hand(SAMPLE_2_GRADABLE_BB_CALL))
    assert reason is None
    assert spot == PushFoldSpot(
        hand_id="10256500251",
        timestamp="2026/07/04 18:01:21 +08",
        info_set_key="BB|SB",
        hero_hand=("8h", "As"),
        effective_stack_bb=8.0,
        hero_action="shove_or_call",
    )


def test_sample_3_skipped_shove_multiway():
    spot, reason = find_spot(parse_hand(SAMPLE_3_SHOVE_MULTIWAY))
    assert spot is None
    assert reason == "shove_multiway"


def test_sample_4_skipped_ante_present():
    spot, reason = find_spot(parse_hand(SAMPLE_4_ANTE_SKIP))
    assert spot is None
    assert reason == "ante_present"
