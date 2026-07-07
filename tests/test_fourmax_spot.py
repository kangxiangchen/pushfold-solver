import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fourmax_spot import find_fourmax_spot
from hand_history import parse_hand
from pushfold_spot import PushFoldSpot
from test_hand_history import SAMPLE_4_ANTE_SKIP

UTG_FOLDS = """\
CoinPoker Hand #700000001: NLH (₮1/₮2) 2026/01/01 01:00:00 +08
Table 'test' 4-max Seat #2 is the button
Seat 1: Hero (₮16 in chips)
Seat 2: Btnp (₮16 in chips)
Seat 3: Sbp (₮16 in chips)
Seat 4: Bbp (₮16 in chips)
Sbp: posts small blind ₮1
Bbp: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Hero [Ah Kh]
Dealt to Btnp
Dealt to Sbp
Dealt to Bbp
Hero: folds
Btnp: folds
Sbp: folds
Bbp: RETURN ₮1
*** SHOWDOWN ***
Bbp collected ₮3 from pot
*** SUMMARY ***
Total pot ₮3 | Rake ₮0
"""

BTN_SHOVES_AFTER_UTG_FOLDS = """\
CoinPoker Hand #700000002: NLH (₮1/₮2) 2026/01/01 01:00:01 +08
Table 'test' 4-max Seat #2 is the button
Seat 1: Utgp (₮16 in chips)
Seat 2: Hero (₮16 in chips)
Seat 3: Sbp (₮16 in chips)
Seat 4: Bbp (₮16 in chips)
Sbp: posts small blind ₮1
Bbp: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Utgp
Dealt to Hero [Ah Kh]
Dealt to Sbp
Dealt to Bbp
Utgp: folds
Hero: ALLIN ₮15
Sbp: folds
Bbp: folds
Hero: RETURN ₮13
*** SHOWDOWN ***
Hero collected ₮4 from pot
*** SUMMARY ***
Total pot ₮4 | Rake ₮0
"""

BTN_FOLDS_TO_UTG_SHOVE = """\
CoinPoker Hand #700000003: NLH (₮1/₮2) 2026/01/01 01:00:02 +08
Table 'test' 4-max Seat #2 is the button
Seat 1: Utgp (₮16 in chips)
Seat 2: Hero (₮16 in chips)
Seat 3: Sbp (₮16 in chips)
Seat 4: Bbp (₮16 in chips)
Sbp: posts small blind ₮1
Bbp: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Utgp
Dealt to Hero [Ah Kh]
Dealt to Sbp
Dealt to Bbp
Utgp: ALLIN ₮16
Hero: folds
Sbp: folds
Bbp: folds
*** SHOWDOWN ***
Utgp collected ₮4 from pot
*** SUMMARY ***
Total pot ₮4 | Rake ₮0
"""

THREE_HANDED_BB_CALLS_BTN_SHOVE = """\
CoinPoker Hand #700000004: NLH (₮1/₮2) 2026/01/01 01:00:03 +08
Table 'test' 3-max Seat #1 is the button
Seat 1: Btnp (₮16 in chips)
Seat 2: Sbp (₮16 in chips)
Seat 3: Hero (₮16 in chips)
Sbp: posts small blind ₮1
Hero: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Btnp
Dealt to Sbp
Dealt to Hero [Ah Kh]
Btnp: ALLIN ₮16
Sbp: folds
Hero: ALLIN ₮14
*** SHOWDOWN ***
Hero collected ₮30 from pot
*** SUMMARY ***
Total pot ₮32 | Rake ₮0
"""

BB_FACES_MULTIWAY_SHOVE = """\
CoinPoker Hand #700000005: NLH (₮1/₮2) 2026/01/01 01:00:04 +08
Table 'test' 4-max Seat #2 is the button
Seat 1: Utgp (₮16 in chips)
Seat 2: Btnp (₮16 in chips)
Seat 3: Sbp (₮16 in chips)
Seat 4: Hero (₮16 in chips)
Sbp: posts small blind ₮1
Hero: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Utgp
Dealt to Btnp
Dealt to Sbp
Dealt to Hero [Ah Kh]
Utgp: ALLIN ₮16
Btnp: calls ₮16
Sbp: folds
Hero: ALLIN ₮14
*** SHOWDOWN ***
Hero collected ₮40 from pot
*** SUMMARY ***
Total pot ₮46 | Rake ₮0
"""

TOO_MANY_ROLES = """\
CoinPoker Hand #700000006: NLH (₮1/₮2) 2026/01/01 01:00:05 +08
Table 'test' 5-max Seat #1 is the button
Seat 1: P1 (₮16 in chips)
Seat 2: P2 (₮16 in chips)
Seat 3: P3 (₮16 in chips)
Seat 4: Sbp (₮16 in chips)
Seat 5: Hero (₮16 in chips)
Sbp: posts small blind ₮1
Hero: posts big blind ₮2
*** HOLE CARDS ***
Dealt to P1
Dealt to P2
Dealt to P3
Dealt to Sbp
Dealt to Hero [Ah Kh]
P1: folds
P2: folds
P3: folds
Sbp: folds
Hero: RETURN ₮1
*** SHOWDOWN ***
Hero collected ₮3 from pot
*** SUMMARY ***
Total pot ₮3 | Rake ₮0
"""

NON_ALLIN_RAISE = """\
CoinPoker Hand #700000007: NLH (₮1/₮2) 2026/01/01 01:00:06 +08
Table 'test' 4-max Seat #2 is the button
Seat 1: Utgp (₮16 in chips)
Seat 2: Btnp (₮16 in chips)
Seat 3: Sbp (₮16 in chips)
Seat 4: Hero (₮16 in chips)
Sbp: posts small blind ₮1
Hero: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Utgp
Dealt to Btnp
Dealt to Sbp
Dealt to Hero [Ah Kh]
Utgp: raises ₮2 to ₮4
Btnp: folds
Sbp: folds
Hero: folds
*** SHOWDOWN ***
Utgp collected ₮7 from pot
*** SUMMARY ***
Total pot ₮7 | Rake ₮0
"""


def test_utg_open_fold():
    spot, reason = find_fourmax_spot(parse_hand(UTG_FOLDS))
    assert reason is None
    assert spot == PushFoldSpot("700000001", "2026/01/01 01:00:00 +08", "UTG|", ("Ah", "Kh"), 8.0, "fold")


def test_btn_open_shove_after_utg_folds():
    spot, reason = find_fourmax_spot(parse_hand(BTN_SHOVES_AFTER_UTG_FOLDS))
    assert reason is None
    assert spot.info_set_key == "BTN|"
    assert spot.hero_action == "shove_or_call"
    assert spot.effective_stack_bb == 8.0


def test_btn_fold_facing_utg_shove():
    spot, reason = find_fourmax_spot(parse_hand(BTN_FOLDS_TO_UTG_SHOVE))
    assert reason is None
    assert spot.info_set_key == "BTN|UTG"
    assert spot.hero_action == "fold"


def test_three_handed_bb_calls_btn_shove_no_utg_seat():
    spot, reason = find_fourmax_spot(parse_hand(THREE_HANDED_BB_CALLS_BTN_SHOVE))
    assert reason is None
    assert spot.info_set_key == "BB|BTN"
    assert spot.hero_action == "shove_or_call"


def test_bb_faces_multiway_shove():
    spot, reason = find_fourmax_spot(parse_hand(BB_FACES_MULTIWAY_SHOVE))
    assert reason is None
    assert spot.info_set_key == "BB|BTN|UTG"
    assert spot.hero_action == "shove_or_call"


def test_too_many_roles_skipped():
    spot, reason = find_fourmax_spot(parse_hand(TOO_MANY_ROLES))
    assert spot is None
    assert reason == "too_many_roles"


def test_non_allin_raise_skipped():
    spot, reason = find_fourmax_spot(parse_hand(NON_ALLIN_RAISE))
    assert spot is None
    assert reason == "non_allin_action"


def test_ante_hand_skipped():
    spot, reason = find_fourmax_spot(parse_hand(SAMPLE_4_ANTE_SKIP))
    assert spot is None
    assert reason == "ante_present"
