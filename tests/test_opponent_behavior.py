import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hand_history import parse_hand
from pushfold_spot import classify_bb_response_when_hero_is_sb, classify_sb_open_when_hero_is_bb
from test_hand_history import (
    SAMPLE_1_HERO_NOT_BLIND,
    SAMPLE_2_GRADABLE_BB_CALL,
    SAMPLE_4_ANTE_SKIP,
)

# Synthetic minimal 2-seat fixtures -- none of the real pasted samples exercise
# the literal-SB-poster-shoves / literal-BB-poster-responds branches directly
# (in the real samples, either a non-blind seat does the shoving, or Hero is
# elsewhere), so these cover that directly.

SB_OPPONENT_SHOVES = """\
CoinPoker Hand #999900001: NLH (₮1/₮2) 2026/01/01 00:00:00 +08
Table 'test' 2-max Seat #1 is the button
Seat 1: Villain (₮16 in chips)
Seat 2: Hero (₮16 in chips)
Villain: posts small blind ₮1
Hero: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Villain
Dealt to Hero [Ah Kh]
Villain: ALLIN ₮15
Hero: folds
*** SHOWDOWN ***
Villain collected ₮4 from pot
*** SUMMARY ***
Total pot ₮4 | Rake ₮0
"""

SB_OPPONENT_FOLDS = """\
CoinPoker Hand #999900002: NLH (₮1/₮2) 2026/01/01 00:00:01 +08
Table 'test' 2-max Seat #1 is the button
Seat 1: Villain (₮16 in chips)
Seat 2: Hero (₮16 in chips)
Villain: posts small blind ₮1
Hero: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Villain
Dealt to Hero [Ah Kh]
Villain: folds
*** SHOWDOWN ***
Hero collected ₮3 from pot
*** SUMMARY ***
Total pot ₮3 | Rake ₮0
"""

BB_OPPONENT_CALLS = """\
CoinPoker Hand #999900003: NLH (₮1/₮2) 2026/01/01 00:00:02 +08
Table 'test' 2-max Seat #1 is the button
Seat 1: Hero (₮16 in chips)
Seat 2: Villain (₮16 in chips)
Hero: posts small blind ₮1
Villain: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Hero [Ah Kh]
Dealt to Villain
Hero: ALLIN ₮15
Villain: calls ₮14
*** SHOWDOWN ***
Hero collected ₮32 from pot
*** SUMMARY ***
Total pot ₮32 | Rake ₮0
"""

BB_OPPONENT_FOLDS = """\
CoinPoker Hand #999900004: NLH (₮1/₮2) 2026/01/01 00:00:03 +08
Table 'test' 2-max Seat #1 is the button
Seat 1: Hero (₮16 in chips)
Seat 2: Villain (₮16 in chips)
Hero: posts small blind ₮1
Villain: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Hero [Ah Kh]
Dealt to Villain
Hero: ALLIN ₮15
Villain: folds
*** SHOWDOWN ***
Hero collected ₮3 from pot
*** SUMMARY ***
Total pot ₮3 | Rake ₮0
"""


def test_classify_sb_open_shove():
    assert classify_sb_open_when_hero_is_bb(parse_hand(SB_OPPONENT_SHOVES)) == "shove"


def test_classify_sb_open_fold():
    assert classify_sb_open_when_hero_is_bb(parse_hand(SB_OPPONENT_FOLDS)) == "fold"


def test_classify_bb_response_call():
    assert classify_bb_response_when_hero_is_sb(parse_hand(BB_OPPONENT_CALLS)) == "call"


def test_classify_bb_response_fold():
    assert classify_bb_response_when_hero_is_sb(parse_hand(BB_OPPONENT_FOLDS)) == "fold"


def test_classify_sb_open_none_when_hero_is_sb_not_bb():
    # Hero is the SB-poster in these two -- not the relevant seat for this check.
    assert classify_sb_open_when_hero_is_bb(parse_hand(BB_OPPONENT_CALLS)) is None
    assert classify_sb_open_when_hero_is_bb(parse_hand(BB_OPPONENT_FOLDS)) is None


def test_classify_bb_response_none_when_hero_is_bb_not_sb():
    assert classify_bb_response_when_hero_is_sb(parse_hand(SB_OPPONENT_SHOVES)) is None
    assert classify_bb_response_when_hero_is_sb(parse_hand(SB_OPPONENT_FOLDS)) is None


def test_classify_none_when_hero_not_a_blind_poster():
    assert classify_sb_open_when_hero_is_bb(parse_hand(SAMPLE_1_HERO_NOT_BLIND)) is None
    assert classify_bb_response_when_hero_is_sb(parse_hand(SAMPLE_1_HERO_NOT_BLIND)) is None


def test_classify_none_when_shove_came_from_non_blind_seat():
    # Sample 2: the real shover is the BTN, not the literal SB-poster -- neither
    # check should misattribute that shove as the SB-poster's own decision.
    assert classify_sb_open_when_hero_is_bb(parse_hand(SAMPLE_2_GRADABLE_BB_CALL)) is None


def test_classify_none_for_ante_hands():
    assert classify_sb_open_when_hero_is_bb(parse_hand(SAMPLE_4_ANTE_SKIP)) is None
    assert classify_bb_response_when_hero_is_sb(parse_hand(SAMPLE_4_ANTE_SKIP)) is None
