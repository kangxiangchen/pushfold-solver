import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hand_history import PreflopAction, parse_file, parse_hand, split_hands

# Real sample hands (anonymized villain IDs), used verbatim across
# test_hand_history.py / test_pushfold_spot.py / test_backtest.py.

SAMPLE_1_HERO_NOT_BLIND = """\
CoinPoker Hand #10251900257: NLH (₮2/₮5) 2026/07/04 18:01:15 +08
Table '100170' 4-max Seat #3 is the button
Seat 2: Hero (₮40 in chips)
Seat 1: 839f27b3 (₮40 in chips)
Seat 3: 942ad6c5 (₮40 in chips)
Seat 4: bcfaf6bb (₮40.90 in chips)
bcfaf6bb: posts small blind ₮2
839f27b3: posts big blind ₮5
*** HOLE CARDS ***
Dealt to Hero [Jh 5h]
Dealt to 839f27b3
Dealt to 942ad6c5
Dealt to bcfaf6bb
Hero: folds
942ad6c5: folds
bcfaf6bb: ALLIN ₮38.90
839f27b3: folds
bcfaf6bb: RETURN ₮35.90
*** SHOWDOWN ***
bcfaf6bb collected ₮8.90 from pot
*** SUMMARY ***
Total pot ₮10 | Rake ₮0.60 | Splash Fee ₮0.50
Hand was run once
Board [  ]
Game ended: 2026/07/04 18:01:24 +08
Seat 2: Hero folded before Flop (didn't bet)
Seat 1: 839f27b3 folded before Flop (didn't bet)
Seat 3: 942ad6c5 folded before Flop (didn't bet)
Seat 4: bcfaf6bb won (₮8.90)
"""

SAMPLE_2_GRADABLE_BB_CALL = """\
CoinPoker Hand #10256500251: NLH (₮1/₮2) 2026/07/04 18:01:21 +08
Table '100169' 4-max Seat #1 is the button
Seat 4: Hero (₮16 in chips)
Seat 1: 14cee93d (₮16 in chips)
Seat 2: 8c6b9c2c (₮16 in chips)
8c6b9c2c: posts small blind ₮1
Hero: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Hero [8h As]
Dealt to 14cee93d
Dealt to 8c6b9c2c
14cee93d: ALLIN ₮16
8c6b9c2c: folds
Hero: ALLIN ₮14
*** FLOP *** [4c 9s Kh]
*** TURN *** [4c 9s Kh] [Ac]
*** RIVER *** [4c 9s Kh Ac] [3d]
*** SHOWDOWN ***
Hero: shows [8h As] (One Pair)
Hero collected ₮32.56 from pot
14cee93d: shows [6h Ad] (One Pair)
*** SUMMARY ***
Total pot ₮33 | Rake ₮0.24 | Splash Fee ₮0.20
Hand was run once
Board [ 4c 9s Kh Ac 3d ]
Game ended: 2026/07/04 18:01:43 +08
Seat 4: Hero showed [8h As] and won (₮32.56)
Seat 1: 14cee93d showed [6h Ad] and lost with One Pair
Seat 2: 8c6b9c2c folded before Flop (didn't bet)
"""

SAMPLE_3_SHOVE_MULTIWAY = """\
CoinPoker Hand #10255500068: NLH (₮2/₮5) 2026/07/04 18:01:25 +08
Table '100170' 4-max Seat #2 is the button
Seat 2: Hero (₮40 in chips)
Seat 1: 13c132e6 (₮40 in chips)
Seat 3: 7c40df87 (₮40 in chips)
Seat 4: cc0a16c7 (₮40 in chips)
7c40df87: posts small blind ₮2
cc0a16c7: posts big blind ₮5
*** HOLE CARDS ***
Dealt to Hero [As 8d]
Dealt to 13c132e6
Dealt to 7c40df87
Dealt to cc0a16c7
13c132e6: folds
Hero: ALLIN ₮40
7c40df87: folds
cc0a16c7: folds
Hero: RETURN ₮35
*** SHOWDOWN ***
Hero collected ₮10.90 from pot
*** SUMMARY ***
Total pot ₮12 | Rake ₮0.60 | Splash Fee ₮0.50
Hand was run once
Board [  ]
Game ended: 2026/07/04 18:02:36 +08
Seat 2: Hero showed [As 8d] and won (₮10.90)
Seat 1: 13c132e6 folded before Flop (didn't bet)
Seat 3: 7c40df87 folded before Flop (didn't bet)
Seat 4: cc0a16c7 folded before Flop (didn't bet)
"""

SAMPLE_4_ANTE_SKIP = """\
CoinPoker Hand #1269280214: NLH (₮0.10/₮0.25/₮0.04) 2026/03/20 11:59:35 +08
Table '200624' 6-max Seat #1 is the button
Seat 1: bf54342b (₮26.17 in chips)
Seat 2: 19130889 (₮32.78 in chips)
Seat 3: Hero (₮25 in chips)
Seat 5: 827eafed (₮10 in chips)
Seat 6: e6a1a058 (₮52.94 in chips)
bf54342b: posts ante ₮0.04
19130889: posts ante ₮0.04
Hero: posts ante ₮0.04
827eafed: posts ante ₮0.04
e6a1a058: posts ante ₮0.04
19130889: posts small blind ₮0.10
Hero: posts big blind ₮0.25
*** HOLE CARDS ***
Dealt to bf54342b
Dealt to 19130889
Dealt to Hero [6c Qh]
Dealt to 827eafed
Dealt to e6a1a058
827eafed: folds
e6a1a058: folds
bf54342b: raises ₮0.37 to ₮0.62
19130889: raises ₮2.35 to ₮2.97
Hero: folds
bf54342b: raises ₮4.45 to ₮7.42
19130889: folds
bf54342b: RETURN ₮4.45
*** SHOWDOWN ***
bf54342b collected ₮6.39 from pot
*** SUMMARY ***
Total pot ₮6.39 | Rake ₮0
Hand was run once
Board [  ]
Game ended: 2026/03/20 11:59:55 +08
Seat 1: bf54342b won (₮6.39)
Seat 2: 19130889 folded before Flop (didn't bet)
Seat 3: Hero folded before Flop (didn't bet)
Seat 5: 827eafed folded before Flop (didn't bet)
Seat 6: e6a1a058 folded before Flop (didn't bet)
"""


def test_split_hands_finds_all_four_samples():
    combined = "\n".join([SAMPLE_1_HERO_NOT_BLIND, SAMPLE_2_GRADABLE_BB_CALL, SAMPLE_3_SHOVE_MULTIWAY])
    hands = split_hands(combined)
    assert len(hands) == 3


def test_parse_sample_1_seats_and_button():
    hand = parse_hand(SAMPLE_1_HERO_NOT_BLIND)
    assert hand.hand_id == "10251900257"
    assert hand.button_seat == 3
    assert hand.max_seats == 4
    assert hand.sb_amount == 2
    assert hand.bb_amount == 5
    assert hand.ante_amount is None
    assert hand.ante_posters == []
    assert hand.sb_poster == "bcfaf6bb"
    assert hand.bb_poster == "839f27b3"
    assert [s.name for s in hand.seats] == ["Hero", "839f27b3", "942ad6c5", "bcfaf6bb"]
    assert hand.hero_name == "Hero"
    assert hand.hero_cards == ("Jh", "5h")


def test_parse_sample_2_blinds_dealt_and_actions():
    hand = parse_hand(SAMPLE_2_GRADABLE_BB_CALL)
    assert hand.sb_poster == "8c6b9c2c"
    assert hand.bb_poster == "Hero"
    assert hand.hero_cards == ("8h", "As")
    assert hand.dealt_names == ["Hero", "14cee93d", "8c6b9c2c"]
    assert hand.preflop_actions == [
        PreflopAction("14cee93d", "allin", 16.0),
        PreflopAction("8c6b9c2c", "folds", None),
        PreflopAction("Hero", "allin", 14.0),
    ]


def test_parse_sample_3_button_and_blinds():
    hand = parse_hand(SAMPLE_3_SHOVE_MULTIWAY)
    assert hand.button_seat == 2
    assert hand.sb_poster == "7c40df87"
    assert hand.bb_poster == "cc0a16c7"


def test_parse_sample_4_ante_header_and_posters():
    hand = parse_hand(SAMPLE_4_ANTE_SKIP)
    assert hand.ante_amount == 0.04
    assert hand.ante_posters == ["bf54342b", "19130889", "Hero", "827eafed", "e6a1a058"]
    assert hand.bb_poster == "Hero"


def test_parse_sample_1_outcome_fields():
    hand = parse_hand(SAMPLE_1_HERO_NOT_BLIND)
    assert hand.collected_by == {"bcfaf6bb": 8.90}
    assert hand.total_pot == 10
    assert hand.rake == 0.60
    assert hand.splash_fee == 0.50


def test_parse_sample_2_outcome_fields():
    hand = parse_hand(SAMPLE_2_GRADABLE_BB_CALL)
    assert hand.collected_by == {"Hero": 32.56}
    assert hand.total_pot == 33
    assert hand.rake == 0.24
    assert hand.splash_fee == 0.20


def test_parse_sample_4_outcome_fields_no_splash_fee():
    hand = parse_hand(SAMPLE_4_ANTE_SKIP)
    assert hand.collected_by == {"bf54342b": 6.39}
    assert hand.total_pot == 6.39
    assert hand.rake == 0
    assert hand.splash_fee == 0.0


def test_parse_amount_handles_thousands_separator_commas():
    from hand_history import parse_amount

    assert parse_amount("₮1,403.08") == 1403.08
    assert parse_amount("₮40") == 40.0


def test_parse_file_never_aborts_on_one_bad_hand(tmp_path):
    # Matches the hand-header shape (so split_hands treats it as its own hand) but
    # has an unparseable stakes token, forcing parse_hand to raise.
    bad = "CoinPoker Hand #999: NLH (X) 2026/07/04 18:01:15 +08\n"
    text = SAMPLE_2_GRADABLE_BB_CALL + "\n" + bad
    path = tmp_path / "hh.txt"
    path.write_text(text)
    hands, failures = parse_file(str(path))
    assert len(hands) == 1
    assert len(failures) == 1
    assert failures[0][0] == "999"
