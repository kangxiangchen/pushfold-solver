import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hand_history
import sessions
from hand_history import ParsedHand


def make_session(**overrides) -> "sessions.Session":
    """A plain, valid session; tests override just the fields they exercise."""
    fields = dict(
        start_time="2026/07/01 20:00:00",
        end_time="2026/07/01 23:30:00",
        start_balance=500.0,
        end_balance=560.0,
        rakeback_balance_start=10.0,
        rakeback_balance_end=14.0,
    )
    fields.update(overrides)
    return sessions.Session(**fields)


# ---------------------------------------------------------------------------------
# Accounting identities -- the two derived numbers this design exists to separate.
# ---------------------------------------------------------------------------------


def test_table_result_is_plain_balance_delta_when_nothing_else_happened():
    s = make_session(start_balance=500.0, end_balance=560.0)
    assert sessions.table_result(s) == pytest.approx(60.0)


def test_mid_session_rakeback_claim_does_not_inflate_table_result():
    # Claiming t20 mid-session jumps the main balance by 20 without any table win;
    # table_result must strip it out, and rakeback_earned must include it.
    s = make_session(
        start_balance=500.0,
        end_balance=580.0,  # +60 from tables, +20 from the claim
        rakeback_balance_start=25.0,
        rakeback_balance_end=9.0,  # earned 4, claimed 20 -> 25 + 4 - 20 = 9
        rakeback_claimed=20.0,
    )
    assert sessions.table_result(s) == pytest.approx(60.0)
    assert sessions.rakeback_earned(s) == pytest.approx(4.0)
    assert sessions.total_result(s) == pytest.approx(64.0)


def test_deposit_and_withdrawal_do_not_pollute_table_result():
    s = make_session(
        start_balance=500.0,
        end_balance=450.0,  # -60 from tables, +100 deposit, -90 withdrawal
        deposits=100.0,
        withdrawals=90.0,
    )
    assert sessions.table_result(s) == pytest.approx(-60.0)


def test_rakeback_earned_without_claims_is_plain_rakeback_delta():
    s = make_session(rakeback_balance_start=10.0, rakeback_balance_end=14.0)
    assert sessions.rakeback_earned(s) == pytest.approx(4.0)


# ---------------------------------------------------------------------------------
# Effective rakeback rate (rho) -- only computable when the rake-paid counter was
# recorded at both ends and actually moved.
# ---------------------------------------------------------------------------------


def test_effective_rakeback_rate_from_counter():
    s = make_session(
        rakeback_balance_start=0.0,
        rakeback_balance_end=4.0,
        rake_paid_start=100.0,
        rake_paid_end=110.0,
    )
    assert sessions.effective_rakeback_rate(s) == pytest.approx(0.4)


def test_effective_rakeback_rate_none_when_counter_absent():
    assert sessions.effective_rakeback_rate(make_session()) is None


def test_effective_rakeback_rate_none_when_counter_did_not_move():
    s = make_session(rake_paid_start=100.0, rake_paid_end=100.0)
    assert sessions.effective_rakeback_rate(s) is None


# ---------------------------------------------------------------------------------
# HH-derived sessions (v2): balances optional, table result can come from the
# hand-history join instead; rakeback tracking is optional per session.
# ---------------------------------------------------------------------------------


def test_hh_only_session_falls_back_to_hh_table_result():
    s = sessions.Session(
        start_time="2026/07/01 20:00:00",
        end_time="2026/07/01 23:30:00",
        hh_table_result=-12.5,
    )
    assert sessions.table_result(s) == pytest.approx(-12.5)
    assert sessions.rakeback_earned(s) is None
    assert sessions.total_result(s) == pytest.approx(-12.5)
    assert sessions.effective_rakeback_rate(s) is None


def test_balances_take_precedence_over_hh_table_result():
    s = make_session(hh_table_result=-999.0)  # balances say +60; HH number is stale/partial
    assert sessions.table_result(s) == pytest.approx(60.0)


def test_session_requires_at_least_one_table_result_source():
    with pytest.raises(ValueError):
        sessions.Session(
            start_time="2026/07/01 20:00:00",
            end_time="2026/07/01 23:30:00",
            rakeback_balance_start=10.0,
            rakeback_balance_end=14.0,
        )


def test_main_balances_must_be_recorded_together_or_not_at_all():
    with pytest.raises(ValueError):
        sessions.Session(
            start_time="2026/07/01 20:00:00", end_time="2026/07/01 21:00:00",
            start_balance=500.0, hh_table_result=5.0,
        )
    with pytest.raises(ValueError):
        sessions.Session(
            start_time="2026/07/01 20:00:00", end_time="2026/07/01 21:00:00",
            end_balance=500.0, hh_table_result=5.0,
        )


def test_rakeback_balances_must_be_recorded_together_or_not_at_all():
    with pytest.raises(ValueError):
        sessions.Session(
            start_time="2026/07/01 20:00:00", end_time="2026/07/01 21:00:00",
            hh_table_result=5.0, rakeback_balance_start=10.0,
        )


def test_hh_table_result_round_trips_through_csv(tmp_path):
    path = str(tmp_path / "sessions.csv")
    s = sessions.Session(
        start_time="2026/07/01 20:00:00", end_time="2026/07/01 23:30:00",
        hh_table_result=-12.5, hands_played=300,
    )
    sessions.append_session(path, s)
    loaded, failures = sessions.load_sessions(path)
    assert failures == []
    assert loaded == [s]


def test_summarize_counts_sessions_missing_rakeback_and_treats_none_as_zero():
    s1 = make_session()  # table +60, rakeback +4
    s2 = sessions.Session(
        start_time="2026/07/02 20:00:00", end_time="2026/07/02 21:00:00",
        hh_table_result=-10.0,
    )
    summary = sessions.summarize([s1, s2])
    assert summary.total_table_result == pytest.approx(50.0)
    assert summary.total_rakeback_earned == pytest.approx(4.0)
    assert summary.total_result == pytest.approx(54.0)
    assert summary.rakeback_missing == 1


# ---------------------------------------------------------------------------------
# Session auto-detection from hand timestamps (gap clustering).
# ---------------------------------------------------------------------------------


def test_detect_sessions_clusters_by_gap():
    hands = [
        make_hand("2026/07/01 20:00:00", 2.0, 0.0),
        make_hand("2026/07/01 20:05:00", 2.0, 0.0),
        make_hand("2026/07/01 20:10:00", 2.0, 0.0),
        make_hand("2026/07/01 22:00:00", 2.0, 0.0),  # 110min gap -> new session
        make_hand("2026/07/01 22:03:00", 2.0, 0.0),
    ]
    detected = sessions.detect_sessions(hands, gap_minutes=45)
    assert len(detected) == 2
    assert detected[0].start_time == "2026/07/01 20:00:00"
    assert detected[0].end_time == "2026/07/01 20:10:00"
    assert len(detected[0].hands) == 3
    assert detected[1].start_time == "2026/07/01 22:00:00"
    assert len(detected[1].hands) == 2


def test_detect_sessions_gap_exactly_at_threshold_starts_new_session():
    hands = [
        make_hand("2026/07/01 20:00:00", 2.0, 0.0),
        make_hand("2026/07/01 20:45:00", 2.0, 0.0),
    ]
    assert len(sessions.detect_sessions(hands, gap_minutes=45)) == 2
    assert len(sessions.detect_sessions(hands, gap_minutes=46)) == 1


def test_detect_sessions_unsorted_input_and_empty():
    assert sessions.detect_sessions([], gap_minutes=45) == []
    hands = [
        make_hand("2026/07/01 22:00:00", 2.0, 0.0),
        make_hand("2026/07/01 20:00:00", 2.0, 0.0),
    ]
    detected = sessions.detect_sessions(hands, gap_minutes=45)
    assert [d.start_time for d in detected] == ["2026/07/01 20:00:00", "2026/07/01 22:00:00"]


# ---------------------------------------------------------------------------------
# Validation at the boundary.
# ---------------------------------------------------------------------------------


def test_end_before_start_rejected():
    with pytest.raises(ValueError):
        make_session(start_time="2026/07/01 23:00:00", end_time="2026/07/01 20:00:00")


def test_negative_amounts_rejected():
    with pytest.raises(ValueError):
        make_session(rakeback_claimed=-5.0)
    with pytest.raises(ValueError):
        make_session(deposits=-1.0)


def test_rake_paid_counter_must_be_recorded_at_both_ends_or_neither():
    with pytest.raises(ValueError):
        make_session(rake_paid_start=100.0)  # end missing
    with pytest.raises(ValueError):
        make_session(rake_paid_end=100.0)  # start missing


def test_rake_paid_counter_cannot_decrease():
    with pytest.raises(ValueError):
        make_session(rake_paid_start=110.0, rake_paid_end=100.0)


# ---------------------------------------------------------------------------------
# CSV persistence -- append/load round trip, malformed rows reported not swallowed.
# ---------------------------------------------------------------------------------


def test_csv_round_trip_including_optional_fields(tmp_path):
    path = str(tmp_path / "sessions.csv")
    s1 = make_session()
    s2 = make_session(
        start_time="2026/07/02 20:00:00",
        end_time="2026/07/02 22:00:00",
        rake_paid_start=100.0,
        rake_paid_end=112.5,
        hands_played=800,
        stakes_note="1/2 + 2/5",
        notes="ran hot",
    )
    sessions.append_session(path, s1)
    sessions.append_session(path, s2)

    loaded, failures = sessions.load_sessions(path)
    assert failures == []
    assert loaded == [s1, s2]
    assert loaded[0].rake_paid_start is None
    assert loaded[1].hands_played == 800


def test_load_sessions_reports_malformed_row_and_keeps_good_ones(tmp_path):
    path = tmp_path / "sessions.csv"
    sessions.append_session(str(path), make_session())
    with open(path, "a", encoding="utf-8") as f:
        f.write("not,a,valid,row\n")
    sessions.append_session(str(path), make_session(start_time="2026/07/03 20:00:00",
                                                    end_time="2026/07/03 21:00:00"))

    loaded, failures = sessions.load_sessions(str(path))
    assert len(loaded) == 2
    assert len(failures) == 1


def test_load_sessions_missing_file_returns_empty():
    loaded, failures = sessions.load_sessions("/nonexistent/sessions.csv")
    assert loaded == []
    assert failures == []


# ---------------------------------------------------------------------------------
# Open-session staging (start/end flow) -- the CSV never holds half-filled rows.
# ---------------------------------------------------------------------------------


def test_open_session_stage_load_clear_round_trip(tmp_path):
    stub_path = str(tmp_path / "session_open.json")
    stub = {"start_time": "2026/07/01 20:00:00", "start_balance": 500.0,
            "rakeback_balance_start": 10.0, "rake_paid_start": None, "stakes_note": ""}
    sessions.stage_open_session(stub_path, stub)
    assert sessions.load_open_session(stub_path) == stub
    sessions.clear_open_session(stub_path)
    assert sessions.load_open_session(stub_path) is None


def test_staging_over_an_existing_open_session_is_refused(tmp_path):
    stub_path = str(tmp_path / "session_open.json")
    stub = {"start_time": "2026/07/01 20:00:00", "start_balance": 500.0,
            "rakeback_balance_start": 10.0, "rake_paid_start": None, "stakes_note": ""}
    sessions.stage_open_session(stub_path, stub)
    with pytest.raises(sessions.OpenSessionExistsError):
        sessions.stage_open_session(stub_path, stub)


# ---------------------------------------------------------------------------------
# Hand-history join: window filter, per-stake grouping, hero net reconstruction.
# ---------------------------------------------------------------------------------


def make_hand(ts: str, bb: float, hero_collected: float, *, hero_is: str = "BB",
              sb_post: float | None = None, extra_actions=(), returns=()) -> ParsedHand:
    """Synthetic preflop-terminal hand: Hero posted a blind per hero_is ('SB'/'BB'),
    optionally acted further, and collected hero_collected."""
    sb_amt = sb_post if sb_post is not None else bb / 2
    actions = [hand_history.PreflopAction(name, act, amt) for name, act, amt in extra_actions]
    actions += [hand_history.PreflopAction("Hero", "return", amt) for amt in returns]
    return ParsedHand(
        hand_id="1", timestamp=f"{ts} +08", table_name="t", max_seats=4, button_seat=1,
        sb_amount=sb_amt, bb_amount=bb, ante_amount=None,
        seats=[hand_history.Seat(1, "Hero", 8 * bb), hand_history.Seat(2, "villain", 8 * bb)],
        sb_poster="Hero" if hero_is == "SB" else "villain",
        bb_poster="Hero" if hero_is == "BB" else "villain",
        dealt_names=["Hero", "villain"], hero_name="Hero", hero_cards=("Ah", "Kd"),
        preflop_actions=actions, collected_by={"Hero": hero_collected} if hero_collected else {},
        total_pot=None, rake=None, splash_fee=0.0, raw_text="",
    )


def test_hands_in_window_inclusive_bounds():
    hands = [
        make_hand("2026/07/01 19:59:59", 2.0, 0.0),
        make_hand("2026/07/01 20:00:00", 2.0, 0.0),
        make_hand("2026/07/01 23:30:00", 2.0, 0.0),
        make_hand("2026/07/01 23:30:01", 2.0, 0.0),
    ]
    inside = sessions.hands_in_window(hands, "2026/07/01 20:00:00", "2026/07/01 23:30:00")
    assert len(inside) == 2


def test_hero_net_walk_matches_real_fixture():
    # Real walk shape: Hero posted BB t2, RETURN t1 (unmatched half), collected t1.56
    # -> net = 1.56 + 1 - 2 = +0.56 (verified against the actual hand in data/).
    hand = make_hand("2026/07/01 20:01:00", 2.0, 1.56, hero_is="BB", returns=(1.0,))
    assert sessions.hero_net_currency(hand) == pytest.approx(0.56)


def test_hero_net_steal_matches_real_fixture():
    # Real steal shape: Hero posted SB t1, ALLIN t15, RETURN t14, collected t3.56
    # -> total in = 1 + 15 - 14 = 2; net = 3.56 - 2 = +1.56.
    hand = make_hand(
        "2026/07/01 20:02:00", 2.0, 3.56, hero_is="SB",
        extra_actions=[("Hero", "allin", 15.0)], returns=(14.0,),
    )
    assert sessions.hero_net_currency(hand) == pytest.approx(1.56)


def test_hero_net_lost_showdown_is_negative_stack():
    # Hero posts BB t2, calls a shove for t14 more, loses: collected 0, in = 16.
    hand = make_hand(
        "2026/07/01 20:03:00", 2.0, 0.0, hero_is="BB",
        extra_actions=[("villain", "allin", 15.0), ("Hero", "calls", 14.0)],
    )
    assert sessions.hero_net_currency(hand) == pytest.approx(-16.0)


def test_hero_net_exact_across_postflop_streets():
    # Hero posts BB 2, checks preflop; bets 4 on the flop (called); raises to 12 on
    # the turn ("raises X to Y" is street-absolute, so the increment is 12), villain
    # folds, 8 returned uncalled. Total in = 2 + 4 + 12 - 8 = 10; collected 15.56.
    hand = make_hand("2026/07/01 20:04:00", 2.0, 15.56, hero_is="BB")
    postflop = [
        [hand_history.PreflopAction("Hero", "bets", 4.0),
         hand_history.PreflopAction("villain", "calls", 4.0)],
        [hand_history.PreflopAction("villain", "bets", 4.0),
         hand_history.PreflopAction("Hero", "raises", 12.0),
         hand_history.PreflopAction("villain", "folds", None),
         hand_history.PreflopAction("Hero", "return", 8.0)],
    ]
    hand = hand_history.ParsedHand(**{**hand.__dict__, "postflop_streets": postflop})
    assert sessions.hero_net_currency(hand) == pytest.approx(15.56 - 10.0)


def test_hero_net_none_only_when_hero_had_unclassified_actions():
    hand = make_hand("2026/07/01 20:04:00", 2.0, 1.56, hero_is="BB", returns=(1.0,))
    hero_junk = hand_history.ParsedHand(**{**hand.__dict__, "unclassified_action_names": ["Hero"]})
    villain_junk = hand_history.ParsedHand(**{**hand.__dict__, "unclassified_action_names": ["villain"]})
    assert sessions.hero_net_currency(hero_junk) is None
    # villain-only oddities can't move hero's own money -- net stays computable
    assert sessions.hero_net_currency(villain_junk) == pytest.approx(0.56)


def test_per_stake_stats_groups_overlapping_multitabled_hands():
    hands = [
        make_hand("2026/07/01 20:01:00", 2.0, 1.56, hero_is="BB", returns=(1.0,)),   # +0.56 at 1/2
        make_hand("2026/07/01 20:01:05", 5.0, 0.0, hero_is="SB"),                    # folded SB: -2.5 at 2/5
        make_hand("2026/07/01 20:01:10", 2.0, 0.0, hero_is="SB"),                    # folded SB: -1 at 1/2
    ]
    stats = sessions.per_stake_stats(hands)
    assert set(stats) == {2.0, 5.0}
    assert stats[2.0].hands == 2
    assert stats[2.0].net_currency == pytest.approx(0.56 - 1.0)
    assert stats[5.0].net_currency == pytest.approx(-2.5)
    assert stats[5.0].net_bb == pytest.approx(-0.5)
    assert stats[5.0].bb_per_100 == pytest.approx(-50.0)


# ---------------------------------------------------------------------------------
# Summary across sessions.
# ---------------------------------------------------------------------------------


def test_summarize_totals_and_rho():
    s1 = make_session()  # table +60, rakeback +4, no counter
    s2 = make_session(
        start_time="2026/07/02 20:00:00", end_time="2026/07/02 22:00:00",
        start_balance=560.0, end_balance=540.0,  # table -20
        rakeback_balance_start=14.0, rakeback_balance_end=18.0,  # +4
        rake_paid_start=100.0, rake_paid_end=110.0,  # rho = 0.4
    )
    summary = sessions.summarize([s1, s2])
    assert summary.n_sessions == 2
    assert summary.total_table_result == pytest.approx(40.0)
    assert summary.total_rakeback_earned == pytest.approx(8.0)
    assert summary.total_result == pytest.approx(48.0)
    assert summary.rho_sessions == 1
    assert summary.rho_mean == pytest.approx(0.4)


def test_summarize_empty():
    summary = sessions.summarize([])
    assert summary.n_sessions == 0
    assert summary.rho_mean is None


# ---------------------------------------------------------------------------------
# parse_timestamp promotion into hand_history (shared by sessions + scripts).
# ---------------------------------------------------------------------------------


def test_parse_timestamp_handles_hand_history_tz_and_bare_session_times():
    with_tz = hand_history.parse_timestamp("2026/06/19 13:42:37 +08")
    bare = hand_history.parse_timestamp("2026/06/19 13:42:37")
    assert with_tz == bare
    assert with_tz.year == 2026 and with_tz.hour == 13
