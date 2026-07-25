import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import backtest
import equity
import solver
from backtest import GradedSpot, SolveCache
from game import FOURMAX_CONFIG
from hand_history import parse_hand
from pushfold_spot import find_spot
from fourmax_spot import find_fourmax_spot
from game import FOURMAX_CONFIG, HU_CONFIG
from session_analysis import (
    GradedHand,
    GradingContext,
    analyze_window,
    grade_hands,
    graded_in_window,
    grades_cache_path,
    is_match,
    load_fourmax_ranges_if_cached,
    load_or_grade_file,
    make_grading_context,
    realized_spot_ev_net_bb,
)
from test_fourmax_spot import (
    BB_FACES_MULTIWAY_SHOVE,
    BTN_FOLDS_TO_UTG_SHOVE,
    BTN_SHOVES_AFTER_UTG_FOLDS,
    POST_FOLD_LIMP,
)
from test_hand_history import (
    SAMPLE_1_HERO_NOT_BLIND,
    SAMPLE_2_GRADABLE_BB_CALL,
    SAMPLE_3_SHOVE_MULTIWAY,
    SAMPLE_4_ANTE_SKIP,
)
from test_solver import FAST_SOLVER_CFG, SYNTHETIC_TABLE, _FakeEvaluator

ALL_SAMPLES = [
    SAMPLE_1_HERO_NOT_BLIND,
    SAMPLE_2_GRADABLE_BB_CALL,
    SAMPLE_3_SHOVE_MULTIWAY,
    SAMPLE_4_ANTE_SKIP,
]


def hu_ctx() -> GradingContext:
    return GradingContext(solve_cache=SolveCache(SYNTHETIC_TABLE), fourmax=None, evaluator=None)


@pytest.fixture(scope="module")
def fourmax_fast():
    """Fast, structurally valid (NOT converged) 4-max solve -- same fixture idea as
    test_backtest.py's config-agnostic grading test."""
    ranges, _info = solver.solve(FOURMAX_CONFIG, FAST_SOLVER_CFG, table=SYNTHETIC_TABLE, evaluator=_FakeEvaluator())
    return FOURMAX_CONFIG, ranges


def make_row(
    hand_id: str = "h1",
    bb: float = 2.0,
    model: str = "HU",
    position: str = "BB",
    hero_action: str = "fold",
    gto_action: str = "fold",
    weight: float = 0.0,
    ev_loss: float = 0.0,
    predicted: float = 0.0,
    realized: float | None = None,
    net: float | None = None,
    timestamp: str = "2026/07/04 18:01:21 +08",
) -> GradedHand:
    spot = GradedSpot(
        hand_id=hand_id,
        timestamp=timestamp,
        position=position,
        effective_stack_bb=8.0,
        hero_hand="A8o",
        hero_action=hero_action,
        gto_weight=weight,
        gto_recommended_action=gto_action,
        ev_loss_bb=ev_loss,
        predicted_ev_net_bb=predicted,
        model=model,
    )
    return GradedHand(graded=spot, bb_amount=bb, net_currency=net, realized_ev_net_bb=realized)


# ---------------------------------------------------------------------------------
# grade_hands
# ---------------------------------------------------------------------------------


def test_grade_hands_grades_the_gradable_sample_and_skips_the_rest():
    hands = [parse_hand(s) for s in ALL_SAMPLES]
    rows, skips = grade_hands(hands, hu_ctx())

    assert len(rows) == 1
    row = rows[0]
    assert row.graded.hand_id == "10256500251"  # SAMPLE_2: BB call, 8bb
    assert row.graded.model == "HU"
    assert row.graded.position == "BB"
    assert row.graded.effective_stack_bb == pytest.approx(8.0)
    assert row.bb_amount == pytest.approx(2.0)
    # collected 32.56 minus (2 posted + 14 all-in) -- ledger ground truth
    assert row.net_currency == pytest.approx(16.56)
    assert row.realized_ev_net_bb is not None
    assert skips == {"hero_not_blind_poster": 1, "shove_multiway": 1, "ante_present": 1}


def test_graded_hand_realized_ev_matches_backtest_realized_ev():
    hand = parse_hand(SAMPLE_2_GRADABLE_BB_CALL)
    cache = SolveCache(SYNTHETIC_TABLE)
    rows, _skips = grade_hands([hand], GradingContext(cache, None, None))

    spot, reason = find_spot(hand)
    assert reason is None
    solved = cache.get(spot.effective_stack_bb)
    assert rows[0].realized_ev_net_bb == pytest.approx(backtest.realized_ev_net_bb(hand, spot, solved.cfg))


def test_fourmax_retry_grades_multiway_and_non_blind_spots_when_ranges_supplied(fourmax_fast):
    hands = [parse_hand(s) for s in ALL_SAMPLES]
    ctx = GradingContext(SolveCache(SYNTHETIC_TABLE), fourmax=fourmax_fast, evaluator=_FakeEvaluator())
    rows, skips = grade_hands(hands, ctx)

    by_id = {r.graded.hand_id: r for r in rows}
    assert len(rows) == 3
    s1 = by_id["10251900257"]  # hero_not_blind_poster -> 4MAX UTG fold
    assert (s1.graded.model, s1.graded.position, s1.graded.hero_action) == ("4MAX", "UTG", "fold")
    assert s1.realized_ev_net_bb == pytest.approx(0.0)  # no blind, no equity: exactly 0
    s3 = by_id["10255500068"]  # shove_multiway -> 4MAX BTN open-shove, everyone folded
    assert (s3.graded.model, s3.graded.position, s3.graded.hero_action) == ("4MAX", "BTN", "shove_or_call")
    # true net: collected 10.90 minus (40 all-in - 35 returned) = +5.90 at bb=5
    assert s3.realized_ev_net_bb == pytest.approx(5.90 / 5)
    assert by_id["10256500251"].graded.model == "HU"
    assert skips == {"ante_present": 1}


def test_fourmax_spot_at_unsolved_depth_is_skipped_not_misgraded(fourmax_fast):
    # Same multiway hand at 10bb: the pickle is solved at 8bb only, so grading it
    # against those ranges would be wrong -- it must land in a skip count instead.
    deeper = SAMPLE_3_SHOVE_MULTIWAY.replace("₮40", "₮50")
    ctx = GradingContext(SolveCache(SYNTHETIC_TABLE), fourmax=fourmax_fast, evaluator=_FakeEvaluator())
    rows, skips = grade_hands([parse_hand(deeper)], ctx)
    assert rows == []
    assert skips == {"fourmax_depth_not_solved": 1}


# ---------------------------------------------------------------------------------
# realized_spot_ev_net_bb: exact ledger (hero_net_currency) + model-convention
# rakeback for the leaf that actually realized. Numbers below are hand-verified.
# ---------------------------------------------------------------------------------

HU_SB_STEAL = """\
CoinPoker Hand #88000001: NLH (₮1/₮2) 2026/07/04 18:03:00 +08
Table 'test' 4-max Seat #3 is the button
Seat 1: Hero (₮16 in chips)
Seat 2: v1 (₮16 in chips)
Seat 3: v2 (₮16 in chips)
Hero: posts small blind ₮1
v1: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Hero [Kh Tc]
Dealt to v1
Dealt to v2
v2: folds
Hero: ALLIN ₮15
v1: folds
Hero: RETURN ₮14
*** SHOWDOWN ***
Hero collected ₮3.56 from pot
*** SUMMARY ***
Total pot ₮4 | Rake ₮0.24 | Splash Fee ₮0.20
"""

HU_SB_FOLD = """\
CoinPoker Hand #88000002: NLH (₮1/₮2) 2026/07/04 18:03:10 +08
Table 'test' 4-max Seat #3 is the button
Seat 1: Hero (₮16 in chips)
Seat 2: v1 (₮16 in chips)
Seat 3: v2 (₮16 in chips)
Hero: posts small blind ₮1
v1: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Hero [7c 2d]
Dealt to v1
Dealt to v2
v2: folds
Hero: folds
v1: RETURN ₮1
*** SHOWDOWN ***
v1 collected ₮1.56 from pot
*** SUMMARY ***
Total pot ₮2 | Rake ₮0.24 | Splash Fee ₮0.20
"""


def _hu_spot(sample: str):
    hand = parse_hand(sample)
    spot, reason = find_spot(hand)
    assert reason is None, reason
    return hand, spot


def _fourmax_spot(sample: str):
    hand = parse_hand(sample)
    spot, reason = find_fourmax_spot(hand)
    assert reason is None, reason
    return hand, spot


def test_realized_hu_steal_is_true_net_not_collected():
    # True net: collected 3.56 minus (1 posted + 15 all-in - 14 returned) = +1.56
    # currency = +0.78bb, matching ev.steal_ev_net's dead(1.0) - drop(0.22) exactly.
    # The collected figure alone (1.78bb) would overstate by hero's own matched
    # 1bb -- the convention gap this function exists to close.
    hand, spot = _hu_spot(HU_SB_STEAL)
    assert realized_spot_ev_net_bb(hand, spot, HU_CONFIG, "HU") == pytest.approx(0.78)


def test_realized_hu_sb_fold():
    # -0.5bb blind + full rakeback share of the 0.5bb "pot" (0.06) = -0.44,
    # the spec's hand-verified SB-fold number.
    hand, spot = _hu_spot(HU_SB_FOLD)
    assert realized_spot_ev_net_bb(hand, spot, HU_CONFIG, "HU") == pytest.approx(-0.44)


def test_realized_hu_called_showdown_agrees_with_backtest_convention():
    # For a genuine called all-in the ledger and collected conventions coincide,
    # so the new function must reproduce backtest.realized_ev_net_bb exactly.
    hand, spot = _hu_spot(SAMPLE_2_GRADABLE_BB_CALL)
    assert realized_spot_ev_net_bb(hand, spot, HU_CONFIG, "HU") == pytest.approx(
        backtest.realized_ev_net_bb(hand, spot, HU_CONFIG))


def test_realized_fourmax_steal():
    # BTN shove, blinds fold: net = 4 - (15 - 13) = +2 currency = +1bb; uncalled
    # shove contributes 0 to the formed pot, so no rakeback term.
    hand, spot = _fourmax_spot(BTN_SHOVES_AFTER_UTG_FOLDS)
    assert realized_spot_ev_net_bb(hand, spot, FOURMAX_CONFIG, "4MAX") == pytest.approx(1.0)


def test_realized_fourmax_fold_from_no_blind_seat_is_zero():
    hand, spot = _fourmax_spot(BTN_FOLDS_TO_UTG_SHOVE)
    assert realized_spot_ev_net_bb(hand, spot, FOURMAX_CONFIG, "4MAX") == pytest.approx(0.0)


def test_realized_fourmax_multiway_call():
    # Hero (BB) calls a 3-way all-in: net = 40 - (2 + 14) = +24 currency = +12bb;
    # rakeback = pool * stack / (3 stacks + SB's dead 0.5bb) = 0.06 * 8/24.5.
    hand, spot = _fourmax_spot(BB_FACES_MULTIWAY_SHOVE)
    expected = 12.0 + FOURMAX_CONFIG.rakeback_pool_bb * 8.0 / 24.5
    assert realized_spot_ev_net_bb(hand, spot, FOURMAX_CONFIG, "4MAX") == pytest.approx(expected)


def test_realized_fourmax_unmappable_continuation_returns_none():
    # Hero's UTG fold is gradeable, but villains limp/check afterwards -- the
    # realization doesn't map onto the shove-or-fold model, so no realized leg.
    hand, spot = _fourmax_spot(POST_FOLD_LIMP)
    assert spot.hero_action == "fold"
    assert realized_spot_ev_net_bb(hand, spot, FOURMAX_CONFIG, "4MAX") is None


# ---------------------------------------------------------------------------------
# is_match / analyze_window
# ---------------------------------------------------------------------------------


def test_is_match_pure_and_mixed_semantics():
    assert is_match(make_row(hero_action="fold", gto_action="fold", weight=0.0).graded)
    assert not is_match(make_row(hero_action="fold", gto_action="shove_or_call", weight=1.0).graded)
    assert not is_match(make_row(hero_action="shove_or_call", gto_action="fold", weight=0.0).graded)
    # strictly-mixed equilibrium weight: either action is GTO, whatever hero picked
    assert is_match(make_row(hero_action="fold", gto_action="shove_or_call", weight=0.5).graded)
    # the WEIGHT wins over a contradicting (MC-noisy) EV recommendation
    assert is_match(make_row(hero_action="fold", gto_action="shove_or_call", weight=0.0).graded)
    assert is_match(make_row(hero_action="shove_or_call", gto_action="fold", weight=1.0).graded)


def test_analyze_window_accuracy_ev_lost_and_luck_sums():
    rows = [
        make_row("h1", bb=2.0, hero_action="fold", gto_action="fold", weight=0.0,
                 ev_loss=0.0, predicted=-1.0, realized=-1.5, net=-2.0),
        make_row("h2", bb=2.0, hero_action="fold", gto_action="shove_or_call", weight=1.0,
                 ev_loss=0.8, predicted=-1.0, realized=2.0, net=5.0),
        make_row("h3", bb=5.0, model="4MAX", position="BTN", hero_action="shove_or_call",
                 gto_action="shove_or_call", weight=1.0, ev_loss=0.0, predicted=0.4,
                 realized=None, net=None),  # unmappable continuation: no realized leg
        make_row("h4", bb=5.0, model="4MAX", position="BTN", hero_action="shove_or_call",
                 gto_action="shove_or_call", weight=1.0, ev_loss=0.0, predicted=0.4,
                 realized=1.0, net=5.0),  # 4MAX rows with a realized leg join the luck sums
    ]
    wa = analyze_window(rows)

    assert (wa.spots, wa.hu_spots, wa.fourmax_spots, wa.matched) == (4, 2, 2, 3)
    assert wa.ev_lost_currency == pytest.approx(0.8 * 2.0)
    # luck legs sum every row with a realized leg (any model), in currency
    assert wa.predicted_currency == pytest.approx(-4.0 + 0.4 * 5.0)
    assert wa.realized_currency == pytest.approx(1.0 + 1.0 * 5.0)
    assert wa.luck_gap_currency == pytest.approx(8.0)
    assert wa.graded_net_currency == pytest.approx(8.0)  # skips the None-net row

    by_bb = {s.bb_amount: s for s in wa.stakes}
    assert sorted(by_bb) == [2.0, 5.0]
    assert (by_bb[2.0].spots, by_bb[2.0].matched, by_bb[2.0].luck_spots) == (2, 1, 2)
    assert by_bb[2.0].ev_lost_bb == pytest.approx(0.8)
    assert by_bb[2.0].predicted_ev_bb == pytest.approx(-2.0)
    assert by_bb[2.0].realized_ev_bb == pytest.approx(0.5)
    assert (by_bb[5.0].spots, by_bb[5.0].luck_spots) == (2, 1)
    assert by_bb[5.0].predicted_ev_bb == pytest.approx(0.4)
    assert by_bb[5.0].realized_ev_bb == pytest.approx(1.0)


def test_analyze_window_empty_returns_none():
    assert analyze_window([]) is None


def test_top_mistakes_sorted_capped_and_exclude_zero_loss_and_matched():
    rows = [
        make_row("h1", ev_loss=0.0),
        make_row("h2", hero_action="fold", gto_action="shove_or_call", weight=1.0, ev_loss=0.3),
        make_row("h3", hero_action="fold", gto_action="shove_or_call", weight=1.0, ev_loss=0.9),
        make_row("h4", hero_action="fold", gto_action="shove_or_call", weight=1.0, ev_loss=0.6),
        # positive "loss" on an equilibrium-endorsed fold (weight 0.0): 4MAX MC
        # noise, not a deviation -- must not appear in the mistakes list
        make_row("h5", model="4MAX", hero_action="fold", gto_action="shove_or_call", weight=0.0, ev_loss=2.0),
    ]
    wa = analyze_window(rows, top_n=2)
    assert [m.hand_id for m in wa.top_mistakes] == ["h3", "h4"]
    assert [m.ev_loss_bb for m in wa.top_mistakes] == [0.9, 0.6]


def test_analyze_window_dedups_by_hand_id():
    # The same hand appearing in two overlapping exports must count once.
    rows = [make_row("h1", net=5.0), make_row("h1", net=5.0), make_row("h2", net=1.0)]
    wa = analyze_window(rows)
    assert wa.spots == 2
    assert wa.graded_net_currency == pytest.approx(6.0)


def test_graded_in_window_filters_inclusively():
    rows = [
        make_row("h1", timestamp="2026/07/04 18:01:21 +08"),
        make_row("h2", timestamp="2026/07/04 18:03:00 +08"),
        make_row("h3", timestamp="2026/07/04 19:00:00 +08"),
    ]
    # bare session-tracker timestamps against +08-suffixed hand-header ones
    inside = graded_in_window(rows, "2026/07/04 18:01:21", "2026/07/04 18:03:00")
    assert [r.graded.hand_id for r in inside] == ["h1", "h2"]


# ---------------------------------------------------------------------------------
# Persisted per-file grade cache
# ---------------------------------------------------------------------------------


class _CountingCache(SolveCache):
    def __init__(self, table):
        super().__init__(table)
        self.calls = 0

    def get(self, effective_stack_bb: float):
        self.calls += 1
        return super().get(effective_stack_bb)


def test_load_or_grade_file_writes_then_reuses_cache(tmp_path):
    src = tmp_path / "export.txt"
    src.write_text(SAMPLE_2_GRADABLE_BB_CALL, encoding="utf-8")
    hands = [parse_hand(SAMPLE_2_GRADABLE_BB_CALL)]
    cache_dir = str(tmp_path / "cache")

    first_cache = _CountingCache(SYNTHETIC_TABLE)
    rows1 = load_or_grade_file(str(src), hands, GradingContext(first_cache, None, None), cache_dir=cache_dir)
    assert first_cache.calls > 0
    assert grades_cache_path(cache_dir, str(src)).exists()

    second_cache = _CountingCache(SYNTHETIC_TABLE)
    rows2 = load_or_grade_file(str(src), hands, GradingContext(second_cache, None, None), cache_dir=cache_dir)
    assert second_cache.calls == 0  # served from the persisted JSON, no re-solve
    assert rows2 == rows1


def test_cache_invalidated_on_mtime_and_version_change(tmp_path):
    import json
    import os

    src = tmp_path / "export.txt"
    src.write_text(SAMPLE_2_GRADABLE_BB_CALL, encoding="utf-8")
    hands = [parse_hand(SAMPLE_2_GRADABLE_BB_CALL)]
    cache_dir = str(tmp_path / "cache")
    load_or_grade_file(str(src), hands, hu_ctx(), cache_dir=cache_dir)

    # mtime bump -> regrade
    os.utime(src, (src.stat().st_atime, src.stat().st_mtime + 10))
    spy = _CountingCache(SYNTHETIC_TABLE)
    load_or_grade_file(str(src), hands, GradingContext(spy, None, None), cache_dir=cache_dir)
    assert spy.calls > 0

    # version mismatch -> regrade
    grades_path = grades_cache_path(cache_dir, str(src))
    stale = json.loads(grades_path.read_text())
    stale["version"] = "grades-v0"
    grades_path.write_text(json.dumps(stale))
    spy2 = _CountingCache(SYNTHETIC_TABLE)
    load_or_grade_file(str(src), hands, GradingContext(spy2, None, None), cache_dir=cache_dir)
    assert spy2.calls > 0


def test_corrupt_grades_cache_regrades_instead_of_crashing(tmp_path):
    src = tmp_path / "export.txt"
    src.write_text(SAMPLE_2_GRADABLE_BB_CALL, encoding="utf-8")
    hands = [parse_hand(SAMPLE_2_GRADABLE_BB_CALL)]
    cache_dir = str(tmp_path / "cache")

    grades_path = grades_cache_path(cache_dir, str(src))
    grades_path.parent.mkdir(parents=True, exist_ok=True)
    grades_path.write_text("{ not json")
    rows = load_or_grade_file(str(src), hands, hu_ctx(), cache_dir=cache_dir)
    assert len(rows) == 1


# ---------------------------------------------------------------------------------
# Load-only artifact helpers (must never trigger a solve or a table build)
# ---------------------------------------------------------------------------------


def test_load_fourmax_ranges_if_cached_accepts_both_pickle_shapes(tmp_path):
    fake_ranges = {"UTG|": [0.0] * 169}

    per_depth = tmp_path / "per_depth"
    per_depth.mkdir()
    with open(per_depth / "fourmax_ranges_8.0bb.pkl", "wb") as f:
        pickle.dump(fake_ranges, f)
    loaded = load_fourmax_ranges_if_cached(str(per_depth))
    assert loaded is not None and loaded[1] == fake_ranges
    assert loaded[0].stack_bb == pytest.approx(8.0)

    wrapped = tmp_path / "wrapped"
    wrapped.mkdir()
    with open(wrapped / "fourmax_ranges.pkl", "wb") as f:
        pickle.dump({"ranges": fake_ranges, "info": {}, "elapsed_sec": 1.0}, f)
    loaded = load_fourmax_ranges_if_cached(str(wrapped))
    assert loaded is not None and loaded[1] == fake_ranges


def test_load_fourmax_ranges_if_cached_returns_none_when_absent(tmp_path):
    assert load_fourmax_ranges_if_cached(str(tmp_path)) is None


def test_make_grading_context_returns_none_without_equity_cache(tmp_path):
    # An empty cache dir must short-circuit to None -- never fall through into
    # equity.build_equity_table (a full-enumeration build takes a long time).
    assert make_grading_context(str(tmp_path)) is None


REPO_CACHE = Path(__file__).resolve().parent.parent / "cache"


@pytest.mark.skipif(not equity._cache_path(str(REPO_CACHE)).exists(), reason="repo equity cache absent")
def test_make_grading_context_loads_from_repo_cache():
    ctx = make_grading_context(str(REPO_CACHE))
    assert ctx is not None
    assert ctx.solve_cache.table.shape == (169, 169)
    if (REPO_CACHE / "fourmax_ranges.pkl").exists():
        assert ctx.fourmax is not None
