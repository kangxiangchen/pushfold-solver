import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

import solver
import tree
from backtest import SolveCache, StackTooSmallError, bucket_depth, grade_spot, grade_spot_against
from game import FOURMAX_CONFIG
from pushfold_spot import PushFoldSpot
from test_solver import FAST_SOLVER_CFG, SYNTHETIC_TABLE, _FakeEvaluator

# Structural checks only -- hand-verifying a full parse -> detect -> solve -> grade
# EV number isn't practical the way test_ev.py's golden fixtures are (those check a
# single settle() call, not a converged 169x169-equity Nash solve).

SPOTS = [
    PushFoldSpot("h1", "t1", "SB|", ("Ah", "Ad"), 8.0, "shove_or_call"),
    PushFoldSpot("h2", "t2", "SB|", ("7c", "2d"), 8.0, "fold"),
    PushFoldSpot("h3", "t3", "BB|SB", ("As", "Kd"), 8.0, "shove_or_call"),
    PushFoldSpot("h4", "t4", "BB|SB", ("7c", "2d"), 8.0, "fold"),
]


def test_bucket_depth_rounds_to_nearest_tenth():
    assert bucket_depth(8.02) == 8.0
    assert bucket_depth(7.96) == 8.0
    assert bucket_depth(8.13) == pytest.approx(8.1)


def test_grade_spot_ev_loss_always_nonnegative():
    cache = SolveCache(SYNTHETIC_TABLE)
    for spot in SPOTS:
        graded = grade_spot(spot, cache)
        assert graded.ev_loss_bb >= -1e-9


def test_grade_spot_recommendation_matches_larger_ev():
    cache = SolveCache(SYNTHETIC_TABLE)
    for spot in SPOTS:
        graded = grade_spot(spot, cache)
        # Whenever hero's actual action already matches the solver's pure best
        # response, the EV loss must be exactly zero (re-derived, not hardcoded).
        if graded.hero_action == graded.gto_recommended_action:
            assert graded.ev_loss_bb == pytest.approx(0.0, abs=1e-9)


def test_stack_too_small_raises():
    cache = SolveCache(SYNTHETIC_TABLE)
    tiny_spot = PushFoldSpot("h5", "t5", "SB|", ("Ah", "Ad"), 0.5, "shove_or_call")
    with pytest.raises(StackTooSmallError):
        grade_spot(tiny_spot, cache)


def test_solve_cache_reuses_solve_for_same_bucket():
    cache = SolveCache(SYNTHETIC_TABLE)
    cache.get(8.02)
    assert len(cache._cache) == 1
    cache.get(7.98)
    assert len(cache._cache) == 1  # both bucket to 8.0
    cache.get(9.0)
    assert len(cache._cache) == 2


def test_grade_spot_against_works_for_fourmax_config():
    # Fast, control-flow-focused (mirrors test_solver.py's own fast fourmax tests):
    # proves grade_spot_against's config-agnostic infoset lookup + evaluator
    # threading works end-to-end across all 4 seats and a genuine 3-way multiway
    # leaf, without waiting on a real (tens-of-minutes-to-hours) 4-max solve.
    fake_evaluator = _FakeEvaluator()
    ranges, _info = solver.solve(FOURMAX_CONFIG, FAST_SOLVER_CFG, table=SYNTHETIC_TABLE, evaluator=fake_evaluator)

    fourmax_spots = [
        PushFoldSpot("h1", "t1", "UTG|", ("Ah", "Ad"), 8.0, "shove_or_call"),
        PushFoldSpot("h2", "t2", "BTN|UTG", ("7c", "2d"), 8.0, "fold"),
        PushFoldSpot("h3", "t3", "BB|BTN", ("As", "Kd"), 8.0, "shove_or_call"),
        PushFoldSpot("h4", "t4", "BB|BTN|SB|UTG", ("7c", "2d"), 8.0, "fold"),  # genuine 3-way multiway leaf
    ]
    for spot in fourmax_spots:
        graded = grade_spot_against(spot, FOURMAX_CONFIG, ranges, SYNTHETIC_TABLE, evaluator=fake_evaluator, model="4MAX")
        assert graded.model == "4MAX"
        assert graded.ev_loss_bb >= -1e-9
        assert graded.position == spot.info_set_key.split("|", 1)[0]
        if graded.hero_action == graded.gto_recommended_action:
            assert graded.ev_loss_bb == pytest.approx(0.0, abs=1e-9)


def test_grade_spot_against_ev_memo_reuses_vectors_across_spots():
    # A genuine 3-way multiway leaf: computing its EV vector needs an evaluator.
    # After the first (memo-populating) call, an identical-infoset spot must grade
    # WITHOUT one -- proof the vector was served from the memo, not recomputed --
    # and hand-dependent outputs must still differ per hero hand.
    fake_evaluator = _FakeEvaluator()
    ranges, _info = solver.solve(FOURMAX_CONFIG, FAST_SOLVER_CFG, table=SYNTHETIC_TABLE, evaluator=fake_evaluator)

    memo: dict = {}
    first = grade_spot_against(
        PushFoldSpot("h1", "t1", "BB|BTN|SB|UTG", ("7c", "2d"), 8.0, "fold"),
        FOURMAX_CONFIG, ranges, SYNTHETIC_TABLE, evaluator=fake_evaluator, model="4MAX", ev_memo=memo)
    assert ("4MAX", FOURMAX_CONFIG.stack_bb, "BB|BTN|SB|UTG") in memo

    again = grade_spot_against(
        PushFoldSpot("h1", "t1", "BB|BTN|SB|UTG", ("7c", "2d"), 8.0, "fold"),
        FOURMAX_CONFIG, ranges, SYNTHETIC_TABLE, evaluator=None, model="4MAX", ev_memo=memo)
    assert again == first  # same memoized vector -> bit-identical grade

    other_hand = grade_spot_against(
        PushFoldSpot("h2", "t2", "BB|BTN|SB|UTG", ("Ah", "Ad"), 8.0, "fold"),
        FOURMAX_CONFIG, ranges, SYNTHETIC_TABLE, evaluator=None, model="4MAX", ev_memo=memo)
    assert other_hand.predicted_ev_net_bb != first.predicted_ev_net_bb or other_hand.gto_weight != first.gto_weight
