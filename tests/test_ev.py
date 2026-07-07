import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
import equity
import ev
from game import GameConfig, HU_CONFIG, FOURMAX_CONFIG
import tree


CFG = GameConfig(seats=("SB", "BB"))  # same numbers as HU_CONFIG; spec-default rake/splash/rakeback


def test_fixture_a_showdown_ledger_net_to_the_cent():
    # Villain2 (BTN) and Villain3 (UTG) are the two all-in players (k=2); SB and BB both
    # folded their blinds (D = 0.5 + 1.0 = 1.5). Villain2 wins outright (equity 1.0).
    contributions = {"Villain2": 8.0, "Villain3": 8.0, "SB": 0.5, "BB": 1.0}
    equities = {"Villain2": 1.0, "Villain3": 0.0}
    result = ev.settle(CFG, contributions, equities)

    assert result["Villain2"]["ledger_net"] == pytest.approx(9.28, abs=1e-9)
    assert result["Villain3"]["ledger_net"] == pytest.approx(-8.0, abs=1e-9)
    assert result["SB"]["ledger_net"] == pytest.approx(-0.5, abs=1e-9)
    assert result["BB"]["ledger_net"] == pytest.approx(-1.0, abs=1e-9)


def test_fixture_b_walk_ledger_net_to_the_cent():
    # Hero (BTN/SB) shoves, Villain1 (BB) folds. D = 1.0 (BB's forfeited
    # blind only -- the shover's bet is returned, uncalled, so contributes 0).
    contributions = {"shover": 0.0, "folder": 1.0}
    equities = {"shover": 1.0}
    result = ev.settle(CFG, contributions, equities)

    assert result["shover"]["ledger_net"] == pytest.approx(0.78, abs=1e-9)
    assert result["folder"]["ledger_net"] == pytest.approx(-1.0, abs=1e-9)


def test_fixture_b_shover_gets_no_rakeback_but_folder_does():
    # Distinguishes ledger_net (real hand-history bookkeeping, no rakeback) from
    # ev_net (ledger_net + rakeback, what the solver actually optimizes) as an
    # executable fact, not just a comment: the shover's contribution to the formed
    # pot is 0 (their bet was returned uncalled), so they earn no rakeback at all --
    # but the folder DID leave their blind in the pot, so they get a small pro-rata
    # rebate on it even though they lost the hand.
    contributions = {"shover": 0.0, "folder": 1.0}
    equities = {"shover": 1.0}
    result = ev.settle(CFG, contributions, equities)

    assert result["shover"]["rakeback"] == pytest.approx(0.0)
    assert result["shover"]["ev_net"] == result["shover"]["ledger_net"]

    assert result["folder"]["rakeback"] == pytest.approx(0.06, abs=1e-9)
    assert result["folder"]["ev_net"] == pytest.approx(-0.94, abs=1e-9)
    assert result["folder"]["ev_net"] != result["folder"]["ledger_net"]


def test_settle_rakeback_pool_sums_to_006_when_fully_allocated():
    # Sanity check on the general formula: when every contributor's rakeback share is
    # summed, it must reproduce the flat 0.06 pool exactly (nothing lost or invented).
    contributions = {"a": 8.0, "b": 8.0, "sb": 0.5, "bb": 1.0}
    equities = {"a": 0.6, "b": 0.4}
    result = ev.settle(CFG, contributions, equities)
    total_rakeback = sum(r["rakeback"] for r in result.values())
    assert total_rakeback == pytest.approx(0.06, abs=1e-9)


# ---------------------------------------------------------------------------------
# fold_baseline_ev_net -- now general over any info-set shape (ranges is a required
# argument since a fold's rakeback depends on how the rest of the hand plays out).
# ---------------------------------------------------------------------------------


def test_fold_baseline_sb_and_bb_hu():
    infosets = tree.build_infosets(HU_CONFIG)
    ranges = tree.init_ranges(infosets)
    sb_open = next(i for i in infosets if i.seat == "SB")
    bb_call = next(i for i in infosets if i.seat == "BB")

    assert ev.fold_baseline_ev_net(HU_CONFIG, sb_open, ranges) == pytest.approx(-0.44, abs=1e-9)
    assert ev.fold_baseline_ev_net(HU_CONFIG, bb_call, ranges) == pytest.approx(-0.94, abs=1e-9)


def test_fold_baseline_matches_reconciliation_table_signs():
    # Cross-check against the spec's own reconciliation numbers: folding SB's ev_net
    # should be more than the flat -0.5 (rakeback recovers a bit), and folding BB's
    # ev_net should be more than the flat -1.0, for the same reason -- never less.
    infosets = tree.build_infosets(HU_CONFIG)
    ranges = tree.init_ranges(infosets)
    sb_open = next(i for i in infosets if i.seat == "SB")
    bb_call = next(i for i in infosets if i.seat == "BB")
    assert ev.fold_baseline_ev_net(HU_CONFIG, sb_open, ranges) > -0.5
    assert ev.fold_baseline_ev_net(HU_CONFIG, bb_call, ranges) > -1.0


def test_fold_baseline_now_supports_three_seats_pending():
    # The old function raised NotImplementedError for exactly this shape (a fold
    # whose payoff depends on more than one subsequent seat's equilibrium strategy).
    # The generalized version must instead return a finite, sane number.
    cfg_3seat = GameConfig(seats=("SB", "MID", "BB"), blinds={"SB": 0.5, "BB": 1.0})
    infosets = tree.build_infosets(cfg_3seat)
    ranges = tree.init_ranges(infosets)
    sb_open = next(i for i in infosets if i.seat == "SB")
    result = ev.fold_baseline_ev_net(cfg_3seat, sb_open, ranges)
    assert np.isfinite(result)
    assert result < 0.0  # folding a posted blind can never be a net-positive baseline


def test_fold_baseline_sb_fold_fourmax_when_utg_and_btn_already_folded():
    # UTG and BTN folded before SB (both post 0, so no dead money); SB now folds too,
    # so BB never even gets a decision (edge case: the last seat facing an empty
    # history isn't a choice, it's an automatic walk). This must reduce to EXACTLY
    # the same -0.44 as the HU "SB folds" fixture, regardless of what BB's range
    # looks like, since BB's range is never actually consulted along this path.
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    ranges = tree.init_ranges(infosets, init_weight=1.0)
    sb_open = next(i for i in infosets if i.seat == "SB" and not i.shoved_before)
    assert ev.fold_baseline_ev_net(FOURMAX_CONFIG, sb_open, ranges) == pytest.approx(-0.44, abs=1e-9)


def test_fold_baseline_bb_fold_fourmax_facing_three_shovers():
    # BB has no seats after it, so this is a single trivial leaf: 3 live opponents
    # (UTG, BTN, SB all shoved), no dead money from earlier folds. The pot BB is
    # folding out of is the 3 other stacks (3*8=24) PLUS BB's own forfeited blind
    # (1.0), which itself becomes dead money in the pot exactly like any other
    # folder's blind -- total_pot = 25.0, and BB earns rakeback on its own 1.0
    # contribution to that pot.
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    ranges = tree.init_ranges(infosets, init_weight=1.0)
    bb_call = next(
        i for i in infosets if i.seat == "BB" and i.shoved_before == frozenset({"UTG", "BTN", "SB"})
    )
    result = ev.fold_baseline_ev_net(FOURMAX_CONFIG, bb_call, ranges)
    expected = -1.0 + 0.06 * (1.0 / 25.0)
    assert result == pytest.approx(expected, abs=1e-9)


def test_fold_baseline_utg_and_btn_are_blind_to_downstream_ranges_since_they_post_no_blind():
    # UTG and BTN post 0 blind, so ledger_net for folding is always exactly 0 for
    # them, and rakeback = pool * (this_blind / total_pot) = pool * (0 / anything) =
    # 0 too, REGARDLESS of how many seats after them end up shoving or what their
    # ranges look like. This is a strong regression check on the leaf-enumeration
    # math: a sign or accounting bug in the multi-leaf branching would very likely
    # perturb this away from exactly 0 for at least one of these fractional ranges.
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    ranges = tree.init_ranges(infosets, init_weight=0.0)
    ranges["SB|"] = np.full(169, 0.4)
    ranges["BB|SB"] = np.full(169, 0.6)
    ranges["BTN|UTG"] = np.full(169, 0.5)

    utg_open = next(i for i in infosets if i.seat == "UTG")
    btn_open = next(i for i in infosets if i.seat == "BTN" and not i.shoved_before)

    assert ev.fold_baseline_ev_net(FOURMAX_CONFIG, utg_open, ranges) == pytest.approx(0.0, abs=1e-9)
    assert ev.fold_baseline_ev_net(FOURMAX_CONFIG, btn_open, ranges) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------------
# shove_ev_net_vec -- replaces the old HU-only call_ev_net_vec (last seat, exactly
# one opponent) and open_shove_ev_net_vec (exactly one seat left to act). Every case
# below that used to be split across those two functions is ported to call this one
# unified function instead, and must still produce byte-identical results.
# ---------------------------------------------------------------------------------


def test_shove_ev_net_vec_bb_call_matches_settle_for_concentrated_range():
    # Build a tiny fake 169x169 table with one known entry and confirm the vectorized
    # showdown formula agrees exactly with a direct settle() call for that hand.
    aa, kk = cards.canon_index("AA"), cards.canon_index("KK")
    eq_aa_vs_kk = 0.82
    table = np.full((169, 169), 0.5)
    table[aa, kk] = eq_aa_vs_kk
    table[kk, aa] = 1 - eq_aa_vs_kk

    opp_range = np.zeros(169)
    opp_range[kk] = 1.0  # opponent always has exactly KK

    infosets = tree.build_infosets(HU_CONFIG)
    bb_call = next(i for i in infosets if i.seat == "BB")
    ranges = {"SB|": opp_range}
    vec = ev.shove_ev_net_vec(HU_CONFIG, bb_call, ranges, table)

    direct = ev.settle(
        HU_CONFIG,
        {"hero": HU_CONFIG.stack_bb, "villain": HU_CONFIG.stack_bb},
        {"hero": eq_aa_vs_kk, "villain": 1 - eq_aa_vs_kk},
    )
    assert vec[aa] == pytest.approx(direct["hero"]["ev_net"], abs=1e-9)


def test_shove_ev_net_vec_reduces_to_steal_when_opponent_never_calls():
    table = np.full((169, 169), 0.5)
    opp_call_range = np.zeros(169)  # BB never calls anything

    infosets = tree.build_infosets(HU_CONFIG)
    sb_open = next(i for i in infosets if i.seat == "SB")
    ranges = {"BB|SB": opp_call_range}
    vec = ev.shove_ev_net_vec(HU_CONFIG, sb_open, ranges, table)

    expected_steal = HU_CONFIG.blind_of("BB") - HU_CONFIG.drop_bb  # 1.0 - 0.22 = 0.78
    assert np.allclose(vec, expected_steal)


def test_shove_ev_net_vec_reduces_to_showdown_when_opponent_always_calls():
    aa, kk = cards.canon_index("AA"), cards.canon_index("KK")
    table = np.full((169, 169), 0.5)
    table[aa, kk] = 0.82
    table[kk, aa] = 0.18

    opp_call_range = np.ones(169)  # BB always calls, with its whole range

    infosets = tree.build_infosets(HU_CONFIG)
    sb_open = next(i for i in infosets if i.seat == "SB")
    ranges = {"BB|SB": opp_call_range}
    vec = ev.shove_ev_net_vec(HU_CONFIG, sb_open, ranges, table)

    # p_call=1 means the blend must reduce to exactly the showdown formula against
    # BB's actual (full, uniform) range -- computed the same way the implementation
    # does, not by hand-picking a single opponent hand (BB's range spans all 169).
    eq_vector = equity.hand_vs_range_equity_vectorized(opp_call_range, table)
    expected = ev.showdown_ev_net_vec(HU_CONFIG, 0.0, eq_vector, num_opponents=1)
    assert np.allclose(vec, expected)
    # and the AA row should still be pulled up relative to the flat 0.5 baseline by
    # its edge against KK specifically, confirming the table entry is actually used
    assert vec[aa] > vec[kk]


def test_shove_ev_net_vec_blends_showdown_and_steal_for_fractional_call_probability():
    # Neither extreme (0% or 100% call) exercises the actual probability-weighted
    # blend -- this picks an intermediate p_call and hand-verifies the exact formula
    # EV_shove = p_call * showdown_ev + (1 - p_call) * steal_ev.
    aa, kk = cards.canon_index("AA"), cards.canon_index("KK")
    table = np.full((169, 169), 0.5)
    table[aa, kk] = 0.82
    table[kk, aa] = 0.18

    p_call = 0.3
    opp_call_range = np.full(169, p_call)

    infosets = tree.build_infosets(HU_CONFIG)
    sb_open = next(i for i in infosets if i.seat == "SB")
    ranges = {"BB|SB": opp_call_range}
    vec = ev.shove_ev_net_vec(HU_CONFIG, sb_open, ranges, table)

    eq_vector = equity.hand_vs_range_equity_vectorized(opp_call_range, table)
    showdown = ev.showdown_ev_net_vec(HU_CONFIG, 0.0, eq_vector, num_opponents=1)
    steal = ev.steal_ev_net(HU_CONFIG, HU_CONFIG.blind_of("BB"))
    expected = p_call * showdown + (1 - p_call) * steal
    assert np.allclose(vec, expected)


def test_shove_ev_net_vec_utg_collapses_to_steal_when_nobody_ever_calls():
    # UTG has THREE seats after it in 4-max -- if all three ranges are all-fold, all
    # 2**3 leaves collapse into the single "everybody folds" leaf, and the result
    # must match the spec's own reconciliation number exactly: D=1.5 -> +1.28.
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    ranges = tree.init_ranges(infosets, init_weight=0.0)
    utg_open = next(i for i in infosets if i.seat == "UTG")
    table = np.full((169, 169), 0.5)  # never consulted on this all-steal path

    vec = ev.shove_ev_net_vec(FOURMAX_CONFIG, utg_open, ranges, table)

    total_dead_money = sum(FOURMAX_CONFIG.blind_of(s) for s in ("BTN", "SB", "BB"))
    expected_steal = total_dead_money - FOURMAX_CONFIG.drop_bb
    assert expected_steal == pytest.approx(1.28, abs=1e-9)
    assert np.allclose(vec, expected_steal)


def test_shove_ev_net_vec_utg_collapses_to_full_multiway_showdown_when_everyone_always_calls():
    # All-ones ranges downstream collapse every leaf's probability onto the single
    # "everyone shoves" realization: a genuine 4-way pot needing multiway_equity_mc
    # (3 live opponents). Uses the real evaluator with an explicitly seeded RNG so an
    # independent, separately-seeded call to multiway_equity_mc reproduces the exact
    # same samples -- letting this be an exact-match test despite being Monte Carlo.
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    ranges = tree.init_ranges(infosets, init_weight=1.0)
    utg_open = next(i for i in infosets if i.seat == "UTG")
    table = np.full((169, 169), 0.5)  # irrelevant here; the num_opponents>=2 path ignores it
    real_ev = equity.get_evaluator()

    vec = ev.shove_ev_net_vec(
        FOURMAX_CONFIG, utg_open, ranges, table,
        evaluator=real_ev, mc_iterations=200, rng=np.random.default_rng(42),
    )

    all_ones = np.ones(169)
    expected_eq = equity.multiway_equity_mc(
        [all_ones, all_ones, all_ones], real_ev, 200, rng=np.random.default_rng(42)
    )
    expected = ev.showdown_ev_net_vec(FOURMAX_CONFIG, 0.0, expected_eq, num_opponents=3)
    assert np.allclose(vec, expected)


def test_shove_ev_net_vec_sb_facing_utg_blends_pairwise_and_multiway_leaves():
    # SB|UTG (BTN folded before SB, UTG already shoved): only BB is left to act, so
    # there are exactly 2 leaves -- "BB folds" (SB faces UTG alone, a pure 1-opponent
    # showdown against the exact table) and "BB shoves too" (a genuine 2-opponent
    # multiway pot). This exercises BOTH code paths inside one shove-EV computation
    # and confirms the probability-weighted combination is exactly right, including
    # each leaf's own correct dead-money accounting (BB's forfeited blind only
    # belongs to the "BB folds" leaf's pot, not the "BB shoves" leaf's).
    aa, kk = cards.canon_index("AA"), cards.canon_index("KK")
    utg_range = np.zeros(169)
    utg_range[kk] = 1.0  # UTG's fixed range: always exactly KK

    bb_call_prob = 0.3
    bb_range = np.full(169, bb_call_prob)

    infosets = tree.build_infosets(FOURMAX_CONFIG)
    ranges = tree.init_ranges(infosets, init_weight=0.0)
    ranges["UTG|"] = utg_range
    ranges["BB|SB|UTG"] = bb_range

    table = np.full((169, 169), 0.5)
    table[aa, kk] = 0.82
    table[kk, aa] = 0.18

    infoset = next(
        i for i in infosets if i.seat == "SB" and i.shoved_before == frozenset({"UTG"})
    )
    real_ev = equity.get_evaluator()
    vec = ev.shove_ev_net_vec(
        FOURMAX_CONFIG, infoset, ranges, table,
        evaluator=real_ev, mc_iterations=300, rng=np.random.default_rng(7),
    )

    # Leaf 1 ("BB folds", prob 1-p): pure pairwise showdown vs UTG; BB's forfeited
    # blind (1.0) is dead money added to this leaf's pot.
    eq_vs_utg = equity.hand_vs_range_equity_vectorized(utg_range, table)
    showdown_leaf = ev.showdown_ev_net_vec(
        FOURMAX_CONFIG, infoset.dead_money_bb + FOURMAX_CONFIG.blind_of("BB"), eq_vs_utg, num_opponents=1
    )

    # Leaf 2 ("BB shoves too", prob p): genuine 2-opponent multiway pot, no dead money
    # (nobody folded after SB in this branch).
    eq_multiway = equity.multiway_equity_mc(
        [utg_range, bb_range], real_ev, 300, rng=np.random.default_rng(7)
    )
    multiway_leaf = ev.showdown_ev_net_vec(
        FOURMAX_CONFIG, infoset.dead_money_bb, eq_multiway, num_opponents=2
    )

    expected = (1 - bb_call_prob) * showdown_leaf + bb_call_prob * multiway_leaf
    assert np.allclose(vec, expected)
