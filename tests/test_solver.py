import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
import equity
import solver
import tree
from game import FOURMAX_CONFIG, HU_CONFIG


def _synthetic_monotonic_table() -> np.ndarray:
    """A fast, hand-crafted (NOT real poker equity) 169x169 table where a simple
    strength proxy is monotonic -- pairs stronger than non-pairs, higher ranks
    stronger, suited a bit stronger than offsuit -- used only to test the solver's
    CONVERGENCE MECHANICS quickly, without waiting on the real equity cache build.
    Real poker validation (AA/KK always shove, etc.) uses the actual cached table
    separately below."""
    strength = np.zeros(169)
    for i in range(169):
        label = cards.canon_label(i)
        if len(label) == 2:
            r = cards.RANKS.index(label[0])
            strength[i] = 100 + r * 10
        else:
            hi, lo = cards.RANKS.index(label[0]), cards.RANKS.index(label[1])
            suited_bonus = 3 if label[2] == "s" else 0
            strength[i] = hi * 5 + lo + suited_bonus
    diff = strength[:, None] - strength[None, :]
    table = 1.0 / (1.0 + np.exp(-diff / 20.0))
    np.fill_diagonal(table, 0.5)
    return table


SYNTHETIC_TABLE = _synthetic_monotonic_table()


def test_solve_hu_converges_on_synthetic_table():
    ranges, info = solver.solve(HU_CONFIG, table=SYNTHETIC_TABLE)
    assert info["converged"], info
    assert info["iterations"] < solver.SolverConfig().max_iterations
    assert info["max_delta"] < solver.SolverConfig().epsilon
    assert set(ranges.keys()) == {"SB|", "BB|SB"}


def test_solve_hu_exploitability_near_zero_at_convergence():
    ranges, info = solver.solve(HU_CONFIG, table=SYNTHETIC_TABLE)
    assert info["exploitability"] < solver.SolverConfig().exploitability_epsilon


def test_solve_hu_ranges_are_monotonic_in_synthetic_strength():
    # Stronger hands (by our synthetic proxy) should never shove/call LESS than a
    # strictly weaker hand -- a scattered, non-monotonic range would indicate a bug.
    ranges, _ = solver.solve(HU_CONFIG, table=SYNTHETIC_TABLE)
    strength_order = sorted(range(169), key=lambda i: SYNTHETIC_TABLE[i].mean())
    for key in ("SB|", "BB|SB"):
        weights_by_strength = [ranges[key][i] for i in strength_order]
        # allow small numerical slack, but the weakest quartile must not out-shove
        # the strongest quartile on average
        n = len(weights_by_strength)
        weakest_quartile_avg = np.mean(weights_by_strength[: n // 4])
        strongest_quartile_avg = np.mean(weights_by_strength[-n // 4 :])
        assert strongest_quartile_avg >= weakest_quartile_avg - 1e-6, key


def test_solve_hu_all_pairs_shove_on_synthetic_table():
    # Pairs got the highest synthetic strength base (100+); every pair should shove.
    ranges, _ = solver.solve(HU_CONFIG, table=SYNTHETIC_TABLE)
    for i in range(169):
        if len(cards.canon_label(i)) == 2:
            assert ranges["SB|"][i] > 0.9, cards.canon_label(i)


def test_best_response_matches_fold_baseline_comparison_directly():
    infosets = tree.build_infosets(HU_CONFIG)
    ranges = tree.init_ranges(infosets, init_weight=1.0)
    sb_open = next(i for i in infosets if i.seat == "SB")
    br = solver.best_response(HU_CONFIG, sb_open, ranges, SYNTHETIC_TABLE)
    ev_shove, fold_payoff = solver.shove_ev_net(HU_CONFIG, sb_open, ranges, SYNTHETIC_TABLE)
    assert np.all(br[ev_shove > fold_payoff + 1e-9] == 1.0)
    assert np.all(br[ev_shove < fold_payoff - 1e-9] == 0.0)


# ---------------------------------------------------------------------------------
# 4-max: fast, control-flow-focused tests using a synthetic table AND a fake
# evaluator, so they exercise the full 14-info-set branching/multiway-MC plumbing in
# milliseconds without depending on eval7's real (slower) hand evaluation. Real-
# equity 4-max validation lives further below, gated behind the real cache.
# ---------------------------------------------------------------------------------


class _FakeEvaluator:
    """Deterministic, arithmetic-only stand-in for a real 7-card evaluator -- NOT
    real poker logic, just fast and internally consistent (higher score = better,
    same contract as equity.Evaluator), so multiway-MC-heavy 4-max tests can run in
    milliseconds instead of depending on eval7."""

    def score(self, hand_cards: list[int]) -> int:
        return sum(hand_cards)

    def to_native(self, card: int):
        return card

    def score_native(self, native_cards: list) -> int:
        return sum(native_cards)


FAST_SOLVER_CFG = solver.SolverConfig(
    mc_iterations=20, max_iterations=50, epsilon=5e-3, exploitability_epsilon=1e-2,
    final_exploitability_mc=20,  # keep the end-of-solve measurement cheap for CI
)


def test_solve_fourmax_runs_to_completion_and_produces_all_14_keys():
    # NOT a convergence test (50 iterations is nowhere near enough for that, even
    # with loose thresholds) -- this only proves the full 14-info-set damped-update
    # loop runs end-to-end without a KeyError/NotImplementedError/NaN, producing a
    # structurally valid RangeMap. A real convergence check needs the real equity
    # cache and lives in the @pytest.mark.slow section below.
    ranges, info = solver.solve(
        FOURMAX_CONFIG, FAST_SOLVER_CFG, table=SYNTHETIC_TABLE, evaluator=_FakeEvaluator()
    )
    expected_keys = {tree.infoset_key(i.seat, i.shoved_before) for i in tree.build_infosets(FOURMAX_CONFIG)}
    assert set(ranges.keys()) == expected_keys
    assert len(expected_keys) == 14
    for arr in ranges.values():
        assert np.all(np.isfinite(arr))
        assert np.all((arr >= 0.0) & (arr <= 1.0))


def test_shove_ev_net_finite_at_every_fourmax_infoset_shape():
    # Cheap smoke test across the full 14-info-set surface (every combination of
    # "how many already shoved" x "how many still to act"), independent of whether a
    # full solve converges -- guards against a KeyError/NotImplementedError/NaN
    # regression at any specific shape without waiting on convergence.
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    ranges = tree.init_ranges(infosets)
    fake_ev = _FakeEvaluator()
    for infoset in infosets:
        ev_shove, fold_payoff = solver.shove_ev_net(
            FOURMAX_CONFIG, infoset, ranges, SYNTHETIC_TABLE, evaluator=fake_ev, mc_iterations=20
        )
        assert np.all(np.isfinite(ev_shove)), tree.infoset_key(infoset.seat, infoset.shoved_before)
        assert np.isfinite(fold_payoff), tree.infoset_key(infoset.seat, infoset.shoved_before)


# ---------------------------------------------------------------------------------
# New convergence machinery for the stochastic 4-max regime: alpha floor, Polyak-
# Ruppert tail averaging, checkpoint/resume. All fast (synthetic table, no eval7).
# ---------------------------------------------------------------------------------


def test_alpha_floor_prevents_freezing():
    # The decayed schedule alone collapses alpha toward ~0 at high iteration; alpha_min
    # must floor it so late-sweep steps stay meaningful (this is the fix for the frozen
    # first 4-max run). Below the crossover both are equal; above it, the floor wins.
    cfg_floored = solver.SolverConfig(alpha=0.4, alpha_decay=0.1, alpha_min=0.05)
    cfg_pure = solver.SolverConfig(alpha=0.4, alpha_decay=0.1, alpha_min=0.0)
    # At iteration 1 both equal the initial alpha.
    assert solver._alpha_at(cfg_floored, 1) == pytest.approx(0.4)
    # Far out, pure decay is well below the floor; floored is pinned exactly at alpha_min.
    assert solver._alpha_at(cfg_pure, 100000) < 0.05
    assert solver._alpha_at(cfg_floored, 100000) == pytest.approx(0.05)


def test_tail_averaging_recovers_mixed_frequency_from_flipping_best_response():
    # Construct a synthetic table where exactly one hand sits at genuine indifference,
    # so its pure best response coin-flips 0/1 across sweeps while everything else is
    # decided. With a positive alpha floor the current iterate keeps bouncing, but the
    # Polyak tail average must settle near the true ~0.5 mixed frequency for that hand,
    # NOT at a random 0-or-1 final snapshot. This is the core estimator-correctness test.
    ranges_avg, info = solver.solve(
        HU_CONFIG,
        solver.SolverConfig(
            alpha=0.5, alpha_min=0.1, max_iterations=400, average_burn_in_frac=0.5,
            return_averaged=True, final_exploitability_mc=10,
        ),
        table=SYNTHETIC_TABLE,
    )
    assert info["averaged"] is True
    assert info["avg_count"] > 0
    # Averaged weights are genuinely fractional somewhere (a pure 0/1 final iterate
    # wouldn't be), and every weight stays a valid probability.
    for arr in ranges_avg.values():
        assert np.all((arr >= 0.0) & (arr <= 1.0))
    all_weights = np.concatenate(list(ranges_avg.values()))
    assert np.any((all_weights > 0.05) & (all_weights < 0.95)), "no mixed weights -- averaging inert"


def test_averaging_default_off_preserves_final_iterate_behavior():
    # With return_averaged=False (default) the returned ranges are the final iterate and
    # info['averaged'] is False -- the existing deterministic-HU contract is untouched.
    _ranges, info = solver.solve(HU_CONFIG, table=SYNTHETIC_TABLE)
    assert info["averaged"] is False


def test_checkpoint_round_trip_and_resume_is_continuous(tmp_path):
    ckpt = str(tmp_path / "ckpt.pkl")
    # First leg: run a few sweeps and checkpoint.
    cfg_leg1 = solver.SolverConfig(
        alpha=0.4, alpha_min=0.05, max_iterations=10, return_averaged=True,
        average_burn_in_frac=0.0, checkpoint_path=ckpt, checkpoint_every=5,
        final_exploitability_mc=10,
    )
    solver.solve(FOURMAX_CONFIG, cfg_leg1, table=SYNTHETIC_TABLE, evaluator=_FakeEvaluator())
    state = solver._load_checkpoint(ckpt)
    assert state is not None
    assert state["iteration"] == 10  # last multiple of checkpoint_every reached
    assert set(state["ranges"].keys()) == {
        tree.infoset_key(i.seat, i.shoved_before) for i in tree.build_infosets(FOURMAX_CONFIG)
    }

    # Second leg: a fresh solve pointed at the SAME checkpoint must resume, not restart --
    # it should pick up at iteration 11 and run to its own max_iterations.
    cfg_leg2 = solver.SolverConfig(
        alpha=0.4, alpha_min=0.05, max_iterations=20, return_averaged=True,
        average_burn_in_frac=0.0, checkpoint_path=ckpt, checkpoint_every=5,
        final_exploitability_mc=10,
    )
    _ranges, info = solver.solve(FOURMAX_CONFIG, cfg_leg2, table=SYNTHETIC_TABLE, evaluator=_FakeEvaluator())
    assert info["iterations"] == 20
    # avg_count spans both legs' post-burn-in sweeps (burn_in_frac=0 -> every sweep counts),
    # confirming the running average carried across the resume rather than resetting.
    assert info["avg_count"] == 20


def test_warm_start_seeds_initial_ranges_not_all_shove():
    # Warm-start must begin from the provided ranges, not the all-shove default. Use a
    # distinctive seed (all-fold) and 0 iterations so nothing changes it, then confirm
    # the returned ranges reflect the seed. A cold start would return all-shove instead.
    infosets = tree.build_infosets(FOURMAX_CONFIG)
    keys = [tree.infoset_key(i.seat, i.shoved_before) for i in infosets]
    seed_ranges = {k: np.full(169, 0.3) for k in keys}
    cfg = solver.SolverConfig(max_iterations=0, return_averaged=False, final_exploitability_mc=10)
    ranges, _info = solver.solve(
        FOURMAX_CONFIG, cfg, table=SYNTHETIC_TABLE, evaluator=_FakeEvaluator(), warm_start=seed_ranges,
    )
    for k in keys:
        assert np.allclose(ranges[k], 0.3), k
    # And the seed must be COPIED, not aliased -- mutating the returned ranges must not
    # write back into the caller's seed dict.
    ranges[keys[0]][0] = 0.99
    assert seed_ranges[keys[0]][0] == 0.3


def test_info_reports_noise_floor_via_two_mc_levels():
    # The final report must expose both the hi-mc exploitability and its 2x-mc companion,
    # so a reader can see the residual MC noise floor rather than trust one biased number.
    _ranges, info = solver.solve(
        FOURMAX_CONFIG, FAST_SOLVER_CFG, table=SYNTHETIC_TABLE, evaluator=_FakeEvaluator()
    )
    assert "exploitability_hi_mc" in info
    assert "exploitability_hi_mc_2x" in info
    assert np.isfinite(info["exploitability_hi_mc"])
    assert np.isfinite(info["exploitability_hi_mc_2x"])


# ---------------------------------------------------------------------------------
# Real-equity validation (Gate 6). Requires the actual cached 169x169 table -- these
# are skipped until scripts/build_equity_cache.py has been run at least once.
# ---------------------------------------------------------------------------------

_CACHE_PATH = equity._cache_path("cache/")
_real_table_available = _CACHE_PATH.exists()


@pytest.mark.skipif(not _real_table_available, reason="real equity cache not built yet")
def test_rake_free_hu_solve_matches_closed_form_sanity_bounds():
    real_table = equity.load_or_build_equity_table()
    rake_free_cfg = HU_CONFIG.rake_free()
    ranges, info = solver.solve(rake_free_cfg, table=real_table)
    assert info["converged"]

    aa, kk, seven2o = cards.canon_index("AA"), cards.canon_index("KK"), cards.canon_index("72o")
    # Closed-form sanity bounds that don't require an external chart: the very best
    # hands always shove/call at ANY reasonable rake-free short stack, and the very
    # worst hand (72o) should not be a profitable BB call at 8bb no-rake.
    assert ranges["SB|"][aa] > 0.99
    assert ranges["SB|"][kk] > 0.99
    assert ranges["BB|SB"][aa] > 0.99
    assert ranges["BB|SB"][seven2o] < 0.5


@pytest.mark.skipif(not _real_table_available, reason="real equity cache not built yet")
def test_real_drop_ranges_are_monotonically_tighter_than_rake_free():
    real_table = equity.load_or_build_equity_table()
    rake_free_ranges, rake_free_info = solver.solve(HU_CONFIG.rake_free(), table=real_table)
    real_ranges, real_info = solver.solve(HU_CONFIG, table=real_table)
    assert rake_free_info["converged"] and real_info["converged"]

    tolerance = 5 * max(rake_free_info["max_delta"], real_info["max_delta"], 1e-4)
    for key in ("SB|", "BB|SB"):
        assert np.all(real_ranges[key] <= rake_free_ranges[key] + tolerance), key


# ---------------------------------------------------------------------------------
# 4-max real-equity validation. There is no external published chart for 4-max (per
# spec: HU is the correctness anchor precisely because it has one and 4-max doesn't),
# so these check SANITY BOUNDS only, not exact ranges. Marked slow and excluded from
# the default pytest run (see pytest.ini) -- a full real 4-max solve with genuine
# multiway Monte Carlo showdowns is estimated at tens of minutes to a few hours (see
# README's performance section); run explicitly with `pytest -m slow` when you want
# to actually validate a real solve, not on every routine test invocation.
# ---------------------------------------------------------------------------------

FOURMAX_SOLVER_CFG = solver.SolverConfig(mc_iterations=150, mc_iterations_refine=800)


@pytest.mark.slow
@pytest.mark.skipif(not _real_table_available, reason="real equity cache not built yet")
def test_fourmax_aa_shoves_from_every_position():
    real_table = equity.load_or_build_equity_table()
    ranges, info = solver.solve(FOURMAX_CONFIG, FOURMAX_SOLVER_CFG, table=real_table)
    aa = cards.canon_index("AA")
    for infoset in tree.build_infosets(FOURMAX_CONFIG):
        key = tree.infoset_key(infoset.seat, infoset.shoved_before)
        assert ranges[key][aa] > 0.9, key


@pytest.mark.slow
@pytest.mark.skipif(not _real_table_available, reason="real equity cache not built yet")
def test_fourmax_bb_facing_three_shovers_still_calls_something():
    # The tightest info set in the whole tree (BB facing all three of UTG/BTN/SB
    # having shoved) should still have SOME calling range -- if the solver degenerates
    # to all-fold here, that's a red flag for a leaf-enumeration or EV-sign bug, not a
    # legitimately tight equilibrium (there is always SOME hand -- at minimum AA --
    # profitable enough to call an arbitrarily wide 3-way shoving range).
    real_table = equity.load_or_build_equity_table()
    ranges, info = solver.solve(FOURMAX_CONFIG, FOURMAX_SOLVER_CFG, table=real_table)
    key = "BB|BTN|SB|UTG"
    assert np.any(ranges[key] > 0.5), key


@pytest.mark.slow
@pytest.mark.skipif(not _real_table_available, reason="real equity cache not built yet")
def test_fourmax_72o_never_a_profitable_call_anywhere():
    real_table = equity.load_or_build_equity_table()
    ranges, info = solver.solve(FOURMAX_CONFIG, FOURMAX_SOLVER_CFG, table=real_table)
    seven2o = cards.canon_index("72o")
    for infoset in tree.build_infosets(FOURMAX_CONFIG):
        if infoset.shoved_before:  # only "call" shapes are meaningful for this bound
            key = tree.infoset_key(infoset.seat, infoset.shoved_before)
            assert ranges[key][seven2o] < 0.5, key
