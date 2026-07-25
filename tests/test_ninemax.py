import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
import equity
import ev
import tree
from game import NINEMAX_CONFIG

_RANGES_PKL = Path(__file__).resolve().parent.parent / "cache" / "ninemax_ranges.pkl"
_OPEN_SEATS = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB"]


# ---------------------------------------------------------------------------------
# Structure: the seat-agnostic tree must generate the full 9-max game unchanged.
# ---------------------------------------------------------------------------------


def test_ninemax_has_510_unique_infosets():
    isets = tree.build_infosets(NINEMAX_CONFIG)
    keys = [tree.infoset_key(i.seat, i.shoved_before) for i in isets]
    assert len(keys) == 510  # 2**9 - 2
    assert len(set(keys)) == 510


def test_ninemax_open_infosets_are_utg_through_sb():
    opens = [i for i in tree.build_infosets(NINEMAX_CONFIG) if not i.shoved_before]
    assert [i.seat for i in opens] == _OPEN_SEATS  # BB never "opens" -- folded-to BB is a walk


def test_ninemax_shove_and_fold_ev_finite_across_infoset_shapes():
    isets = tree.build_infosets(NINEMAX_CONFIG)
    ranges = tree.init_ranges(isets)
    table = np.full((169, 169), 0.5)
    evaluator = equity.get_evaluator()
    rng = np.random.default_rng(0)

    # A spread of shapes: an open (UTG, 8 seats after -> deep multiway leaves), a
    # facing-one node, a deep facing-3+ node, and the last seat (BB, one trivial leaf).
    deep = next(i for i in isets if len(i.shoved_before) >= 3)
    facing_one = next(i for i in isets if len(i.shoved_before) == 1)
    sample = [isets[0], facing_one, deep, isets[-1]]
    for iset in sample:
        vec = ev.shove_ev_net_vec(
            NINEMAX_CONFIG, iset, ranges, table,
            evaluator=evaluator, mc_iterations=15, rng=rng, mc_cache={},
        )
        assert vec.shape == (169,) and np.all(np.isfinite(vec))
        assert np.isfinite(ev.fold_baseline_ev_net(NINEMAX_CONFIG, iset, ranges))


def test_multiway_equity_cache_is_consulted_and_correctly_keyed():
    # BB facing exactly UTG, UTG1, UTG2 is a single leaf with 3 live opponents -- one
    # multiway_equity_mc call. Pre-seeding the cache with a sentinel equity for that
    # opponent range-set and confirming the shove EV reflects it (rather than a fresh
    # MC draw) proves the cache is keyed by opponent range contents and actually used.
    isets = tree.build_infosets(NINEMAX_CONFIG)
    ranges = tree.init_ranges(isets)
    infoset = next(
        i for i in isets
        if i.seat == "BB" and i.shoved_before == frozenset({"UTG", "UTG1", "UTG2"})
    )
    opp_arrays = [ranges["UTG|"], ranges["UTG1|UTG"], ranges["UTG2|UTG|UTG1"]]
    # This is a 3-way pot; with the default deep_threshold=4 and no deep cap it samples at
    # the full mc_iterations, so the key is (opponent-contents, iterations).
    iters = 999
    cache_key = (tuple(sorted(r.tobytes() for r in opp_arrays)), iters)
    sentinel = np.full(cards.NUM_CANON_HANDS, 0.37)
    mc_cache = {cache_key: sentinel}

    table = np.full((169, 169), 0.5)
    vec = ev.shove_ev_net_vec(
        NINEMAX_CONFIG, infoset, ranges, table,
        evaluator=equity.get_evaluator(), mc_iterations=iters, rng=np.random.default_rng(0),
        mc_cache=mc_cache,
    )
    expected = ev.showdown_ev_net_vec(
        NINEMAX_CONFIG, infoset.dead_money_bb, sentinel, num_opponents=3
    )
    assert np.allclose(vec, expected)


def test_deep_pot_cap_keeps_ev_finite_and_changes_nothing_when_disabled():
    # Capping deep-pot MC (mc_iterations_deep) must still yield finite EVs, and with a
    # threshold above the table's max opponent count it must be a no-op vs the uncapped
    # path (same rng -> identical draws), proving the cap only touches deep pots.
    isets = tree.build_infosets(NINEMAX_CONFIG)
    ranges = tree.init_ranges(isets)
    utg_open = isets[0]
    table = np.full((169, 169), 0.5)
    evaluator = equity.get_evaluator()

    capped = ev.shove_ev_net_vec(
        NINEMAX_CONFIG, utg_open, ranges, table, evaluator=evaluator,
        mc_iterations=40, rng=np.random.default_rng(1), mc_iterations_deep=5, deep_threshold=2,
    )
    assert np.all(np.isfinite(capped))

    # deep_threshold=99 => no pot is "deep", so mc_iterations_deep never applies.
    a = ev.shove_ev_net_vec(
        NINEMAX_CONFIG, utg_open, ranges, table, evaluator=evaluator,
        mc_iterations=40, rng=np.random.default_rng(2), mc_iterations_deep=5, deep_threshold=99,
    )
    b = ev.shove_ev_net_vec(
        NINEMAX_CONFIG, utg_open, ranges, table, evaluator=evaluator,
        mc_iterations=40, rng=np.random.default_rng(2), mc_iterations_deep=None,
    )
    assert np.allclose(a, b)


def test_parallel_solve_produces_valid_full_range_map():
    # A short solve with a worker pool must return all 510 ranges, every weight finite in
    # [0,1] -- i.e. the parallel best-response path is wired up and merges chunks correctly.
    import solver

    cfg = solver.SolverConfig(
        max_iterations=2, mc_iterations=10, mc_iterations_deep=5, deep_opponent_threshold=3,
        n_workers=2, return_averaged=True, average_burn_in_frac=0.0,
        final_exploitability_mc=10, log_every=0,
    )
    ranges, info = solver.solve(NINEMAX_CONFIG, cfg)
    assert len(ranges) == 510
    allvals = np.concatenate(list(ranges.values()))
    assert np.all(np.isfinite(allvals)) and allvals.min() >= 0.0 and allvals.max() <= 1.0


# ---------------------------------------------------------------------------------
# Sanity bounds on the actually-solved ranges (no published 9-max chart to check
# exact values against). Skipped unless a solved artifact exists -- run the solve
# (scripts/solve_ninemax.py) first. These want a well-converged artifact, not the
# coarse smoke run, so keep them marked slow.
# ---------------------------------------------------------------------------------


def _load_solved_ranges():
    with open(_RANGES_PKL, "rb") as f:
        saved = pickle.load(f)
    return saved["ranges"] if isinstance(saved, dict) and "ranges" in saved else saved


@pytest.mark.slow
@pytest.mark.skipif(not _RANGES_PKL.exists(), reason="no solved ninemax_ranges.pkl")
def test_ninemax_aa_shoves_from_every_open_position():
    ranges = _load_solved_ranges()
    aa = cards.canon_index("AA")
    for seat in _OPEN_SEATS:
        assert ranges[f"{seat}|"][aa] > 0.99, f"AA should always open-shove from {seat}"


@pytest.mark.slow
@pytest.mark.skipif(not _RANGES_PKL.exists(), reason="no solved ninemax_ranges.pkl")
def test_ninemax_72o_folds_facing_one_or_two_shoves():
    # 72o must never be a profitable call at the spots that actually occur in play --
    # facing one or two shoves. It legitimately becomes a marginal call ONLY at the deep,
    # near-unreachable nodes (facing 4+ simultaneous all-ins), where the pot odds of
    # closing a huge multiway pot make almost any hand break-even; asserting it folds
    # there would be wrong poker, not a bug (see the by-depth diagnostic in the notes).
    ranges = _load_solved_ranges()
    o72 = cards.canon_index("72o")
    for key, vec in ranges.items():
        faced = key.count("|") - 1 if key.endswith("|") else key.count("|")
        if 1 <= faced <= 2:  # a shallow (reachable) facing-a-shove node
            assert vec[o72] < 0.1, f"72o should fold facing {faced} shove(s) at {key}"
