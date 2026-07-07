import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cards
import equity


@pytest.fixture(scope="module")
def ev():
    e = equity.get_evaluator()
    equity.smoke_check(e)
    return e


def test_smoke_check_passes_on_active_evaluator(ev):
    equity.smoke_check(ev)  # must not raise


def test_smoke_check_catches_wrong_polarity(ev):
    class InvertedEvaluator:
        """Wraps the real evaluator but negates its score -- simulates exactly the
        real bug smoke_check must catch (e.g. forgetting to negate treys' lower-is-
        better ranking, or double-negating eval7's)."""

        def score(self, hand_cards):
            return -ev.score(hand_cards)

    with pytest.raises(AssertionError):
        equity.smoke_check(InvertedEvaluator())


def test_aa_vs_kk_approx_82_percent(ev):
    aa, kk = cards.canon_index("AA"), cards.canon_index("KK")
    eq_aa, eq_kk = equity.equity_pair_exact(aa, kk, ev)
    assert eq_aa == pytest.approx(0.82, abs=0.01)
    assert eq_aa + eq_kk == pytest.approx(1.0)


def test_coinflip_pair_vs_two_overcards(ev):
    # 22 vs AKo is a classic near-coinflip: pair racing two overcards.
    two2, ako = cards.canon_index("22"), cards.canon_index("AKo")
    eq_22, eq_ako = equity.equity_pair_exact(two2, ako, ev)
    assert eq_22 == pytest.approx(0.52, abs=0.03)
    assert eq_22 + eq_ako == pytest.approx(1.0)


def test_dominated_hand_loses_big(ev):
    # AKo dominates AQo hard (same ace, better kicker) -- should be well above 70%.
    ako, aqo = cards.canon_index("AKo"), cards.canon_index("AQo")
    eq_ako, eq_aqo = equity.equity_pair_exact(ako, aqo, ev)
    assert eq_ako > 0.70
    assert eq_aqo < 0.30


def test_suit_overlap_actually_changes_equity(ev):
    # Regression guard for the suit-relabeling-orbit logic itself: AA vs KK equity
    # must differ depending on suit overlap (same-suits vs disjoint-suits), confirming
    # equity_pair_exact isn't silently collapsing to a single representative combo.
    ac, ad = cards.card_int("A", "c"), cards.card_int("A", "d")
    kc, kd = cards.card_int("K", "c"), cards.card_int("K", "d")
    kh, ks = cards.card_int("K", "h"), cards.card_int("K", "s")
    same_suits = equity._exact_equity_one_combo((ac, ad), (kc, kd), ev)
    disjoint_suits = equity._exact_equity_one_combo((ac, ad), (kh, ks), ev)
    assert same_suits != pytest.approx(disjoint_suits, abs=1e-9)
    assert same_suits > disjoint_suits  # sharing suits with AA slightly helps KK less


def test_hand_vs_range_equity_matches_pairwise_for_single_hand_range(ev):
    aa, kk = cards.canon_index("AA"), cards.canon_index("KK")
    eq_aa_kk, _ = equity.equity_pair_exact(aa, kk, ev)
    opp_range = np.zeros(cards.NUM_CANON_HANDS)
    opp_range[kk] = 1.0
    result = equity.hand_vs_range_equity(aa, opp_range, _tiny_table(aa, kk, eq_aa_kk))
    assert result == pytest.approx(eq_aa_kk)


def test_hand_vs_range_equity_vectorized_matches_scalar_version(ev):
    aa, kk, qq = cards.canon_index("AA"), cards.canon_index("KK"), cards.canon_index("QQ")
    eq_aa_kk, _ = equity.equity_pair_exact(aa, kk, ev)
    eq_qq_kk, _ = equity.equity_pair_exact(qq, kk, ev)
    table = np.full((cards.NUM_CANON_HANDS, cards.NUM_CANON_HANDS), 0.5)
    table[aa, kk] = eq_aa_kk
    table[qq, kk] = eq_qq_kk
    opp_range = np.zeros(cards.NUM_CANON_HANDS)
    opp_range[kk] = 1.0
    vec = equity.hand_vs_range_equity_vectorized(opp_range, table)
    assert vec[aa] == pytest.approx(eq_aa_kk)
    assert vec[qq] == pytest.approx(eq_qq_kk)


def test_multiway_equity_mc_symmetric_ranges_average_to_roughly_one_third_each(ev):
    # Two identical, uniform (all-hands, full-weight) opponent ranges make a fully
    # symmetric 3-way pot (hero + 2 opponents) -- across all 169 hero hands, the
    # combo-weighted average equity must be close to 1/3 (no hand is systematically
    # favored or disfavored by the sampling itself). Generous tolerance: this is MC,
    # not exact enumeration.
    uniform_range = np.ones(cards.NUM_CANON_HANDS)
    eq = equity.multiway_equity_mc(
        [uniform_range, uniform_range], ev, 400, rng=np.random.default_rng(0)
    )
    weights = np.asarray(cards.CANON_COMBO_WEIGHT, dtype=np.float64)
    weighted_avg = np.dot(eq, weights) / weights.sum()
    assert weighted_avg == pytest.approx(1.0 / 3.0, abs=0.03)


def test_multiway_equity_mc_strong_hand_favored_heads_up_still_favored_three_way(ev):
    # AA is the strongest starting hand heads-up; against two uniform opponent
    # ranges it should still be favored well above the symmetric 1/3 baseline --
    # a directional sanity check, not an exact combinatorial target.
    aa = cards.canon_index("AA")
    uniform_range = np.ones(cards.NUM_CANON_HANDS)
    eq = equity.multiway_equity_mc(
        [uniform_range, uniform_range], ev, 400, rng=np.random.default_rng(1)
    )
    assert eq[aa] > 0.45  # comfortably above the symmetric two-opponent baseline of 1/3


def test_multiway_equity_mc_zero_valid_sample_fallback(monkeypatch, ev):
    # Force every iteration's opponent-combo sampling to fail (simulating a range so
    # heavily blocked that no collision-free combo could ever be found), regardless
    # of real card-removal probabilities -- this deterministically exercises the
    # documented "hero hand never got a valid sample -> return 0.5, warn" fallback,
    # rather than relying on a probabilistic setup that might occasionally succeed.
    monkeypatch.setattr(equity, "_sample_combo_from_range", lambda *a, **k: None)
    uniform_range = np.ones(cards.NUM_CANON_HANDS)
    eq = equity.multiway_equity_mc([uniform_range], ev, 5, rng=np.random.default_rng(3))
    assert np.all(eq == 0.5)


def _tiny_table(hand_idx, opp_idx, eq_value):
    table = np.full((cards.NUM_CANON_HANDS, cards.NUM_CANON_HANDS), 0.5)
    table[hand_idx, opp_idx] = eq_value
    return table
