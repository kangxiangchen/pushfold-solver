import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from game import GameConfig, HU_CONFIG, FOURMAX_CONFIG, NINEMAX_CONFIG


# ---------------------------------------------------------------------------------
# Percentage-with-cap rake model (the 9-max game: 2.5% of pot capped at 0.5bb).
# ---------------------------------------------------------------------------------


def test_drop_for_pot_is_flat_percentage_below_the_cap():
    # A heads-up all-in at 5bb makes a ~10bb pot: 2.5% = 0.25bb, well under the 0.5 cap.
    assert NINEMAX_CONFIG.drop_for_pot(10.0, contested=True) == pytest.approx(0.25)


def test_drop_for_pot_binds_the_cap_on_large_multiway_pots():
    # 2.5% of 30bb would be 0.75bb, but the cap clips it to 0.5bb (a 4+-way all-in).
    assert NINEMAX_CONFIG.drop_for_pot(30.0, contested=True) == pytest.approx(0.5)
    # The cap first binds exactly at a 20bb pot (2.5% * 20 = 0.5).
    assert NINEMAX_CONFIG.drop_for_pot(20.0, contested=True) == pytest.approx(0.5)


def test_drop_for_pot_is_zero_on_uncontested_pots_for_this_game():
    # rake_uncontested is False: a walk/steal (no showdown) pays no rake at all.
    assert NINEMAX_CONFIG.drop_for_pot(1.5, contested=False) == 0.0


def test_rakeback_pool_is_eighty_percent_of_the_rake_charged():
    # 80% rakeback on a contested 10bb pot: 0.8 * 0.25 = 0.20bb pool.
    assert NINEMAX_CONFIG.rakeback_pool_for_pot(10.0, contested=True) == pytest.approx(0.20)
    # ... and nothing on an uncontested pot, since no rake was charged.
    assert NINEMAX_CONFIG.rakeback_pool_for_pot(1.5, contested=False) == 0.0


# ---------------------------------------------------------------------------------
# Backward compatibility: the flat rake_bb/splash model (HU, 4-max) is unchanged --
# the new pot-dependent methods must return the old constants regardless of pot size.
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("pot", [1.0, 10.0, 25.0])
@pytest.mark.parametrize("contested", [True, False])
def test_flat_model_drop_is_pot_independent_and_matches_drop_bb(pot, contested):
    # rake_pct == 0 (the default) and rake_uncontested defaults True, so both HU and
    # 4-max keep their flat 0.22bb drop and 0.06bb rakeback pool for any pot/contest.
    for cfg in (HU_CONFIG, FOURMAX_CONFIG):
        assert cfg.drop_for_pot(pot, contested=contested) == pytest.approx(cfg.drop_bb)
        assert cfg.rakeback_pool_for_pot(pot, contested=contested) == pytest.approx(
            cfg.rakeback_pool_bb
        )


# ---------------------------------------------------------------------------------
# NINEMAX_CONFIG shape and the rake-free baseline.
# ---------------------------------------------------------------------------------


def test_ninemax_config_shape():
    assert NINEMAX_CONFIG.seats == ("UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB")
    assert NINEMAX_CONFIG.stack_bb == 5.0
    assert NINEMAX_CONFIG.rakeback_frac == 0.8
    assert NINEMAX_CONFIG.rake_uncontested is False
    assert NINEMAX_CONFIG.blind_of("SB") == 0.5 and NINEMAX_CONFIG.blind_of("BB") == 1.0
    assert NINEMAX_CONFIG.blind_of("UTG") == 0.0  # no blind posted


def test_rake_free_zeroes_the_percentage_rake_too():
    rf = NINEMAX_CONFIG.rake_free()
    assert rf.rake_pct == 0.0
    assert rf.drop_for_pot(10.0, contested=True) == 0.0
    assert rf.rakeback_pool_for_pot(10.0, contested=True) == 0.0


def test_negative_rake_pct_is_rejected():
    with pytest.raises(ValueError):
        GameConfig(seats=("SB", "BB"), rake_pct=-0.01)


def test_negative_rake_cap_is_rejected():
    with pytest.raises(ValueError):
        GameConfig(seats=("SB", "BB"), rake_cap_bb=-1.0)
