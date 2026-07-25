import sys
from pathlib import Path

import pytest

# scripts/ first, then src/ on top: src/ must win name collisions (scripts/backtest.py
# shadows src/backtest.py and would self-import if it resolved first).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import population
import sessions
from backtest import SolveCache
from export_session_viewer import _population_payload, build_payload, build_population_payload, render_html
from hand_history import parse_hand
from session_analysis import GradingContext, grade_hands
from test_hand_history import SAMPLE_2_GRADABLE_BB_CALL
from test_sessions import make_session
from test_solver import SYNTHETIC_TABLE


def test_build_payload_empty_ledger(tmp_path):
    payload = build_payload(str(tmp_path / "none.csv"))
    assert payload["summary"]["n"] == 0
    assert payload["sessions"] == []
    assert payload["hasHH"] is False


def test_build_payload_without_graded_has_no_grades(tmp_path):
    path = str(tmp_path / "sessions.csv")
    sessions.append_session(path, make_session())
    payload = build_payload(path)
    assert payload["hasGrades"] is False
    assert payload["sessions"][0]["analysis"] is None


def test_build_payload_analysis_block_shape(tmp_path):
    path = str(tmp_path / "sessions.csv")
    sessions.append_session(path, make_session(
        start_time="2026/07/04 18:00:00", end_time="2026/07/04 19:00:00"))
    sessions.append_session(path, make_session(  # no hands in this window
        start_time="2026/07/05 18:00:00", end_time="2026/07/05 19:00:00"))

    hand = parse_hand(SAMPLE_2_GRADABLE_BB_CALL)  # gradable BB call @8bb, ₮1/₮2
    graded, _skips = grade_hands([hand], GradingContext(SolveCache(SYNTHETIC_TABLE), None, None))
    payload = build_payload(path, hands=[hand], graded=graded)

    assert payload["hasGrades"] is True
    analysis = payload["sessions"][0]["analysis"]
    assert analysis["spots"] == 1
    assert analysis["huSpots"] == 1
    assert analysis["fourmaxSpots"] == 0
    assert analysis["luckSpots"] == 1
    assert 0.0 <= analysis["accuracyPct"] <= 100.0
    assert analysis["gradedNetCurrency"] == pytest.approx(16.56)
    assert analysis["luckGapCurrency"] == pytest.approx(
        analysis["realizedCurrency"] - analysis["predictedCurrency"], abs=0.02)
    stake = analysis["stakes"][0]
    assert stake["bb"] == 2.0
    assert stake["spots"] == 1
    assert {"matched", "evLostBb", "luckSpots", "predictedBb", "realizedBb"} <= set(stake)
    for m in analysis["mistakes"]:
        assert {"hand", "pos", "stackBb", "hero", "gto", "weight", "lossBb", "bb", "model", "time"} <= set(m)

    assert payload["sessions"][1]["analysis"] is None

    html = render_html(payload)
    assert "Push/fold analysis (GTO grade & luck)" in html
    assert '"hasGrades":true' in html


# ---------------------------------------------------------------------------------
# Population card payload
# ---------------------------------------------------------------------------------

FLAT_RANGES = {key: np.full(169, 0.5) for key in population.INFOSET_ORDER}


def test_population_payload_block_shape_and_render(tmp_path):
    path = str(tmp_path / "sessions.csv")
    sessions.append_session(path, make_session())

    hand = parse_hand(SAMPLE_2_GRADABLE_BB_CALL)  # villain BTN go (shows 6h Ad), SB fold
    pp = build_population_payload([hand], FLAT_RANGES)

    assert pp["decisions"] == 2
    assert pp["hands"] == 1
    assert pp["hasGto"] is True
    assert len(pp["rows"]) == 14
    rows = {r["key"]: r for r in pp["rows"]}
    btn = rows["BTN|"]
    assert btn["desc"]  # plain-English situation line present
    assert btn["gtoPct"] == pytest.approx(50.0)
    assert btn["cells"][0]["bb"] is None and btn["cells"][0]["n"] == 1  # all-stakes cell first
    assert btn["cells"][1]["bb"] == 2.0
    assert btn["shown"]["n"] == 1  # the revealed BTN shove joins the composition

    payload = build_payload(path, population=pp)
    assert payload["hasPopulation"] is True
    html = render_html(payload)
    assert "Population vs GTO" in html
    assert '"hasPopulation":true' in html


def test_population_absent_by_default(tmp_path):
    path = str(tmp_path / "sessions.csv")
    sessions.append_session(path, make_session())
    assert build_payload(path)["hasPopulation"] is False


def test_population_degrades_without_ranges(tmp_path):
    hand = parse_hand(SAMPLE_2_GRADABLE_BB_CALL)
    pp = build_population_payload([hand], None)
    assert pp["hasGto"] is False
    assert all(r["gtoPct"] is None and r["shown"] is None for r in pp["rows"])

    path = str(tmp_path / "sessions.csv")
    sessions.append_session(path, make_session())
    html = render_html(build_payload(path, population=pp))
    assert '"hasGto":false' in html


def test_build_payload_cumulative_series_and_summary(tmp_path):
    path = str(tmp_path / "sessions.csv")
    # out-of-order append: payload must sort chronologically before accumulating
    sessions.append_session(path, make_session(
        start_time="2026/07/02 20:00:00", end_time="2026/07/02 22:00:00",
        start_balance=560.0, end_balance=540.0,  # table -20
        rakeback_balance_start=14.0, rakeback_balance_end=18.0,  # +4
        rake_paid_start=100.0, rake_paid_end=110.0,
    ))
    sessions.append_session(path, make_session())  # 07/01: table +60, rakeback +4

    payload = build_payload(path)
    rows = payload["sessions"]
    assert [r["start"] for r in rows] == ["2026/07/01 20:00:00", "2026/07/02 20:00:00"]
    assert rows[0]["cumTotal"] == pytest.approx(64.0)
    assert rows[1]["cumTotal"] == pytest.approx(48.0)
    assert rows[1]["cumTable"] == pytest.approx(40.0)
    assert rows[0]["rho"] is None
    assert rows[1]["rho"] == pytest.approx(0.4)
    assert payload["summary"]["total"] == pytest.approx(48.0)
    assert payload["summary"]["rhoMean"] == pytest.approx(0.4)
