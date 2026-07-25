import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# scripts/ first, then src/ on top: src/ must win name collisions (scripts/backtest.py
# shadows src/backtest.py and would self-import if it resolved first).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sessions
from backtest import SolveCache
from session_analysis import GradingContext
from session_server import make_server
from test_hand_history import SAMPLE_2_GRADABLE_BB_CALL
from test_solver import SYNTHETIC_TABLE


def walk_hand(hand_id: int, date: str, time: str) -> str:
    """One parseable walk hand in the real CoinPoker export shape.
    Hero (BB) nets 1.56 + 1 - 2 = +0.56 currency."""
    return f"""CoinPoker Hand #{hand_id}: NLH (₮1/₮2) {date} {time} +08
Table '100169' 4-max Seat #3 is the button
Seat 2: Hero (₮16 in chips)
Seat 4: villain (₮16 in chips)
villain: posts small blind ₮1
Hero: posts big blind ₮2
*** HOLE CARDS ***
Dealt to Hero [5h 3h]
Dealt to villain
villain: folds
Hero: RETURN ₮1
*** SHOWDOWN ***
Hero collected ₮1.56 from pot
*** SUMMARY ***
Total pot ₮2 | Rake ₮0.24 | Splash Fee ₮0.20
"""


@pytest.fixture()
def server(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # two sessions separated by a 3h gap: (20:00, 20:02) and (23:00)
    export = walk_hand(1, "2026/07/01", "20:00:00") + walk_hand(2, "2026/07/01", "20:02:00") \
        + walk_hand(3, "2026/07/01", "23:00:00")
    (data_dir / "export.txt").write_text(export, encoding="utf-8")

    csv_path = str(tmp_path / "sessions.csv")
    # grading=False keeps these endpoint tests hermetic (no real cache/ artifacts);
    # the graded dashboard path has its own fixture below with an injected context.
    srv = make_server(("127.0.0.1", 0), csv_path=csv_path, data_dir=str(data_dir), grading=False)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield {"base": base, "csv": csv_path, "data_dir": data_dir}
    srv.shutdown()
    srv.server_close()


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def post(url: str, body: bytes, headers: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def post_json(url: str, obj: dict) -> tuple[int, dict]:
    return post(url, json.dumps(obj).encode(), {"Content-Type": "application/json"})


def test_get_root_serves_dashboard_with_form_enabled(server):
    with urllib.request.urlopen(server["base"] + "/") as resp:
        html = resp.read().decode()
    assert resp.status == 200
    assert "Session tracker" in html
    assert '"served":true' in html


def test_detect_proposes_gap_clustered_unlogged_sessions(server):
    data = get_json(server["base"] + "/api/detect")
    proposals = data["proposals"]
    assert len(proposals) == 2
    first = proposals[0]
    assert first["start"] == "2026/07/01 20:00:00"
    assert first["end"] == "2026/07/01 20:02:00"
    assert first["hands"] == 2
    assert first["hhNet"] == pytest.approx(1.12)
    assert first["stakes"] == "2bb"
    assert proposals[1]["hands"] == 1


def test_log_session_appends_row_and_hides_proposal(server):
    proposal = get_json(server["base"] + "/api/detect")["proposals"][0]
    status, data = post_json(server["base"] + "/api/log", {
        "start_time": proposal["start"],
        "end_time": proposal["end"],
        "rakeback_balance_start": 10.0,
        "rakeback_balance_end": 12.0,
        "hh_table_result": proposal["hhNet"],
        "hands_played": proposal["hands"],
        "stakes_note": proposal["stakes"],
    })
    assert status == 200 and data.get("ok"), data

    loaded, failures = sessions.load_sessions(server["csv"])
    assert failures == []
    assert len(loaded) == 1
    assert sessions.table_result(loaded[0]) == pytest.approx(1.12)
    assert sessions.rakeback_earned(loaded[0]) == pytest.approx(2.0)

    remaining = get_json(server["base"] + "/api/detect")["proposals"]
    assert len(remaining) == 1
    assert remaining[0]["start"] == "2026/07/01 23:00:00"


def test_log_validation_error_is_a_clean_message_not_a_500(server):
    status, data = post_json(server["base"] + "/api/log", {
        "start_time": "2026/07/01 23:00:00",
        "end_time": "2026/07/01 20:00:00",  # backwards
        "hh_table_result": 1.0,
    })
    assert status == 400
    assert "end_time" in data["error"]
    loaded, _ = sessions.load_sessions(server["csv"])
    assert loaded == []


def test_upload_export_joins_the_scan(server):
    new_hand = walk_hand(9, "2026/07/02", "19:00:00")
    status, data = post(server["base"] + "/api/upload", new_hand.encode(),
                        {"X-Filename": "fresh_export.txt"})
    assert status == 200 and data.get("ok"), data

    proposals = get_json(server["base"] + "/api/detect")["proposals"]
    assert any(p["start"] == "2026/07/02 19:00:00" for p in proposals)
    uploads = list((server["data_dir"] / "uploads").glob("*.txt"))
    assert len(uploads) == 1


def test_upload_rejects_unparseable_content(server):
    status, data = post(server["base"] + "/api/upload", b"not a hand history",
                        {"X-Filename": "junk.txt"})
    assert status == 400
    assert "error" in data


def test_grading_disabled_still_serves_dashboard(server):
    with urllib.request.urlopen(server["base"] + "/") as resp:
        html = resp.read().decode()
    assert resp.status == 200
    assert '"hasGrades":false' in html


# ---------------------------------------------------------------------------------
# Graded dashboard: injected GradingContext (synthetic equity table), tmp cache dir.
# ---------------------------------------------------------------------------------


def make_graded_server(tmp_path, grading_ctx):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    # a walk plus a genuine gradable BB-call spot (SAMPLE_2: 2026/07/04 18:01:21, 8bb);
    # written once -- a rewrite would bump mtime and (correctly) invalidate the grades
    export_path = data_dir / "export.txt"
    if not export_path.exists():
        export = walk_hand(1, "2026/07/04", "18:00:30") + SAMPLE_2_GRADABLE_BB_CALL
        export_path.write_text(export, encoding="utf-8")
    return make_server(
        ("127.0.0.1", 0),
        csv_path=str(tmp_path / "sessions.csv"),
        data_dir=str(data_dir),
        grading_ctx=grading_ctx,
        cache_dir=str(tmp_path / "cache"),
    )


@pytest.fixture()
def graded_server(tmp_path):
    ctx = GradingContext(SolveCache(SYNTHETIC_TABLE), fourmax=None, evaluator=None)
    srv = make_graded_server(tmp_path, ctx)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield {"base": f"http://127.0.0.1:{srv.server_address[1]}", "tmp": tmp_path, "srv": srv}
    srv.shutdown()
    srv.server_close()


def test_dashboard_serves_grade_cards_with_injected_ctx(graded_server):
    status, data = post_json(graded_server["base"] + "/api/log", {
        "start_time": "2026/07/04 18:00:00",
        "end_time": "2026/07/04 19:00:00",
        "hh_table_result": 17.12,
    })
    assert status == 200 and data.get("ok"), data

    with urllib.request.urlopen(graded_server["base"] + "/") as resp:
        html = resp.read().decode()
    assert '"hasGrades":true' in html
    assert '"spots":1' in html  # SAMPLE_2 graded; the walk has no gradable decision
    assert (graded_server["tmp"] / "cache" / "session_grades" / "export.json").exists()


def test_dashboard_serves_population_card_degraded_without_ranges(graded_server):
    # graded_server's injected ctx has fourmax=None and its cache dir has no pickle:
    # the population card must still render, GTO-less.
    with urllib.request.urlopen(graded_server["base"] + "/") as resp:
        html = resp.read().decode()
    assert '"hasPopulation":true' in html
    assert '"hasGto":false' in html
    assert '"decisions":2' in html  # SAMPLE_2: villain BTN go + SB fold (walk hand adds none)


def test_population_gto_from_injected_fourmax_ranges(tmp_path):
    import numpy as np

    from game import FOURMAX_CONFIG
    from population import INFOSET_ORDER

    flat = {key: np.full(169, 0.5) for key in INFOSET_ORDER}
    ctx = GradingContext(SolveCache(SYNTHETIC_TABLE), fourmax=(FOURMAX_CONFIG, flat), evaluator=None)
    srv = make_graded_server(tmp_path, ctx)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/") as resp:
            html = resp.read().decode()
    finally:
        srv.shutdown()
        srv.server_close()
    assert '"hasGto":true' in html
    assert '"gtoPct":50.0' in html


def test_scan_graded_uses_persisted_cache_across_instances(graded_server):
    rows1 = graded_server["srv"].scan_graded()
    assert len(rows1) == 1

    # A second server over the same dirs with a POISONED context (any grading
    # attempt would crash on solve_cache=None) must serve the persisted grades.
    poisoned = GradingContext(solve_cache=None, fourmax=None, evaluator=None)
    srv2 = make_graded_server(graded_server["tmp"], poisoned)
    try:
        rows2 = srv2.scan_graded()
    finally:
        srv2.server_close()
    assert rows2 == rows1
