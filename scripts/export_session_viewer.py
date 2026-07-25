#!/usr/bin/env python3
"""Export the session ledger as a single self-contained HTML dashboard.

Companion to export_range_viewer.py, same pattern: build a JSON payload from the
tested src/ logic (src/sessions.py here), inject it into one offline HTML file
(no CDNs, no fetches), write it under cache/. The CLI report (scripts/session.py
report) stays the quick terminal view; this is the visual one.

Page anatomy: stat tiles (table result / rakeback earned / combined / effective
rakeback rate rho), a cumulative bankroll line chart (total vs table-only, with
crosshair + tooltip), the full sessions table (which doubles as the chart's
accessible table view), and -- when --hand-history is given -- a per-stake
breakdown card per session with the balance-vs-HH cross-check.

Colors are the dataviz reference palette's validated steps (blue/aqua series
pair re-validated for this page: light worst-adjacent CVD dE 73.6, dark 69.8,
all lightness/chroma checks pass; aqua's light-mode 2.74:1 surface contrast is
covered by the relief rule -- both lines carry direct end labels and the table
view). Dark mode is its own selected set, not an automatic flip.

Rerun after recording sessions: python scripts/export_session_viewer.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hand_history
import population
import session_analysis
import sessions as sessions_mod
import tree
from game import FOURMAX_CONFIG
from session_viewer_template import HTML_TEMPLATE

_INFOSET_DESCRIPTIONS = {
    tree.infoset_key(i.seat, i.shoved_before): tree.infoset_description(FOURMAX_CONFIG, i)
    for i in tree.build_infosets(FOURMAX_CONFIG)
}


def _population_cell(bb: float | None, stats, gto_freq: float | None) -> dict:
    """One (stake, info set) cell; `stats` is InfosetPoolStats or StakePoolStats
    (same n/go/pool_freq/se surface). z is per-cell so the stake dropdown stays honest."""
    p, se = stats.pool_freq, stats.se
    z = (p - gto_freq) / se if (gto_freq is not None and p is not None and se is not None) else None
    return {
        "bb": bb,
        "n": stats.n,
        "goPct": round(100 * p, 1) if p is not None else None,
        "sePct": round(100 * se, 1) if se is not None else None,
        "z": round(z, 1) if z is not None else None,
    }


def _population_payload(deviations, compositions, decisions_n: int, hands_n: int) -> dict:
    """population.compare_to_gto/shown_composition results -> the JSON block the
    'Population vs GTO' card renders from. gtoPct/shown are null when the solved
    ranges are unavailable (pool stats still render)."""
    comp_by_key = {c.key: c for c in compositions}
    rows = []
    for d in deviations:
        shown = None
        c = comp_by_key.get(d.stats.key)
        if c is not None:
            shown = {
                "n": c.shown_go,
                "inGtoPct": round(100 * c.in_gto / c.shown_go, 1),
                "meanWeight": round(c.mean_gto_weight, 2) if c.mean_gto_weight is not None else None,
                "top": [[label, count] for label, count in c.top_out_of_range],
            }
        rows.append({
            "key": d.stats.key,
            "desc": _INFOSET_DESCRIPTIONS.get(d.stats.key, ""),
            "gtoPct": round(100 * d.gto_freq, 1) if d.gto_freq is not None else None,
            "cells": [_population_cell(None, d.stats, d.gto_freq)]
            + [_population_cell(st.bb_amount, st, d.gto_freq) for st in d.stats.per_stake],
            "shown": shown,
        })
    return {
        "decisions": decisions_n,
        "hands": hands_n,
        "hasGto": any(d.gto_freq is not None for d in deviations),
        "rows": rows,
    }


def build_population_payload(hands: list, ranges) -> dict:
    """Census over pre-parsed hands (deduped by hand_id inside population.census)
    -> the dashboard's population block. `ranges` is the solved 4-max RangeMap or
    None (load-only callers pass session_analysis.load_fourmax_ranges_if_cached)."""
    decisions = population.census(hands)
    stats = population.summarize(decisions)
    deviations = population.compare_to_gto(stats, ranges)
    compositions = population.shown_composition(decisions, ranges) if ranges else []
    hands_n = len({h.hand_id for h in hands})
    return _population_payload(deviations, compositions, len(decisions), hands_n)


def _analysis_payload(wa: session_analysis.WindowAnalysis) -> dict:
    """WindowAnalysis -> the JSON block one session's grade & luck card renders from."""
    return {
        "spots": wa.spots,
        "huSpots": wa.hu_spots,
        "fourmaxSpots": wa.fourmax_spots,
        "matched": wa.matched,
        "accuracyPct": round(100.0 * wa.matched / wa.spots, 1),
        "evLostCurrency": round(wa.ev_lost_currency, 2),
        "predictedCurrency": round(wa.predicted_currency, 2),
        "realizedCurrency": round(wa.realized_currency, 2),
        "luckGapCurrency": round(wa.luck_gap_currency, 2),
        "gradedNetCurrency": round(wa.graded_net_currency, 2),
        "luckSpots": sum(st.luck_spots for st in wa.stakes),
        "stakes": [
            {
                "bb": st.bb_amount,
                "spots": st.spots,
                "matched": st.matched,
                "evLostBb": round(st.ev_lost_bb, 2),
                "luckSpots": st.luck_spots,
                "predictedBb": round(st.predicted_ev_bb, 2),
                "realizedBb": round(st.realized_ev_bb, 2),
            }
            for st in wa.stakes
        ],
        "mistakes": [
            {
                "hand": m.hero_hand,
                "pos": m.position,
                "stackBb": round(m.effective_stack_bb, 1),
                "hero": m.hero_action,
                "gto": m.gto_action,
                "weight": round(m.gto_weight, 2),
                "lossBb": round(m.ev_loss_bb, 3),
                "bb": m.bb_amount,
                "model": m.model,
                "time": m.timestamp,
            }
            for m in wa.top_mistakes
        ],
    }


def build_payload(
    csv_path: str,
    hh_path: str | None = None,
    currency: str = "₮",
    hands: list | None = None,
    served: bool = False,
    graded: list | None = None,
    population: dict | None = None,
) -> dict:
    """`hands` (pre-parsed, e.g. the logging server's multi-export scan) takes
    precedence over `hh_path`; `served=True` reveals the in-page log form; `graded`
    (session_analysis.GradedHand rows over the same exports) adds each session's
    GTO grade & luck card; `population` (build_population_payload's block) adds the
    pool-vs-GTO card."""
    loaded, failures = sessions_mod.load_sessions(csv_path)
    ordered = sorted(loaded, key=lambda s: hand_history.parse_timestamp(s.start_time))
    summary = sessions_mod.summarize(ordered)

    if hands is None and hh_path is not None:
        hands, _hh_failures = hand_history.parse_file(hh_path)

    session_rows = []
    running_total = running_table = 0.0
    for s in ordered:
        table = sessions_mod.table_result(s)
        rakeback = sessions_mod.rakeback_earned(s)
        rho = sessions_mod.effective_rakeback_rate(s)
        running_total += sessions_mod.total_result(s)
        running_table += table

        row = {
            "start": s.start_time,
            "end": s.end_time,
            "table": round(table, 2),
            "rakeback": round(rakeback, 2) if rakeback is not None else None,
            "total": round(sessions_mod.total_result(s), 2),
            "cumTotal": round(running_total, 2),
            "cumTable": round(running_table, 2),
            "rho": round(rho, 3) if rho is not None else None,
            "stakes": s.stakes_note,
            "notes": s.notes,
            "perStake": None,
            "analysis": None,
        }
        if graded is not None:
            wa = session_analysis.analyze_window(
                session_analysis.graded_in_window(graded, s.start_time, s.end_time))
            row["analysis"] = _analysis_payload(wa) if wa else None
        if hands is not None:
            window = sessions_mod.hands_in_window(hands, s.start_time, s.end_time)
            if window:
                stats = sessions_mod.per_stake_stats(window)
                hh_net = sum(st.net_currency for st in stats.values())
                row["perStake"] = {
                    "hands": len(window),
                    "hhNet": round(hh_net, 2),
                    # the cross-check only means something when the table result came
                    # from balance readings; HH-sourced sessions ARE the HH net
                    "residual": round(table - hh_net, 2) if s.start_balance is not None else None,
                    "stakes": [
                        {
                            "bb": st.bb_amount,
                            "hands": st.hands,
                            "excluded": st.hands - st.hands_with_net,
                            "net": round(st.net_currency, 2),
                            "netBb": round(st.net_bb, 1),
                            "bbPer100": round(st.bb_per_100, 1) if st.bb_per_100 is not None else None,
                        }
                        for st in sorted(stats.values(), key=lambda x: x.bb_amount)
                    ],
                }
        session_rows.append(row)

    return {
        "currency": currency,
        "sessions": session_rows,
        "summary": {
            "n": summary.n_sessions,
            "table": round(summary.total_table_result, 2),
            "rakeback": round(summary.total_rakeback_earned, 2),
            "total": round(summary.total_result, 2),
            "rhoN": summary.rho_sessions,
            "rhoMean": round(summary.rho_mean, 3) if summary.rho_mean is not None else None,
            "rhoSe": round(summary.rho_se, 3) if summary.rho_se is not None else None,
            "rakebackMissing": summary.rakeback_missing,
        },
        "loadFailures": len(failures),
        "hasHH": hands is not None,
        "hasGrades": graded is not None,
        "hasPopulation": population is not None,
        "population": population,
        "served": served,
    }



def render_html(payload: dict) -> str:
    """Payload -> the full self-contained page. Shared with the logging server
    (scripts/session_server.py), which serves the same template with served=True."""
    # "</" -> "<\/" so user-typed notes can never terminate the script tag early.
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA__", data_json)


def _scan_data_dir_hands(data_dir: str) -> list:
    """All hands under data_dir (top level + uploads/) -- same globs as the logging
    server's scan; the population census is global, not per session window."""
    base = Path(data_dir)
    all_hands: list = []
    for path in sorted(base.glob("*.txt")) + sorted((base / "uploads").glob("*.txt")):
        hands, _failures = hand_history.parse_file(str(path))
        all_hands.extend(hands)
    return all_hands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/sessions.csv")
    parser.add_argument("--hand-history", default=None,
                        help="HH export to join for the per-stake section")
    parser.add_argument("--out", default="cache/session_viewer.html")
    parser.add_argument("--currency", default="₮")
    parser.add_argument("--cache-dir", default="cache/",
                        help="solver artifacts (equity table, 4-max ranges) + grade cache")
    parser.add_argument("--data-dir", default="data/",
                        help="directory whose *.txt exports feed the population census")
    parser.add_argument("--no-grade", action="store_true",
                        help="skip the per-session GTO grade & luck cards")
    parser.add_argument("--no-population", action="store_true",
                        help="skip the population-vs-GTO card")
    args = parser.parse_args()

    hands = graded = None
    if args.hand_history is not None:
        hands, _hh_failures = hand_history.parse_file(args.hand_history)
        if not args.no_grade:
            ctx = session_analysis.make_grading_context(args.cache_dir)
            if ctx is None:
                print("grading skipped: no equity table under "
                      f"{args.cache_dir} (see README's backtesting section)")
            else:
                graded = session_analysis.load_or_grade_file(
                    args.hand_history, hands, ctx, cache_dir=args.cache_dir)

    pop = None
    if not args.no_population:
        pop_hands = _scan_data_dir_hands(args.data_dir)
        if not pop_hands and hands is not None:
            pop_hands = hands  # no exports under data_dir: census the joined file alone
        if pop_hands:
            loaded = session_analysis.load_fourmax_ranges_if_cached(args.cache_dir)
            pop = build_population_payload(pop_hands, loaded[1] if loaded is not None else None)

    payload = build_payload(args.csv, currency=args.currency, hands=hands,
                            graded=graded, population=pop)
    html = render_html(payload)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html) // 1024} KB, {payload['summary']['n']} sessions)")


if __name__ == "__main__":
    main()
