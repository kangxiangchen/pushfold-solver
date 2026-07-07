#!/usr/bin/env python3
"""Session tracker CLI: record account-level poker sessions, report results.

Workflow (live session):
    python scripts/session.py start          # prompts for the numbers visible NOW
    ... play ...
    python scripts/session.py end            # prompts for the end-of-session numbers

Or enter a finished session in one shot (scriptable, no prompts):
    python scripts/session.py add --start-time "2026/07/07 20:00:00" ... (see --help)

Reporting:
    python scripts/session.py report [--curve] [--hand-history data/export.txt]

What the numbers mean (see src/sessions.py for the full rationale):
  - main balance and the SEPARATE claimable rakeback balance are both recorded at
    each boundary; that split is what separates table winnings from rakeback earned.
  - the optional mission/leaderboard "rake paid" counter turns each session into a
    measurement of the effective rakeback rate rho.
  - sessions are account-level (multitabling mixes stakes); per-stake results come
    from --hand-history, which joins each session's time window against the export.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hand_history
import sessions
from sessions import Session

DEFAULT_CSV = "data/sessions.csv"
DEFAULT_STUB = "data/session_open.json"
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"


def _now() -> str:
    return datetime.now().strftime(TIME_FORMAT)


def _prompt_float(label: str, *, optional: bool = False, default: float | None = None) -> float | None:
    suffix = " (enter to skip)" if optional else (f" [{default}]" if default is not None else "")
    while True:
        raw = input(f"  {label}{suffix}: ").strip().replace(",", "")
        if not raw:
            if optional:
                return None
            if default is not None:
                return default
            print("    required, please enter a number")
            continue
        try:
            return float(raw)
        except ValueError:
            print(f"    not a number: {raw!r}")


def cmd_start(args) -> None:
    print("Session start -- read these off the client now:")
    stub = {
        "start_time": _now(),
        "start_balance": _prompt_float("main balance"),
        "rakeback_balance_start": _prompt_float("rakeback balance"),
        "rake_paid_start": _prompt_float("rake-paid counter (missions/leaderboard)", optional=True),
        "stakes_note": input("  stakes you plan to play (free text, optional): ").strip(),
    }
    sessions.stage_open_session(args.stub, stub)
    print(f"session started at {stub['start_time']} -- finish it with 'end'")


def cmd_end(args) -> None:
    stub = sessions.load_open_session(args.stub)
    if stub is None:
        raise SystemExit(f"no open session staged at {args.stub} -- use 'start' first (or 'add')")
    print(f"Ending session started {stub['start_time']} -- read these off the client now:")
    end_values = {
        "end_time": _now(),
        "end_balance": _prompt_float("main balance"),
        "rakeback_balance_end": _prompt_float("rakeback balance"),
        "rakeback_claimed": _prompt_float("rakeback claimed DURING the session", default=0.0),
        "deposits": _prompt_float("deposits during the session", default=0.0),
        "withdrawals": _prompt_float("withdrawals during the session", default=0.0),
    }
    if stub.get("rake_paid_start") is not None:
        end_values["rake_paid_end"] = _prompt_float("rake-paid counter")
    notes = input("  notes (optional): ").strip()

    try:
        session = Session(
            start_time=stub["start_time"],
            start_balance=stub["start_balance"],
            rakeback_balance_start=stub["rakeback_balance_start"],
            rake_paid_start=stub.get("rake_paid_start"),
            stakes_note=stub.get("stakes_note", ""),
            notes=notes,
            **end_values,
        )
    except ValueError as exc:
        # The staged stub is left in place so 'end' can simply be rerun.
        raise SystemExit(f"session not recorded: {exc}") from exc
    sessions.append_session(args.csv, session)
    sessions.clear_open_session(args.stub)
    _print_session_line(session)
    print(f"recorded to {args.csv}")


def cmd_add(args) -> None:
    try:
        session = Session(
            start_time=args.start_time,
            end_time=args.end_time,
            start_balance=args.start_balance,
            end_balance=args.end_balance,
            rakeback_balance_start=args.rakeback_start,
            rakeback_balance_end=args.rakeback_end,
            rakeback_claimed=args.claimed,
            deposits=args.deposits,
            withdrawals=args.withdrawals,
            rake_paid_start=args.rake_paid_start,
            rake_paid_end=args.rake_paid_end,
            hands_played=args.hands,
            stakes_note=args.stakes or "",
            notes=args.notes or "",
        )
    except ValueError as exc:
        raise SystemExit(f"session not recorded: {exc}") from exc
    sessions.append_session(args.csv, session)
    _print_session_line(session)
    print(f"recorded to {args.csv}")


def _print_session_line(s: Session) -> None:
    rho = sessions.effective_rakeback_rate(s)
    rho_str = f"  rho={rho:.2f}" if rho is not None else ""
    print(
        f"{s.start_time} -> {s.end_time}  "
        f"table {sessions.table_result(s):+.2f}  "
        f"rakeback {sessions.rakeback_earned(s):+.2f}  "
        f"total {sessions.total_result(s):+.2f}{rho_str}"
        f"{'  [' + s.stakes_note + ']' if s.stakes_note else ''}"
    )


def cmd_report(args) -> None:
    loaded, failures = sessions.load_sessions(args.csv)
    for row_num, message in failures:
        print(f"WARNING: sessions.csv row {row_num} unreadable: {message}", file=sys.stderr)
    if not loaded:
        raise SystemExit(f"no sessions recorded yet in {args.csv}")

    print(f"{len(loaded)} sessions:")
    for s in loaded:
        _print_session_line(s)

    summary = sessions.summarize(loaded)
    print(f"\ntotals:  table {summary.total_table_result:+.2f}   "
          f"rakeback {summary.total_rakeback_earned:+.2f}   "
          f"combined {summary.total_result:+.2f}")
    if summary.std_session_result is not None:
        print(f"per session: mean {summary.mean_session_result:+.2f}  "
              f"std {summary.std_session_result:.2f}  (n={summary.n_sessions})")
    if summary.rho_mean is not None:
        se_str = f" +/- {summary.rho_se:.3f} (1 SE)" if summary.rho_se is not None else ""
        print(f"effective rakeback rate rho: {summary.rho_mean:.3f}{se_str} "
              f"over {summary.rho_sessions} counter-tracked session(s)")

    if args.hand_history:
        _report_per_stake(loaded, args.hand_history)

    if args.curve:
        out = _save_curve(loaded, args.out_dir)
        print(f"bankroll curve saved to {out}")


def _report_per_stake(loaded: list[Session], hh_path: str) -> None:
    hands, hh_failures = hand_history.parse_file(hh_path)
    if hh_failures:
        print(f"WARNING: {len(hh_failures)} hands failed to parse in {hh_path}", file=sys.stderr)

    print("\nper-stake (from hand-history join):")
    for s in loaded:
        window = sessions.hands_in_window(hands, s.start_time, s.end_time)
        if not window:
            print(f"  {s.start_time}: no hands in export for this window")
            continue
        stats = sessions.per_stake_stats(window)
        hh_net = sum(st.net_currency for st in stats.values())
        residual = sessions.table_result(s) - hh_net
        print(f"  {s.start_time}: {len(window)} hands")
        for bb in sorted(stats):
            st = stats[bb]
            bb100 = f"{st.bb_per_100:+.1f} bb/100" if st.bb_per_100 is not None else "n/a"
            excluded = f" ({st.hands - st.hands_with_net} postflop hands excluded from net)" \
                if st.hands_with_net < st.hands else ""
            print(f"    {bb:g}bb stake: {st.hands} hands  net {st.net_currency:+.2f} "
                  f"({st.net_bb:+.1f}bb, {bb100}){excluded}")
        print(f"    cross-check: balance table_result {sessions.table_result(s):+.2f} "
              f"vs HH net {hh_net:+.2f}  (residual {residual:+.2f})")


def _save_curve(loaded: list[Session], out_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(loaded, key=lambda s: hand_history.parse_timestamp(s.start_time))
    cumulative_total, cumulative_table = [], []
    running_total = running_table = 0.0
    for s in ordered:
        running_total += sessions.total_result(s)
        running_table += sessions.table_result(s)
        cumulative_total.append(running_total)
        cumulative_table.append(running_table)

    x = range(1, len(ordered) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, cumulative_total, marker="o", label="total (table + rakeback)")
    ax.plot(x, cumulative_table, marker="o", linestyle="--", label="table only")
    ax.axhline(0.0, linewidth=0.8, color="gray")
    ax.set_xlabel("session #")
    ax.set_ylabel("cumulative result (currency)")
    ax.set_title("Bankroll curve by session")
    ax.legend()
    fig.tight_layout()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / "session_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--stub", default=DEFAULT_STUB)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("start", help="stage a live session (prompts for start-of-session numbers)")
    sub.add_parser("end", help="complete the staged session (prompts for end-of-session numbers)")

    p_add = sub.add_parser("add", help="record a finished session in one non-interactive shot")
    p_add.add_argument("--start-time", required=True, help='e.g. "2026/07/07 20:00:00"')
    p_add.add_argument("--end-time", required=True)
    p_add.add_argument("--start-balance", type=float, required=True)
    p_add.add_argument("--end-balance", type=float, required=True)
    p_add.add_argument("--rakeback-start", type=float, required=True)
    p_add.add_argument("--rakeback-end", type=float, required=True)
    p_add.add_argument("--claimed", type=float, default=0.0)
    p_add.add_argument("--deposits", type=float, default=0.0)
    p_add.add_argument("--withdrawals", type=float, default=0.0)
    p_add.add_argument("--rake-paid-start", type=float, default=None)
    p_add.add_argument("--rake-paid-end", type=float, default=None)
    p_add.add_argument("--hands", type=int, default=None)
    p_add.add_argument("--stakes", default=None)
    p_add.add_argument("--notes", default=None)

    p_report = sub.add_parser("report", help="print sessions, totals, rho estimate")
    p_report.add_argument("--curve", action="store_true", help="save cumulative bankroll PNG")
    p_report.add_argument("--hand-history", default=None,
                          help="HH export to join for per-stake stats and the net cross-check")
    p_report.add_argument("--out-dir", default="cache/")

    args = parser.parse_args()
    {"start": cmd_start, "end": cmd_end, "add": cmd_add, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
