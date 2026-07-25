"""Session tracker: account-level session ledger + per-stake hand-history join.

The owner multitables 2-3 stakes simultaneously, so a session is an ACCOUNT-level
unit: balance deltas cannot be attributed to a single stake. Two layers, kept
strictly apart:

  session level  -- balance-derived truth, in currency only (never bb, which is
      ill-defined across mixed stakes): table_result, rakeback_earned, and the
      effective rakeback rate rho when the client's rake-paid counter was recorded.
  per-stake level -- derived exclusively from the hand-history join
      (hands_in_window + per_stake_stats), which groups the window's hands by
      bb_amount. Overlapping timestamps from multitabling are fine; the window
      filter doesn't care which table a hand came from.

CoinPoker keeps the main balance and a SEPARATE claimable rakeback balance.
Recording both at each session boundary is what splits table winnings from
rakeback earned -- a bare balance delta conflates the two the moment rakeback is
claimed (or money is deposited/withdrawn) mid-session, which is why those fields
are required with explicit 0 defaults rather than optional.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import hand_history
from hand_history import ParsedHand, parse_timestamp


class OpenSessionExistsError(RuntimeError):
    """Raised when `start` is attempted while a staged open session already exists."""


@dataclass(frozen=True)
class Session:
    start_time: str  # "YYYY/MM/DD HH:MM:SS", same format as hand-history headers
    end_time: str
    # Main balances are OPTIONAL (both-or-neither): a session logged from a
    # hand-history upload can source its table result from hh_table_result instead.
    # When both sources exist, balances win (they capture postflop-excluded net too)
    # and the HH number becomes the cross-check.
    start_balance: float | None = None
    end_balance: float | None = None
    # Rakeback balances are OPTIONAL (both-or-neither): skipping them loses the
    # table/rakeback split and the rho estimate for this session, nothing else.
    rakeback_balance_start: float | None = None
    rakeback_balance_end: float | None = None
    rakeback_claimed: float = 0.0  # rakeback moved into the main balance mid-session
    deposits: float = 0.0
    withdrawals: float = 0.0
    rake_paid_start: float | None = None  # mission/leaderboard "rake paid" counter
    rake_paid_end: float | None = None
    hands_played: int | None = None  # manual estimate; HH join supersedes it
    hh_table_result: float | None = None  # sum of hero_net_currency over the session's
    # hands, stored at logging time so the ledger stands alone without the export
    stakes_note: str = ""  # human reference only (e.g. "1/2 + 2/5"), never used for math
    notes: str = ""

    def __post_init__(self) -> None:
        start = parse_timestamp(self.start_time)
        end = parse_timestamp(self.end_time)
        if end <= start:
            raise ValueError(f"end_time {self.end_time!r} must be after start_time {self.start_time!r}")
        non_negative = {
            "start_balance": self.start_balance,
            "end_balance": self.end_balance,
            "rakeback_balance_start": self.rakeback_balance_start,
            "rakeback_balance_end": self.rakeback_balance_end,
            "rakeback_claimed": self.rakeback_claimed,
            "deposits": self.deposits,
            "withdrawals": self.withdrawals,
        }
        for name, value in non_negative.items():
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        for a, b in (("start_balance", "end_balance"),
                     ("rakeback_balance_start", "rakeback_balance_end"),
                     ("rake_paid_start", "rake_paid_end")):
            if (getattr(self, a) is None) != (getattr(self, b) is None):
                raise ValueError(f"{a} and {b} must be recorded together or not at all")
        if self.start_balance is None and self.hh_table_result is None:
            raise ValueError(
                "a session needs a table-result source: either both main balances "
                "or hh_table_result (from a hand-history join)"
            )
        if self.hh_table_result is not None and not math.isfinite(self.hh_table_result):
            raise ValueError(f"hh_table_result must be finite, got {self.hh_table_result}")
        if self.rake_paid_start is not None and self.rake_paid_end < self.rake_paid_start:
            raise ValueError(
                f"rake_paid counter cannot decrease ({self.rake_paid_start} -> {self.rake_paid_end})"
            )
        if self.hands_played is not None and self.hands_played < 0:
            raise ValueError(f"hands_played must be non-negative, got {self.hands_played}")


def table_result(s: Session) -> float:
    """What the tables actually paid/cost this session. Balance-derived when balance
    readings exist (they also capture postflop-excluded net, and the three non-table
    movements are stripped back out); else the stored hand-history-derived net.
    Validation guarantees at least one source exists."""
    if s.start_balance is not None:
        return s.end_balance - s.start_balance - s.rakeback_claimed - s.deposits + s.withdrawals
    return s.hh_table_result


def rakeback_earned(s: Session) -> float | None:
    """Rakeback accrued during the session, whether or not it was claimed: the
    claimable balance's delta plus whatever was moved out of it mid-session.
    None when the rakeback balance wasn't recorded for this session."""
    if s.rakeback_balance_start is None:
        return None
    return s.rakeback_balance_end - s.rakeback_balance_start + s.rakeback_claimed


def total_result(s: Session) -> float:
    """Table plus rakeback; a session without rakeback readings contributes its
    table result alone (the rakeback it earned is simply unmeasured, not zero --
    summarize() reports how many sessions are in that state)."""
    rakeback = rakeback_earned(s)
    return table_result(s) + (rakeback if rakeback is not None else 0.0)


def effective_rakeback_rate(s: Session) -> float | None:
    """rho = rakeback earned per unit of rake attributed by the client's own
    counter -- the number the solver's fee model needs. None when the counter or
    the rakeback balance wasn't recorded, or the counter didn't move (an all-fold
    session can legitimately not move it)."""
    rakeback = rakeback_earned(s)
    if rakeback is None or s.rake_paid_start is None or s.rake_paid_end is None:
        return None
    delta = s.rake_paid_end - s.rake_paid_start
    if delta <= 0:
        return None
    return rakeback / delta


# ---------------------------------------------------------------------------------
# CSV persistence. Append-only, human-editable; one bad row never hides the rest
# (same (results, failures) contract as hand_history.parse_file).
# ---------------------------------------------------------------------------------

CSV_COLUMNS = tuple(f.name for f in fields(Session))


def append_session(path: str, session: Session) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with open(p, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        row = {k: ("" if v is None else v) for k, v in asdict(session).items()}
        writer.writerow(row)


def _session_from_row(row: dict) -> Session:
    """Build a Session from a dict of raw values (CSV strings, or the JSON types the
    logging server receives -- floats/None pass through unchanged)."""
    def opt_float(name: str) -> float | None:
        raw = row.get(name)
        return float(raw) if raw not in (None, "") else None

    def def_float(name: str, default: float) -> float:
        raw = row.get(name)
        return float(raw) if raw not in (None, "") else default

    hands_raw = row.get("hands_played")
    return Session(
        start_time=row.get("start_time") or "",
        end_time=row.get("end_time") or "",
        start_balance=opt_float("start_balance"),
        end_balance=opt_float("end_balance"),
        rakeback_balance_start=opt_float("rakeback_balance_start"),
        rakeback_balance_end=opt_float("rakeback_balance_end"),
        rakeback_claimed=def_float("rakeback_claimed", 0.0),
        deposits=def_float("deposits", 0.0),
        withdrawals=def_float("withdrawals", 0.0),
        rake_paid_start=opt_float("rake_paid_start"),
        rake_paid_end=opt_float("rake_paid_end"),
        hands_played=int(hands_raw) if hands_raw not in (None, "") else None,
        hh_table_result=opt_float("hh_table_result"),
        stakes_note=str(row.get("stakes_note") or ""),
        notes=str(row.get("notes") or ""),
    )


def load_sessions(path: str) -> tuple[list[Session], list[tuple[int, str]]]:
    """Returns (sessions, failures); failures are (1-based data-row number, error)."""
    p = Path(path)
    if not p.exists():
        return [], []
    sessions: list[Session] = []
    failures: list[tuple[int, str]] = []
    with open(p, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=1):
            try:
                sessions.append(_session_from_row(row))
            except Exception as exc:  # noqa: BLE001 -- one bad row must not hide the ledger
                failures.append((row_num, str(exc)))
    return sessions, failures


# ---------------------------------------------------------------------------------
# Open-session staging: `start` stages a JSON stub, `end` completes it into a CSV
# row -- the CSV itself never holds half-filled rows.
# ---------------------------------------------------------------------------------


def stage_open_session(path: str, stub: dict) -> None:
    p = Path(path)
    if p.exists():
        raise OpenSessionExistsError(
            f"an open session is already staged at {path} -- finish it with 'end' first"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(stub, f, indent=2)


def load_open_session(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def clear_open_session(path: str) -> None:
    Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------------
# Hand-history join: the only source of per-stake stats.
# ---------------------------------------------------------------------------------

def hands_in_window(hands: list[ParsedHand], start_time: str, end_time: str) -> list[ParsedHand]:
    """Hands whose header timestamp falls inside [start_time, end_time], inclusive."""
    start = parse_timestamp(start_time)
    end = parse_timestamp(end_time)
    return [h for h in hands if start <= parse_timestamp(h.timestamp) <= end]


@dataclass(frozen=True)
class DetectedSession:
    start_time: str  # first hand's timestamp, session-tracker format (no tz)
    end_time: str  # last hand's timestamp
    hands: tuple  # the ParsedHand slice, chronological


def detect_sessions(hands: list[ParsedHand], gap_minutes: float = 45.0) -> list[DetectedSession]:
    """Cluster hands into sessions by inter-hand time gap: a gap >= gap_minutes starts
    a new session. On the real data this is nearly unambiguous (13,105 of 13,132 gaps
    are <5min and the next tier is hours; only 4 gaps fall in 20-90min over 5 months),
    so the default threshold hardly matters -- it's a knob, not a tuning problem."""
    if not hands:
        return []
    stamped = sorted(((parse_timestamp(h.timestamp), h) for h in hands), key=lambda p: p[0])
    fmt = "%Y/%m/%d %H:%M:%S"

    clusters: list[list] = [[stamped[0]]]
    for prev, cur in zip(stamped, stamped[1:]):
        if (cur[0] - prev[0]).total_seconds() >= gap_minutes * 60:
            clusters.append([])
        clusters[-1].append(cur)

    return [
        DetectedSession(
            start_time=cluster[0][0].strftime(fmt),
            end_time=cluster[-1][0].strftime(fmt),
            hands=tuple(h for _, h in cluster),
        )
        for cluster in clusters
    ]


def hero_net_currency(hand: ParsedHand) -> float | None:
    """Hero's net for one hand (collected minus total chips put in), in currency --
    exact across ALL streets: blinds/antes posted, then each street's actions with
    the bet level reset per street ("raises ... to X" is street-absolute), minus
    uncalled-bet returns; collected sums every board of a run-it-twice hand.

    Returns None only when HERO produced an action line the parser couldn't classify
    (hand.unclassified_action_names) -- then hero's money movement is untrustworthy.
    Villains' unparsed oddities don't matter: collected is ground truth and only
    hero's own actions move hero's chips.
    """
    hero = hand.hero_name or "Hero"
    if hero in hand.unclassified_action_names:
        return None
    posted = 0.0
    if hero == hand.sb_poster:
        posted += hand.sb_amount
    if hero == hand.bb_poster:
        posted += hand.bb_amount
    if hero in hand.ante_posters and hand.ante_amount:
        posted += hand.ante_amount

    total_in = posted
    for street_idx, actions in enumerate([hand.preflop_actions, *hand.postflop_streets]):
        # preflop level starts at hero's posted blinds; each later street starts at 0
        level = posted if street_idx == 0 else 0.0
        for action in actions:
            if action.name != hero or action.to_amount is None:
                continue
            if action.action == "raises":
                total_in += action.to_amount - level
                level = action.to_amount
            elif action.action in ("calls", "bets", "allin", "post"):  # additional chips
                total_in += action.to_amount
                level += action.to_amount
            elif action.action == "return":
                total_in -= action.to_amount

    return hand.collected_by.get(hero, 0.0) - total_in


@dataclass(frozen=True)
class StakeStats:
    bb_amount: float
    hands: int  # all hands seen at this stake in the window
    hands_with_net: int  # hands whose net was reconstructable (preflop-terminal)
    net_currency: float

    @property
    def net_bb(self) -> float:
        return self.net_currency / self.bb_amount

    @property
    def bb_per_100(self) -> float | None:
        if self.hands_with_net == 0:
            return None
        return 100.0 * self.net_bb / self.hands_with_net


def per_stake_stats(hands: list[ParsedHand]) -> dict[float, StakeStats]:
    """Group hands by stake (bb_amount) -> hands played and hero net per stake."""
    counts: dict[float, int] = {}
    with_net: dict[float, int] = {}
    nets: dict[float, float] = {}
    for hand in hands:
        bb = hand.bb_amount
        counts[bb] = counts.get(bb, 0) + 1
        net = hero_net_currency(hand)
        if net is not None:
            with_net[bb] = with_net.get(bb, 0) + 1
            nets[bb] = nets.get(bb, 0.0) + net
    return {
        bb: StakeStats(
            bb_amount=bb,
            hands=counts[bb],
            hands_with_net=with_net.get(bb, 0),
            net_currency=nets.get(bb, 0.0),
        )
        for bb in counts
    }


# ---------------------------------------------------------------------------------
# Summary across sessions.
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    n_sessions: int
    total_table_result: float
    total_rakeback_earned: float  # over sessions that recorded rakeback balances
    total_result: float
    mean_session_result: float | None  # total_result per session
    std_session_result: float | None  # sample std (ddof=1), needs >= 2 sessions
    rho_sessions: int  # sessions where the rake-paid counter allowed a rho estimate
    rho_mean: float | None
    rho_se: float | None  # standard error of the mean rho, needs >= 2 estimates
    rakeback_missing: int  # sessions logged without rakeback balances (unmeasured, not 0)


def summarize(sessions: list[Session]) -> Summary:
    n = len(sessions)
    results = [total_result(s) for s in sessions]
    rakebacks = [rakeback_earned(s) for s in sessions]
    rhos = [r for r in (effective_rakeback_rate(s) for s in sessions) if r is not None]

    mean_result = sum(results) / n if n else None
    std_result = None
    if n >= 2:
        std_result = math.sqrt(sum((x - mean_result) ** 2 for x in results) / (n - 1))

    rho_mean = sum(rhos) / len(rhos) if rhos else None
    rho_se = None
    if len(rhos) >= 2:
        rho_var = sum((r - rho_mean) ** 2 for r in rhos) / (len(rhos) - 1)
        rho_se = math.sqrt(rho_var / len(rhos))

    return Summary(
        n_sessions=n,
        total_table_result=sum(table_result(s) for s in sessions),
        total_rakeback_earned=sum(r for r in rakebacks if r is not None),
        total_result=sum(results),
        mean_session_result=mean_result,
        std_session_result=std_result,
        rho_sessions=len(rhos),
        rho_mean=rho_mean,
        rho_se=rho_se,
        rakeback_missing=sum(1 for r in rakebacks if r is None),
    )
