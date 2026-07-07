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
import re
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
    start_balance: float
    end_balance: float
    rakeback_balance_start: float
    rakeback_balance_end: float
    rakeback_claimed: float = 0.0  # rakeback moved into the main balance mid-session
    deposits: float = 0.0
    withdrawals: float = 0.0
    rake_paid_start: float | None = None  # mission/leaderboard "rake paid" counter
    rake_paid_end: float | None = None
    hands_played: int | None = None  # manual estimate; HH join supersedes it
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
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if (self.rake_paid_start is None) != (self.rake_paid_end is None):
            raise ValueError("rake_paid_start and rake_paid_end must be recorded together or not at all")
        if self.rake_paid_start is not None and self.rake_paid_end < self.rake_paid_start:
            raise ValueError(
                f"rake_paid counter cannot decrease ({self.rake_paid_start} -> {self.rake_paid_end})"
            )
        if self.hands_played is not None and self.hands_played < 0:
            raise ValueError(f"hands_played must be non-negative, got {self.hands_played}")


def table_result(s: Session) -> float:
    """What the tables actually paid/cost this session, with the three non-table
    balance movements (claims, deposits, withdrawals) stripped back out."""
    return s.end_balance - s.start_balance - s.rakeback_claimed - s.deposits + s.withdrawals


def rakeback_earned(s: Session) -> float:
    """Rakeback accrued during the session, whether or not it was claimed: the
    claimable balance's delta plus whatever was moved out of it mid-session."""
    return s.rakeback_balance_end - s.rakeback_balance_start + s.rakeback_claimed


def total_result(s: Session) -> float:
    return table_result(s) + rakeback_earned(s)


def effective_rakeback_rate(s: Session) -> float | None:
    """rho = rakeback earned per unit of rake attributed by the client's own
    counter -- the number the solver's fee model needs. None when the counter
    wasn't recorded or didn't move (an all-fold session can legitimately not move it)."""
    if s.rake_paid_start is None or s.rake_paid_end is None:
        return None
    delta = s.rake_paid_end - s.rake_paid_start
    if delta <= 0:
        return None
    return rakeback_earned(s) / delta


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
    def opt_float(name: str) -> float | None:
        raw = row.get(name)
        return float(raw) if raw not in (None, "") else None

    def req_float(name: str) -> float:
        raw = row.get(name)
        if raw in (None, ""):
            raise ValueError(f"missing required field {name!r}")
        return float(raw)

    hands_raw = row.get("hands_played")
    return Session(
        start_time=row.get("start_time") or "",
        end_time=row.get("end_time") or "",
        start_balance=req_float("start_balance"),
        end_balance=req_float("end_balance"),
        rakeback_balance_start=req_float("rakeback_balance_start"),
        rakeback_balance_end=req_float("rakeback_balance_end"),
        rakeback_claimed=req_float("rakeback_claimed"),
        deposits=req_float("deposits"),
        withdrawals=req_float("withdrawals"),
        rake_paid_start=opt_float("rake_paid_start"),
        rake_paid_end=opt_float("rake_paid_end"),
        hands_played=int(hands_raw) if hands_raw not in (None, "") else None,
        stakes_note=row.get("stakes_note") or "",
        notes=row.get("notes") or "",
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

# Street banners with betting rounds. Runout-only all-ins in this data do NOT print
# these (the board appears in the SUMMARY line instead -- verified against real
# exports), so their presence means genuine postflop streets happened.
_POSTFLOP_BANNER_RE = re.compile(r"^\*\*\* (?:FIRST |SECOND )?(?:FLOP|TURN|RIVER) \*\*\*", re.MULTILINE)


def hands_in_window(hands: list[ParsedHand], start_time: str, end_time: str) -> list[ParsedHand]:
    """Hands whose header timestamp falls inside [start_time, end_time], inclusive."""
    start = parse_timestamp(start_time)
    end = parse_timestamp(end_time)
    return [h for h in hands if start <= parse_timestamp(h.timestamp) <= end]


def hero_net_currency(hand: ParsedHand) -> float | None:
    """Hero's net for one hand (collected minus total chips put in), in currency.

    Exact for preflop-terminal hands (every hand in the all-in-or-fold formats):
    blinds/antes posted, plus each voluntary action tracked at the level the parser
    recorded, minus uncalled-bet returns. Returns None for hands with genuine
    postflop betting streets -- the parser only captures preflop actions, so a net
    for those would be silently wrong rather than approximately right.
    """
    if _POSTFLOP_BANNER_RE.search(hand.raw_text):
        return None
    hero = hand.hero_name or "Hero"
    posted = 0.0
    if hero == hand.sb_poster:
        posted += hand.sb_amount
    if hero == hand.bb_poster:
        posted += hand.bb_amount
    if hero in hand.ante_posters and hand.ante_amount:
        posted += hand.ante_amount

    total_in = posted
    level = posted  # hero's current committed bet level ("raises ... to X" is absolute)
    for action in hand.preflop_actions:
        if action.name != hero or action.to_amount is None:
            continue
        if action.action == "raises":
            total_in += action.to_amount - level
            level = action.to_amount
        elif action.action in ("calls", "allin"):  # both are stated as additional chips
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
    total_rakeback_earned: float
    total_result: float
    mean_session_result: float | None  # total_result per session
    std_session_result: float | None  # sample std (ddof=1), needs >= 2 sessions
    rho_sessions: int  # sessions where the rake-paid counter allowed a rho estimate
    rho_mean: float | None
    rho_se: float | None  # standard error of the mean rho, needs >= 2 estimates


def summarize(sessions: list[Session]) -> Summary:
    n = len(sessions)
    results = [total_result(s) for s in sessions]
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
        total_rakeback_earned=sum(rakeback_earned(s) for s in sessions),
        total_result=sum(results),
        mean_session_result=mean_result,
        std_session_result=std_result,
        rho_sessions=len(rhos),
        rho_mean=rho_mean,
        rho_se=rho_se,
    )
