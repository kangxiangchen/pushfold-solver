"""Raw hand-history text parser: CoinPoker .txt export -> structured ParsedHand.

Pure text-format knowledge only -- this module has no idea what a "push/fold spot"
is (see pushfold_spot.py for that). Splitting on the hand-header line (not on blank
lines) makes this robust to whitespace variance in real exports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

NUMERIC_RE = re.compile(r"[\d,]+\.?\d*")


def parse_timestamp(ts: str) -> datetime:
    """'2026/06/19 13:42:37 +08' (hand-header form) or bare '2026/06/19 13:42:37'
    (session-tracker form) -> naive datetime. The timezone suffix is dropped: all
    data comes from one venue in one fixed offset, so naive comparison is safe."""
    parts = ts.split()
    return datetime.strptime(f"{parts[0]} {parts[1]}", "%Y/%m/%d %H:%M:%S")


def parse_amount(raw: str) -> float:
    """Extract a float from currency text, tolerating thousands-separator commas
    (e.g. '₮1,403.08' -- a deep-pocketed opponent's stack)."""
    match = NUMERIC_RE.search(raw)
    if not match:
        raise ValueError(f"no numeric amount found in {raw!r}")
    return float(match.group().replace(",", ""))


HAND_HEADER_RE = re.compile(
    r"^CoinPoker Hand #(?P<hand_id>\d+): NLH \((?P<stakes>[^)]+)\) "
    r"(?P<date>\S+) (?P<time>\S+) (?P<tz>\S+)\s*$",
    re.MULTILINE,
)
TABLE_RE = re.compile(r"^Table '(?P<table>[^']+)' (?P<maxseats>\d+)-max Seat #(?P<button>\d+) is the button$")
SEAT_RE = re.compile(r"^Seat (?P<seat_num>\d+): (?P<name>.+?) \((?P<stack_raw>[^)]+) in chips\)$")
ANTE_RE = re.compile(r"^(?P<name>.+?): posts ante (?P<amt_raw>.+)$")
SB_POST_RE = re.compile(r"^(?P<name>.+?): posts small blind (?P<amt_raw>.+)$")
BB_POST_RE = re.compile(r"^(?P<name>.+?): posts big blind (?P<amt_raw>.+)$")
HOLE_CARDS_MARKER_RE = re.compile(r"^\*\*\* HOLE CARDS \*\*\*$")
# Deliberately NOT anchored at $: street banners can carry trailing cards
# ("*** FLOP *** [Ts Jh 4h]" -- verified in real exports), and an end-anchored
# pattern would miss them, silently leaking postflop actions into preflop_actions.
ANY_BANNER_RE = re.compile(r"^\*\*\*.*\*\*\*")
DEALT_RE = re.compile(r"^Dealt to (?P<name>.+?)(?: \[(?P<c1>\w+) (?P<c2>\w+)\])?$")

FOLD_RE = re.compile(r"^(?P<name>.+?): folds$")
CHECK_RE = re.compile(r"^(?P<name>.+?): checks$")
RETURN_RE = re.compile(r"^(?P<name>.+?): RETURN (?P<amt_raw>.+)$")
ALLIN_RE = re.compile(r"^(?P<name>.+?): ALLIN (?P<amt_raw>.+)$")
RAISE_RE = re.compile(r"^(?P<name>.+?): raises (?P<by_raw>.+) to (?P<to_raw>.+)$")
CALL_RE = re.compile(r"^(?P<name>.+?): calls (?P<amt_raw>.+)$")
BET_RE = re.compile(r"^(?P<name>.+?): bets (?P<amt_raw>.+)$")
# AUTOBB (auto-posted big blind on joining/returning) and STRADDLE both appear in the
# action region of real exports (56 occurrences across 14.5k hands) -- both are chips
# into the pot, classified as one "post" action type.
POST_RE = re.compile(r"^(?P<name>.+?): (?:AUTOBB|STRADDLE) (?P<amt_raw>.+)$")

# Street banners carry trailing cards ("*** FLOP *** [Ts Jh 4h]") and appear both for
# genuine betting streets and for the board runout of a called all-in (newer export
# format); FIRST/SECOND variants come from run-it-twice hands. They start a new street;
# any OTHER banner (SHOWDOWN, SUMMARY, and anything future) ends the action region.
STREET_BANNER_RE = re.compile(r"^\*\*\* (?:FIRST |SECOND )?(?:FLOP|TURN|RIVER) \*\*\*")
# A line shaped like an action ("name: verb ...") that _classify_action can't parse --
# counted per actor so downstream money math can bail out honestly (see ParsedHand).
ACTIONISH_RE = re.compile(r"^(?P<name>.+?): \S+")

COLLECTED_RE = re.compile(r"^(?P<name>.+?) collected (?P<amt_raw>.+) from pot$", re.MULTILINE)
# Shown hole cards: the showdown region's "X: shows [A B] (One Pair)" line, with the
# summary's "Seat N: X showed [A B] and won/lost ..." as a fallback for names that
# never got a shows-line (e.g. the winner of an uncalled shove). Not end-anchored --
# both carry trailing hand-strength text.
SHOWS_RE = re.compile(r"^(?P<name>.+?): shows \[(?P<c1>\w+) (?P<c2>\w+)\]", re.MULTILINE)
SUMMARY_SHOWED_RE = re.compile(r"^Seat \d+: (?P<name>.+?) showed \[(?P<c1>\w+) (?P<c2>\w+)\]", re.MULTILINE)
TOTAL_POT_RE = re.compile(
    r"^Total pot (?P<pot_raw>\S+)(?: \| Rake (?P<rake_raw>\S+))?(?: \| Splash Fee (?P<splash_raw>\S+))?$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Seat:
    seat_num: int
    name: str
    starting_stack: float


@dataclass(frozen=True)
class PreflopAction:
    name: str
    action: str  # "folds" | "checks" | "calls" | "bets" | "raises" | "allin" | "return" | "post"
    to_amount: float | None  # "to" amount for raises; the stated amount otherwise; None for folds/checks


@dataclass(frozen=True)
class ParsedHand:
    hand_id: str
    timestamp: str
    table_name: str
    max_seats: int
    button_seat: int
    sb_amount: float
    bb_amount: float
    ante_amount: float | None
    seats: list[Seat] = field(default_factory=list)
    sb_poster: str | None = None
    bb_poster: str | None = None
    ante_posters: list[str] = field(default_factory=list)
    dealt_names: list[str] = field(default_factory=list)
    hero_name: str | None = None
    hero_cards: tuple[str, str] | None = None
    preflop_actions: list[PreflopAction] = field(default_factory=list)
    postflop_streets: list[list[PreflopAction]] = field(default_factory=list)  # one list per
    # street banner; EMPTY lists for the runout streets of a called all-in
    unclassified_action_names: list[str] = field(default_factory=list)  # actors of action-shaped
    # lines the classifier couldn't parse -- money math for those actors is untrustworthy
    collected_by: dict[str, float] = field(default_factory=dict)  # currency amounts, from SUMMARY/SHOWDOWN,
    # SUMMED per name (run-it-twice hands collect once per board)
    shown_cards: dict[str, tuple[str, str]] = field(default_factory=dict)  # name -> hole cards
    # revealed at showdown (SHOWS_RE, SUMMARY_SHOWED_RE); one entry per name even when
    # a run-it-twice hand prints the shows-line once per board
    total_pot: float | None = None  # currency, from "Total pot X | Rake Y | Splash Fee Z"
    rake: float | None = None
    splash_fee: float = 0.0
    raw_text: str = ""


def split_hands(full_text: str) -> list[str]:
    """Slice full_text at every hand-header match start; each slice runs to the
    next match (or EOF). Never depends on blank lines."""
    starts = [m.start() for m in HAND_HEADER_RE.finditer(full_text)]
    if not starts:
        return []
    starts.append(len(full_text))
    return [full_text[starts[i] : starts[i + 1]] for i in range(len(starts) - 1)]


def _parse_stakes(stakes: str) -> tuple[float, float, float | None]:
    tokens = [t.strip() for t in stakes.split("/")]
    if len(tokens) not in (2, 3):
        raise ValueError(f"unexpected stakes header {stakes!r}")
    sb = parse_amount(tokens[0])
    bb = parse_amount(tokens[1])
    ante = parse_amount(tokens[2]) if len(tokens) == 3 else None
    return sb, bb, ante


def _classify_action(line: str) -> PreflopAction | None:
    if m := FOLD_RE.match(line):
        return PreflopAction(m.group("name"), "folds", None)
    if m := CHECK_RE.match(line):
        return PreflopAction(m.group("name"), "checks", None)
    if m := RETURN_RE.match(line):
        return PreflopAction(m.group("name"), "return", parse_amount(m.group("amt_raw")))
    if m := ALLIN_RE.match(line):
        return PreflopAction(m.group("name"), "allin", parse_amount(m.group("amt_raw")))
    if m := RAISE_RE.match(line):
        return PreflopAction(m.group("name"), "raises", parse_amount(m.group("to_raw")))
    if m := CALL_RE.match(line):
        return PreflopAction(m.group("name"), "calls", parse_amount(m.group("amt_raw")))
    if m := BET_RE.match(line):
        return PreflopAction(m.group("name"), "bets", parse_amount(m.group("amt_raw")))
    if m := POST_RE.match(line):
        return PreflopAction(m.group("name"), "post", parse_amount(m.group("amt_raw")))
    return None


def parse_hand(hand_text: str) -> ParsedHand:
    lines = [line.strip() for line in hand_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty hand text")

    header_match = HAND_HEADER_RE.match(lines[0])
    if not header_match:
        raise ValueError(f"first line is not a hand header: {lines[0]!r}")
    hand_id = header_match.group("hand_id")
    timestamp = f"{header_match.group('date')} {header_match.group('time')} {header_match.group('tz')}"
    sb_amount, bb_amount, ante_amount = _parse_stakes(header_match.group("stakes"))

    table_name = ""
    max_seats = 0
    button_seat = 0
    seats: list[Seat] = []
    ante_posters: list[str] = []
    sb_poster: str | None = None
    bb_poster: str | None = None
    dealt_names: list[str] = []
    hero_name: str | None = None
    hero_cards: tuple[str, str] | None = None
    preflop_actions: list[PreflopAction] = []

    i = 1
    n = len(lines)

    while i < n:
        line = lines[i]
        if m := TABLE_RE.match(line):
            table_name = m.group("table")
            max_seats = int(m.group("maxseats"))
            button_seat = int(m.group("button"))
        elif m := SEAT_RE.match(line):
            seats.append(Seat(int(m.group("seat_num")), m.group("name"), parse_amount(m.group("stack_raw"))))
        elif m := ANTE_RE.match(line):
            ante_posters.append(m.group("name"))
        elif m := SB_POST_RE.match(line):
            sb_poster = m.group("name")
        elif m := BB_POST_RE.match(line):
            bb_poster = m.group("name")
        elif HOLE_CARDS_MARKER_RE.match(line):
            i += 1
            break
        i += 1

    while i < n and (m := DEALT_RE.match(lines[i])):
        name = m.group("name")
        dealt_names.append(name)
        if m.group("c1") and m.group("c2"):
            hero_name = name
            hero_cards = (m.group("c1"), m.group("c2"))
        i += 1

    # Action region: preflop actions, then one bucket per street banner (runout-only
    # streets simply stay empty). Any non-street banner (SHOWDOWN/SUMMARY/unknown)
    # ends the region -- verified against real exports that RETURN never appears
    # after those banners. Action-shaped lines the classifier can't parse are
    # recorded by actor instead of silently dropped.
    streets: list[list[PreflopAction]] = [preflop_actions]
    unclassified_action_names: list[str] = []
    while i < n:
        line = lines[i]
        if STREET_BANNER_RE.match(line):
            streets.append([])
        elif ANY_BANNER_RE.match(line):
            break
        else:
            action = _classify_action(line)
            if action is not None:
                streets[-1].append(action)
            elif m := ACTIONISH_RE.match(line):
                unclassified_action_names.append(m.group("name"))
        i += 1
    postflop_streets = streets[1:]

    stripped_text = "\n".join(lines)
    collected_by: dict[str, float] = {}
    for m in COLLECTED_RE.finditer(stripped_text):
        name = m.group("name")
        collected_by[name] = collected_by.get(name, 0.0) + parse_amount(m.group("amt_raw"))
    # setdefault gives all three semantics at once: run-it-twice duplicate shows-lines
    # dedup, the showdown-region line wins over a conflicting summary line, and the
    # summary "showed" fills names that never got a shows-line.
    shown_cards: dict[str, tuple[str, str]] = {}
    for m in SHOWS_RE.finditer(stripped_text):
        shown_cards.setdefault(m.group("name"), (m.group("c1"), m.group("c2")))
    for m in SUMMARY_SHOWED_RE.finditer(stripped_text):
        shown_cards.setdefault(m.group("name"), (m.group("c1"), m.group("c2")))

    total_pot: float | None = None
    rake: float | None = None
    splash_fee = 0.0
    if pot_match := TOTAL_POT_RE.search(stripped_text):
        total_pot = parse_amount(pot_match.group("pot_raw"))
        rake = parse_amount(pot_match.group("rake_raw")) if pot_match.group("rake_raw") else None
        splash_fee = parse_amount(pot_match.group("splash_raw")) if pot_match.group("splash_raw") else 0.0

    return ParsedHand(
        hand_id=hand_id,
        timestamp=timestamp,
        table_name=table_name,
        max_seats=max_seats,
        button_seat=button_seat,
        sb_amount=sb_amount,
        bb_amount=bb_amount,
        ante_amount=ante_amount,
        seats=seats,
        sb_poster=sb_poster,
        bb_poster=bb_poster,
        ante_posters=ante_posters,
        dealt_names=dealt_names,
        hero_name=hero_name,
        hero_cards=hero_cards,
        preflop_actions=preflop_actions,
        postflop_streets=postflop_streets,
        unclassified_action_names=unclassified_action_names,
        collected_by=collected_by,
        shown_cards=shown_cards,
        total_pot=total_pot,
        rake=rake,
        splash_fee=splash_fee,
        raw_text=hand_text,
    )


def parse_file(path: str) -> tuple[list[ParsedHand], list[tuple[str, str]]]:
    """Returns (hands, failures) -- one bad hand never aborts the whole file.
    failures is a list of (hand_id_or_'?', error_message)."""
    with open(path, encoding="utf-8") as f:
        full_text = f.read()

    hands: list[ParsedHand] = []
    failures: list[tuple[str, str]] = []
    for hand_text in split_hands(full_text):
        try:
            hands.append(parse_hand(hand_text))
        except Exception as exc:  # noqa: BLE001 -- one malformed hand must not abort the run
            header_match = HAND_HEADER_RE.match(hand_text.strip().splitlines()[0]) if hand_text.strip() else None
            hand_id = header_match.group("hand_id") if header_match else "?"
            failures.append((hand_id, str(exc)))
    return hands, failures
