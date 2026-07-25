"""Per-session GTO analysis: joins the backtest grading engine (backtest.py) with
the session tracker's hand-history windows (sessions.py) to answer, per logged
session, (a) how well hero's push/fold decisions matched the solved equilibrium
and (b) how much of the realized result was variance rather than decision quality.

Feature A (luck vs skill) compares, per gradeable spot (HU and 4MAX alike), the
model's predicted EV of hero's ACTUAL decision (GradedSpot.predicted_ev_net_bb)
with a realized EV built from the hand's true outcome: realized_spot_ev_net_bb =
exact ledger net (sessions.hero_net_currency -- handles RETURN lines and all
streets, and is balance-cross-checked by the session tracker) plus the MODEL's own
rakeback term for the leaf that actually realized. This deliberately diverges from
backtest.realized_ev_net_bb's `collected - model contribution` ledger: CoinPoker's
`collected` for an UNCALLED shove includes hero's own matched-but-returned-pot
chips (~+1bb vs ev.steal_ev_net's convention, verified against a real hand), a
quirk that stays inside noise for HU validation but would be first-order across
the steal-heavy 4MAX spots. For called showdowns and folds the two conventions
coincide exactly. A 4MAX spot only gets a realized leg when the whole hand
resolves as pure shove-or-fold (fourmax_spot.classify_fourmax_realization);
villains limping behind hero's fold has no model leaf and is excluded from the
luck sums (never from the grade). Ungraded hands (walks, limps, non-push/fold,
deep stacks) have no model EV at all; callers show their realized net separately,
honestly labeled.

Possible v3, deliberately not built: parse villains' shown cards from the showdown
region into ParsedHand and compute all-in-adjusted EV with the 169x169 equity
table. Range-EV (this module) already removes strictly more variance (it also
covers folds and uncontested shoves), so the extension would only refine hands
that reached a called all-in.

Everything expensive is load-only here: the equity table and 4-max ranges must
already exist on disk (make_grading_context returns None / a HU-only context
otherwise). Solving 4-max or building the equity table takes minutes-to-hours and
must never be triggered from a dashboard request. Per-depth HU solves (seconds,
via backtest.SolveCache) are the only solver work this module ever causes, and
graded results persist per source export under cache/session_grades/ so even that
cost is paid once per file version.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import backtest
import equity
import tree
from backtest import GradedSpot, SolveCache, StackTooSmallError
from fourmax_spot import classify_fourmax_realization, find_fourmax_spot
from game import FOURMAX_CONFIG, GameConfig
from hand_history import ParsedHand, parse_timestamp
from pushfold_spot import classify_bb_response_when_hero_is_sb, find_spot
from sessions import hero_net_currency

GRADING_VERSION = "grades-v2"  # bump to invalidate every persisted grades file
# (v2: realized leg switched to hero_net_currency + model rakeback, 4MAX included)

# A gto_weight strictly inside this band means the equilibrium itself mixes both
# actions for the hand -- either choice is GTO, so accuracy counts it as a match
# regardless of which one hero picked. Deterministic, unlike an ev_loss==0 test,
# which 4MAX Monte Carlo noise would break.
MIXED_BAND = (0.01, 0.99)

# Below this, an ev_loss_bb is numerical noise on a matched decision, not a mistake.
EV_LOSS_EPSILON = 1e-6

# HU skip reasons a 4-max model could still grade -- same set as scripts/backtest.py
# (not imported from there: src/ modules never depend on scripts/).
FOURMAX_RETRY_REASONS = frozenset({"hero_not_blind_poster", "shove_multiway", "call_multiway"})


@dataclass(frozen=True)
class GradingContext:
    """Everything grade_hands needs, built once and reused across files/requests."""

    solve_cache: SolveCache  # per-depth HU solves (cheap, in-memory)
    fourmax: tuple[GameConfig, dict] | None  # pre-solved (cfg, ranges) or None -> HU-only
    evaluator: object | None  # 7-card evaluator, needed only for 4-max multiway leaves
    hero_name: str = "Hero"
    # Shared (model, depth, info set) -> EV-vector memo (see grade_spot_against's
    # docstring): turns bulk 4-max grading from one MC evaluation per SPOT into one
    # per INFO SET. Mutable on purpose -- it's a cache, not state.
    ev_memo: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GradedHand:
    """One gradeable hand: the GradedSpot plus the per-hand context grading strips off."""

    graded: GradedSpot
    bb_amount: float
    net_currency: float | None  # sessions.hero_net_currency -- ledger ground truth
    realized_ev_net_bb: float | None  # realized_spot_ev_net_bb; None when the hand's
    # continuation doesn't map onto the model (or hero's money movement is unparseable)


def load_fourmax_ranges_if_cached(cache_dir: str = "cache/", stack_bb: float = 8.0) -> tuple[GameConfig, dict] | None:
    """Load-ONLY counterpart of backtest.load_or_solve_fourmax_ranges: tries the
    per-depth pickle (bare RangeMap) then scripts/solve_fourmax.py's wrapped shape
    ({'ranges': ...}); returns None when neither exists. NEVER solves -- a fresh
    4-max solve takes tens of minutes to hours and must not hide behind a page load."""
    cfg = dataclasses.replace(FOURMAX_CONFIG, stack_bb=stack_bb)
    per_depth = backtest.fourmax_ranges_cache_path(cache_dir, stack_bb)
    if per_depth.exists():
        with open(per_depth, "rb") as f:
            return cfg, pickle.load(f)
    wrapped = Path(cache_dir) / "fourmax_ranges.pkl"
    if wrapped.exists():
        with open(wrapped, "rb") as f:
            payload = pickle.load(f)
        ranges = payload["ranges"] if isinstance(payload, dict) and "ranges" in payload else payload
        return cfg, ranges
    return None


def make_grading_context(cache_dir: str = "cache/", hero_name: str = "Hero") -> GradingContext | None:
    """None when the equity table isn't on disk (existence check -- never falls
    through into equity.build_equity_table). 4-max ranges and the evaluator are
    optional extras: missing either just degrades to a HU-only context."""
    table_path = equity._cache_path(cache_dir)
    if not table_path.exists():
        return None
    table = np.load(table_path)

    fourmax = load_fourmax_ranges_if_cached(cache_dir)
    evaluator = None
    if fourmax is not None:
        try:
            evaluator = equity.get_evaluator()
        except Exception:  # noqa: BLE001 -- no 7-card evaluator installed
            fourmax = None  # multiway leaves would be ungradable without one
    return GradingContext(solve_cache=SolveCache(table), fourmax=fourmax, evaluator=evaluator, hero_name=hero_name)


# ---------------------------------------------------------------------------------
# Realized EV: what the hand's actual outcome was worth, in the model's own units.
# ---------------------------------------------------------------------------------


def _hu_rakeback_leaf(hand: ParsedHand, spot, cfg: GameConfig, hero_name: str) -> tuple[float, float] | None:
    """(hero contribution, total pot) of the HU leaf that realized -- the same three
    shapes backtest.realized_ev_net_bb derives (fold / uncalled shove / showdown)."""
    position = "SB" if spot.info_set_key == "SB|" else "BB"
    if spot.hero_action == "fold":
        blind = cfg.blind_of(position)
        return blind, blind
    if position == "SB" and classify_bb_response_when_hero_is_sb(hand, hero_name=hero_name) != "call":
        return 0.0, cfg.blind_of("BB")  # uncalled shove: contributes 0, no rakeback
    return cfg.stack_bb, 2 * cfg.stack_bb


def _fourmax_rakeback_leaf(hand: ParsedHand, spot, cfg: GameConfig, hero_name: str) -> tuple[float, float] | None:
    """(hero contribution, total pot) of the 4MAX leaf that realized, mirroring
    ev.fold_baseline_ev_net / shove_ev_net_vec's leaf conventions: pre-hero dead
    blinds from the info set itself, post-hero folders' blinds observed from the
    hand, one cfg.stack_bb per genuinely live seat (2+ live others required for a
    fold's pot to contain stacks -- ev.py's edge case (2))."""
    realization = classify_fourmax_realization(hand, hero_name=hero_name)
    if realization is None or realization.hero_action != spot.hero_action:
        return None
    infoset = tree.infosets_by_key(tree.build_infosets(cfg))[spot.info_set_key]
    dead = infoset.dead_money_bb + sum(cfg.blind_of(r) for r in realization.folded_after)
    live = len(realization.live_others)

    if spot.hero_action == "fold":
        blind = cfg.blind_of(realization.hero_role)
        pot = dead + blind + (live * cfg.stack_bb if live >= 2 else 0.0)
        return blind, pot
    if live == 0:
        return 0.0, dead  # uncalled shove: contributes 0, no rakeback
    return cfg.stack_bb, (1 + live) * cfg.stack_bb + dead


def realized_spot_ev_net_bb(
    hand: ParsedHand, spot, cfg: GameConfig, model: str, hero_name: str = "Hero"
) -> float | None:
    """The REAL observed ev_net of this hand's outcome, comparable to
    GradedSpot.predicted_ev_net_bb: exact ledger net from hero_net_currency (all
    streets, RETURN lines handled) plus the model-convention rakeback of the leaf
    that realized. See the module docstring for why the ledger leg comes from
    hero_net_currency rather than backtest.realized_ev_net_bb's collected-based
    derivation. None when hero's money movement is unparseable or (4MAX) the
    hand's continuation doesn't map onto the shove-or-fold model."""
    net = hero_net_currency(hand)
    if net is None:
        return None
    leaf_fn = _hu_rakeback_leaf if model == "HU" else _fourmax_rakeback_leaf
    leaf = leaf_fn(hand, spot, cfg, hero_name)
    if leaf is None:
        return None
    contribution_bb, total_pot_bb = leaf
    rakeback = cfg.rakeback_pool_bb * (contribution_bb / total_pot_bb) if total_pot_bb > 0 else 0.0
    return net / hand.bb_amount + rakeback


# ---------------------------------------------------------------------------------
# Grading a hand list (mirrors scripts/backtest.py's loop, kept in src/ so the
# dashboard and any future CLI share one implementation).
# ---------------------------------------------------------------------------------


def _grade_one(hand: ParsedHand, ctx: GradingContext) -> tuple[GradedHand | None, str | None]:
    spot, reason = find_spot(hand, hero_name=ctx.hero_name)
    if spot is not None:
        try:
            solved = ctx.solve_cache.get(spot.effective_stack_bb)
        except StackTooSmallError:
            return None, "stack_too_small_for_config"
        graded = backtest.grade_spot_against(
            spot, solved.cfg, solved.ranges, ctx.solve_cache.table, ev_memo=ctx.ev_memo)
        realized = realized_spot_ev_net_bb(hand, spot, solved.cfg, "HU", hero_name=ctx.hero_name)
        return GradedHand(graded, hand.bb_amount, hero_net_currency(hand), realized), None

    if ctx.fourmax is None or reason not in FOURMAX_RETRY_REASONS:
        return None, reason
    fm_spot, fm_reason = find_fourmax_spot(hand, hero_name=ctx.hero_name)
    if fm_spot is None:
        return None, f"{reason}_then_fourmax_{fm_reason}"
    fm_cfg, fm_ranges = ctx.fourmax
    if backtest.bucket_depth(fm_spot.effective_stack_bb) != fm_cfg.stack_bb:
        return None, "fourmax_depth_not_solved"
    graded = backtest.grade_spot_against(
        fm_spot, fm_cfg, fm_ranges, ctx.solve_cache.table,
        evaluator=ctx.evaluator, model="4MAX", ev_memo=ctx.ev_memo,
    )
    realized = realized_spot_ev_net_bb(hand, fm_spot, fm_cfg, "4MAX", hero_name=ctx.hero_name)
    return GradedHand(graded, hand.bb_amount, hero_net_currency(hand), realized), None


def grade_hands(hands: list[ParsedHand], ctx: GradingContext) -> tuple[list[GradedHand], dict[str, int]]:
    rows: list[GradedHand] = []
    skips: dict[str, int] = {}
    for hand in hands:
        row, reason = _grade_one(hand, ctx)
        if row is not None:
            rows.append(row)
        else:
            skips[reason] = skips.get(reason, 0) + 1
    return rows, skips


def graded_in_window(rows: list[GradedHand], start_time: str, end_time: str) -> list[GradedHand]:
    """Rows whose hand timestamp falls inside [start_time, end_time], inclusive --
    mirrors sessions.hands_in_window (parse_timestamp normalizes the +08 suffix)."""
    start = parse_timestamp(start_time)
    end = parse_timestamp(end_time)
    return [r for r in rows if start <= parse_timestamp(r.graded.timestamp) <= end]


def is_match(g: GradedSpot) -> bool:
    """Deterministic accuracy test against the solved WEIGHT, never the EV
    comparison behind gto_recommended_action: a pure weight demands the matching
    action, a mixed weight endorses either. For HU (analytic EVs) the two agree at
    convergence, but a 4MAX Monte Carlo EV estimate can contradict a pure weight on
    a near-indifferent hand -- the weight is the solved artifact, so it wins."""
    if g.gto_weight >= MIXED_BAND[1]:
        return g.hero_action == "shove_or_call"
    if g.gto_weight <= MIXED_BAND[0]:
        return g.hero_action == "fold"
    return True


# ---------------------------------------------------------------------------------
# Window aggregation: one WindowAnalysis per logged session.
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class StakeGrade:
    """Per-(window, stake) sums -- bb is only well-defined within one stake."""

    bb_amount: float
    spots: int
    matched: int
    ev_lost_bb: float
    luck_spots: int  # spots (any model) with a realized leg, feeding the luck sums
    predicted_ev_bb: float
    realized_ev_bb: float


@dataclass(frozen=True)
class Mistake:
    hand_id: str
    timestamp: str
    bb_amount: float
    model: str
    position: str
    effective_stack_bb: float
    hero_hand: str
    hero_action: str
    gto_action: str
    gto_weight: float
    ev_loss_bb: float


@dataclass(frozen=True)
class WindowAnalysis:
    spots: int
    hu_spots: int
    fourmax_spots: int
    matched: int  # accuracy = matched / spots
    ev_lost_currency: float  # sum(ev_loss_bb * bb_amount) over ALL graded spots
    predicted_currency: float  # luck spots only (realized leg present), via * bb_amount
    realized_currency: float
    luck_gap_currency: float  # realized - predicted
    graded_net_currency: float  # ledger net over graded hands whose net is known
    stakes: tuple[StakeGrade, ...]
    top_mistakes: tuple[Mistake, ...]


def _stake_grade(bb_amount: float, group: list[GradedHand]) -> StakeGrade:
    luck = [r for r in group if r.realized_ev_net_bb is not None]
    return StakeGrade(
        bb_amount=bb_amount,
        spots=len(group),
        matched=sum(1 for r in group if is_match(r.graded)),
        ev_lost_bb=sum(r.graded.ev_loss_bb for r in group),
        luck_spots=len(luck),
        predicted_ev_bb=sum(r.graded.predicted_ev_net_bb for r in luck),
        realized_ev_bb=sum(r.realized_ev_net_bb for r in luck),
    )


def _mistake(row: GradedHand) -> Mistake:
    g = row.graded
    return Mistake(
        hand_id=g.hand_id,
        timestamp=g.timestamp,
        bb_amount=row.bb_amount,
        model=g.model,
        position=g.position,
        effective_stack_bb=g.effective_stack_bb,
        hero_hand=g.hero_hand,
        hero_action=g.hero_action,
        gto_action=g.gto_recommended_action,
        gto_weight=g.gto_weight,
        ev_loss_bb=g.ev_loss_bb,
    )


def analyze_window(rows: list[GradedHand], top_n: int = 5) -> WindowAnalysis | None:
    """None when the window has no graded spots (renders as 'no gradeable spots').
    Dedups by hand_id: the same hand appearing in two overlapping exports must
    count once (the existing perStake join has the same exposure, left as-is)."""
    unique = list({r.graded.hand_id: r for r in rows}.values())
    if not unique:
        return None

    luck = [r for r in unique if r.realized_ev_net_bb is not None]
    predicted = sum(r.graded.predicted_ev_net_bb * r.bb_amount for r in luck)
    realized = sum(r.realized_ev_net_bb * r.bb_amount for r in luck)

    by_stake: dict[float, list[GradedHand]] = {}
    for r in unique:
        by_stake.setdefault(r.bb_amount, []).append(r)

    # A "mistake" must both cost EV and contradict the equilibrium weight -- a
    # positive ev_loss_bb alone can be 4MAX MC noise on a correctly played hand.
    mistakes = sorted(
        (r for r in unique if r.graded.ev_loss_bb > EV_LOSS_EPSILON and not is_match(r.graded)),
        key=lambda r: -r.graded.ev_loss_bb,
    )
    return WindowAnalysis(
        spots=len(unique),
        hu_spots=sum(1 for r in unique if r.graded.model == "HU"),
        fourmax_spots=sum(1 for r in unique if r.graded.model == "4MAX"),
        matched=sum(1 for r in unique if is_match(r.graded)),
        ev_lost_currency=sum(r.graded.ev_loss_bb * r.bb_amount for r in unique),
        predicted_currency=predicted,
        realized_currency=realized,
        luck_gap_currency=realized - predicted,
        graded_net_currency=sum(r.net_currency for r in unique if r.net_currency is not None),
        stakes=tuple(_stake_grade(bb, group) for bb, group in sorted(by_stake.items())),
        top_mistakes=tuple(_mistake(r) for r in mistakes[:top_n]),
    )


# ---------------------------------------------------------------------------------
# Persisted per-file grade cache: cache/session_grades/<source stem>.json, keyed by
# (source file mtime, GRADING_VERSION). Persistence (vs in-memory only) is what
# makes the static exporter and server restarts instant -- same reason equity.py
# and backtest's fourmax pickle persist under cache/.
# ---------------------------------------------------------------------------------


def grades_cache_path(cache_dir: str, source_path: str) -> Path:
    return Path(cache_dir) / "session_grades" / f"{Path(source_path).stem}.json"


def _row_to_dict(row: GradedHand) -> dict:
    return {
        "bb_amount": float(row.bb_amount),
        "net_currency": None if row.net_currency is None else float(row.net_currency),
        "realized_ev_net_bb": None if row.realized_ev_net_bb is None else float(row.realized_ev_net_bb),
        "graded": dataclasses.asdict(row.graded),
    }


def _row_from_dict(d: dict) -> GradedHand:
    return GradedHand(
        graded=GradedSpot(**d["graded"]),
        bb_amount=d["bb_amount"],
        net_currency=d["net_currency"],
        realized_ev_net_bb=d["realized_ev_net_bb"],
    )


def _load_grades_if_fresh(path: Path, source_mtime: float) -> list[GradedHand] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != GRADING_VERSION or data.get("source_mtime") != source_mtime:
            return None
        return [_row_from_dict(d) for d in data["rows"]]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None  # corrupt/stale-shaped file -> regrade and rewrite it


def load_or_grade_file(
    source_path: str, hands: list[ParsedHand], ctx: GradingContext, cache_dir: str = "cache/"
) -> list[GradedHand]:
    """Graded rows for one source export, served from the persisted JSON when its
    (mtime, version) key still matches, else regraded from `hands` and rewritten."""
    path = grades_cache_path(cache_dir, source_path)
    source_mtime = Path(source_path).stat().st_mtime
    cached = _load_grades_if_fresh(path, source_mtime)
    if cached is not None:
        return cached

    rows, _skips = grade_hands(hands, ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": GRADING_VERSION, "source_mtime": source_mtime, "rows": [_row_to_dict(r) for r in rows]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return rows
