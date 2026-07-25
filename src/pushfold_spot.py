"""Push/fold spot detector: ParsedHand -> a gradable HU PushFoldSpot, or a skip
reason. No text-format knowledge -- that's hand_history.py's job.

A spot is gradable only when Hero is one of the two named blind-posters (ground
truth from the actual "posts small/big blind" lines -- never inferred from seat
numbers, since sit-outs create gaps) and, at the moment Hero must decide, exactly
one opponent is contesting the pot. Everything else (antes, non-all-in raises,
4-max shoves with 2+ live responders) is out of scope for the current HU-only
solver and is skipped rather than misgraded -- see ev.py's own NotImplementedError
guards on open_shove_ev_net_vec/call_ev_net_vec for the same boundary.

Known, deliberate approximation: if a third seat folds its blind before the final
two-way confrontation (e.g. the SB-poster folds and a non-blind seat's shove is
what Hero, as BB-poster, actually faces), that seat's forfeited blind is not added
into the graded info set's dead money -- doing so exactly would require solving a
real 3-seat subgame, out of scope here. This only affects hands where the shover
facing Hero isn't literally the SB-poster.
"""

from __future__ import annotations

from dataclasses import dataclass

from hand_history import ParsedHand

MAX_PUSHFOLD_STACK_BB = 20.0  # above this, a "fold to the blinds" is an ordinary deep-
# stack lay-down with a full betting spectrum available, not a real push/fold
# decision -- structurally identical to a genuine spot (action folds to Hero as SB
# with one live opponent) but strategically meaningless to grade against a
# shove-or-fold-only Nash model. Confirmed against real data: genuine push/fold
# hands cluster at 8bb (one at 7bb); every spot above this line was an incidental
# fold in an unrelated deep-stack cash game.

SKIP_REASONS = {
    "ante_present": "ante_bb is declared in GameConfig but never consumed by ev.py/tree.py -- ante hands are not modeled",
    "missing_blind_posts": "could not identify both a small-blind and big-blind poster line",
    "hero_not_blind_poster": "hero was not the SB-poster or BB-poster of the two blind seats in this hand",
    "non_allin_action": "a preflop check/bet/limp-call/non-all-in-raise occurred -- not a push/fold spot",
    "shove_multiway": "more than one seat was still live behind the shover at the moment of the shove -- not reducible to the 2-seat HU game",
    "call_multiway": "more than one opponent was live when hero had to respond -- not reducible to the 2-seat HU game",
    "no_gradable_decision": "walked the whole preflop action list without hero facing a graded fold/shove/call",
    "stack_too_deep_for_pushfold": f"effective_stack_bb exceeded {MAX_PUSHFOLD_STACK_BB}bb -- not a real push/fold decision",
}


@dataclass(frozen=True)
class PushFoldSpot:
    hand_id: str
    timestamp: str
    info_set_key: str  # "SB|" or "BB|SB"
    hero_hand: tuple[str, str]
    effective_stack_bb: float
    hero_action: str  # "fold" | "shove_or_call"


def find_spot(hand: ParsedHand, hero_name: str = "Hero") -> tuple[PushFoldSpot | None, str | None]:
    if hand.ante_posters:
        return None, "ante_present"
    if hand.sb_poster is None or hand.bb_poster is None:
        return None, "missing_blind_posts"

    stack_of = {s.name: s.starting_stack for s in hand.seats}
    live = set(hand.dealt_names)
    all_in_actors: list[str] = []

    def _spot(info_set_key: str, opponent: str, hero_action: str) -> tuple[PushFoldSpot | None, str | None]:
        eff_stack = min(stack_of[hero_name], stack_of[opponent]) / hand.bb_amount
        if eff_stack > MAX_PUSHFOLD_STACK_BB:
            return None, "stack_too_deep_for_pushfold"
        return PushFoldSpot(hand.hand_id, hand.timestamp, info_set_key, hand.hero_cards, eff_stack, hero_action), None

    def _hero_opens(hero_action: str) -> tuple[PushFoldSpot | None, str | None]:
        if hero_name == hand.sb_poster and len(live) == 2:
            other = next(iter(live - {hero_name}))
            return _spot("SB|", other, hero_action)
        reason = "hero_not_blind_poster" if hero_action == "fold" else "shove_multiway"
        return None, reason

    def _hero_responds_to_shove(hero_action: str) -> tuple[PushFoldSpot | None, str | None]:
        if len(all_in_actors) > 1:
            return None, "call_multiway"
        shover = all_in_actors[0]
        if hero_name != hand.bb_poster:
            return None, "hero_not_blind_poster"
        if live != {shover, hero_name}:
            return None, "call_multiway"
        return _spot("BB|SB", shover, hero_action)

    for action in hand.preflop_actions:
        name = action.name
        stack = stack_of.get(name)

        if action.action == "return":
            continue
        if action.action in ("checks", "bets", "post"):
            return None, "non_allin_action"

        is_full_shove = action.action == "allin" or (
            action.action == "raises"
            and stack is not None
            and action.to_amount is not None
            and action.to_amount >= stack - 0.01
        )
        is_non_allin_raise = action.action == "raises" and not is_full_shove
        is_plain_call = action.action == "calls" and not all_in_actors
        is_closing_call = action.action == "calls" and bool(all_in_actors)

        if is_non_allin_raise or is_plain_call:
            return None, "non_allin_action"

        if action.action == "folds":
            if name == hero_name:
                if not all_in_actors:
                    return _hero_opens("fold")
                elif len(all_in_actors) == 1:
                    return _hero_responds_to_shove("fold")
                else:
                    return None, "call_multiway"
            live.discard(name)
            continue

        if is_full_shove:
            if name == hero_name:
                if not all_in_actors:
                    return _hero_opens("shove_or_call")
                else:
                    return _hero_responds_to_shove("shove_or_call")
            all_in_actors.append(name)
            continue

        if is_closing_call:
            if name == hero_name:
                return _hero_responds_to_shove("shove_or_call")
            continue

    return None, "no_gradable_decision"


def _is_full_shove(action, stack_of: dict[str, float]) -> bool:
    stack = stack_of.get(action.name)
    return action.action == "allin" or (
        action.action == "raises" and stack is not None and action.to_amount is not None and action.to_amount >= stack - 0.01
    )


def classify_sb_open_when_hero_is_bb(hand: ParsedHand, hero_name: str = "Hero") -> str | None:
    """Mirrors find_spot's _hero_opens, but reports the literal SB-poster's OWN
    open decision (shove or fold) instead of grading Hero's -- only meaningful
    when Hero is the BB-poster, watching what the SB-poster actually does in a
    clean 2-seat spot. Doesn't need the SB-poster's hole cards, only their action
    label, so it measures real opponents' shove frequency against the solved
    chart's own shove frequency (cards.range_average_weight(ranges["SB|"])),
    independent of Hero's own play. Returns "shove" | "fold" | None (not a clean,
    in-scope 2-seat spot for this check)."""
    if hand.ante_posters or hand.sb_poster is None or hand.bb_poster is None:
        return None
    if hero_name != hand.bb_poster:
        return None

    stack_of = {s.name: s.starting_stack for s in hand.seats}
    live = set(hand.dealt_names)

    for action in hand.preflop_actions:
        name = action.name
        if action.action == "return":
            continue
        if action.action in ("checks", "bets", "post"):
            return None
        is_full_shove = _is_full_shove(action, stack_of)
        is_non_allin_raise = action.action == "raises" and not is_full_shove
        is_plain_call = action.action == "calls"
        if is_non_allin_raise or is_plain_call:
            return None

        if name == hand.sb_poster:
            if live != {hand.sb_poster, hero_name}:
                return None
            if action.action == "folds":
                return "fold"
            if is_full_shove:
                return "shove"
            return None

        if action.action == "folds":
            live.discard(name)
            continue
        # anyone other than the SB-poster going all-in before SB's own turn means
        # this isn't the SB-poster's own clean open decision.
        return None

    return None


def classify_bb_response_when_hero_is_sb(hand: ParsedHand, hero_name: str = "Hero") -> str | None:
    """Mirrors find_spot's _hero_opens shove branch, but reports the literal
    BB-poster's OWN response (call or fold) instead of grading Hero's -- only
    meaningful when Hero is the SB-poster and shoves in a clean 2-seat spot.
    Measures real opponents' calling frequency against the solved chart's own call
    frequency (cards.range_average_weight(ranges["BB|SB"])), independent of Hero's
    own play. Returns "call" | "fold" | None (not a clean, in-scope 2-seat spot)."""
    if hand.ante_posters or hand.sb_poster is None or hand.bb_poster is None:
        return None
    if hero_name != hand.sb_poster:
        return None

    stack_of = {s.name: s.starting_stack for s in hand.seats}
    live = set(hand.dealt_names)
    hero_shoved = False

    for action in hand.preflop_actions:
        name = action.name
        if action.action == "return":
            continue

        if hero_shoved:
            if name != hand.bb_poster:
                return None
            if action.action == "folds":
                return "fold"
            if action.action in ("calls", "allin") or _is_full_shove(action, stack_of):
                return "call"
            return None

        if action.action in ("checks", "bets", "post"):
            return None
        is_full_shove = _is_full_shove(action, stack_of)
        is_non_allin_raise = action.action == "raises" and not is_full_shove
        is_plain_call = action.action == "calls"
        if is_non_allin_raise or is_plain_call:
            return None

        if name == hero_name:
            if not is_full_shove or live != {hero_name, hand.bb_poster}:
                return None
            hero_shoved = True
            continue

        if action.action == "folds":
            live.discard(name)
            continue
        return None

    return None
