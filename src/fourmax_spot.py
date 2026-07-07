"""4-max push/fold spot detector: ParsedHand -> a gradable PushFoldSpot against the
FOURMAX_CONFIG tree (14 info sets: UTG|, BTN|*, SB|*, BB|* -- see tree.py), or a
skip reason. Mirrors pushfold_spot.py's HU-only walk, generalized from "is hero one
of the two blinds" to "which of the 14 info sets does hero's decision land on."

Kept as a SEPARATE module from pushfold_spot.py rather than a single generalized
detector, because role assignment is genuinely config-shape-specific (HU only needs
the two blind-poster names; 4-max additionally needs the button seat and the
"leftover" UTG seat) -- see _assign_fourmax_roles. The action-walk itself, once
each dealt player is mapped to a role, is seat-count-agnostic and closely parallels
find_spot's, tracking which EARLIER roles (in FOURMAX_CONFIG.seats order) have
shoved by the time hero must decide.

Reuses pushfold_spot.PushFoldSpot as-is: the fields (hand_id, timestamp,
info_set_key, hero_hand, effective_stack_bb, hero_action) are identical, just with
info_set_key drawn from 14 possible keys instead of 2.

Effective stack: MIN starting stack across every seat actually dealt into the hand
(not just hero and one opponent, since 3+ players may be live) -- this is the real
money genuinely at risk in a multiway all-in (any excess from a bigger stack is
returned uncalled, exactly as in the HU case), and it's what GameConfig.stack_bb (a
single scalar for the whole table) must be solved at. Confirmed against the real
data this was built for: every multiway push/fold hand's minimum stack is exactly
8.0bb, matching HU_CONFIG's own default depth -- so in practice only one FOURMAX_CONFIG
solve (at 8.0bb) is needed to grade all of them, not a per-hand depth sweep.
"""

from __future__ import annotations

from game import FOURMAX_CONFIG
from hand_history import ParsedHand
from pushfold_spot import MAX_PUSHFOLD_STACK_BB, PushFoldSpot, _is_full_shove
from tree import infoset_key

ACTION_ORDER = FOURMAX_CONFIG.seats  # ("UTG", "BTN", "SB", "BB")

FOURMAX_SKIP_REASONS = {
    "ante_present": "ante_bb is not consumed by ev.py/tree.py -- ante hands are not modeled",
    "missing_blind_posts": "could not identify both a small-blind and big-blind poster line",
    "too_many_roles": "more than 4 distinct dealt-in seats -- doesn't reduce to the FOURMAX_CONFIG shape",
    "hero_role_unknown": "hero was not among the dealt-in, role-assigned seats",
    "non_allin_action": "a preflop check/bet/limp-call/non-all-in-raise occurred -- not a push/fold spot",
    "unknown_actor": "a preflop action came from a seat that couldn't be role-assigned",
    "no_gradable_decision": "walked the whole preflop action list without hero facing a graded fold/shove",
    "stack_too_deep_for_pushfold": f"effective_stack_bb exceeded {MAX_PUSHFOLD_STACK_BB}bb -- not a real push/fold decision",
}


def _assign_fourmax_roles(hand: ParsedHand) -> dict[str, str] | None:
    """name -> role ('UTG'/'BTN'/'SB'/'BB'), or None if this hand has more than the
    4 FOURMAX_CONFIG roles' worth of dealt-in seats. Button and blinds come
    straight from the parsed ground truth (never inferred from seat-number
    arithmetic, since sit-outs create gaps -- same principle as pushfold_spot.py).
    Any dealt-in seat that's neither the button nor a blind poster is UTG; there
    can be at most one such seat (3-4 handed only) for this detector to apply.
    A 3-handed table missing a UTG seat entirely is handled naturally: UTG is
    simply never assigned to anyone and never appears in the action walk, which
    is exactly equivalent to UTG having folded pre-hand with 0 dead money."""
    if hand.sb_poster is None or hand.bb_poster is None:
        return None
    button_name = next((s.name for s in hand.seats if s.seat_num == hand.button_seat), None)
    if button_name is None:
        return None

    roles: dict[str, str] = {hand.sb_poster: "SB", hand.bb_poster: "BB"}
    if button_name not in roles:
        roles[button_name] = "BTN"

    leftover = [n for n in hand.dealt_names if n not in roles]
    if len(leftover) > 1:
        return None
    if len(leftover) == 1:
        roles[leftover[0]] = "UTG"
    return roles


def find_fourmax_spot(hand: ParsedHand, hero_name: str = "Hero") -> tuple[PushFoldSpot | None, str | None]:
    if hand.ante_posters:
        return None, "ante_present"
    if hand.sb_poster is None or hand.bb_poster is None:
        return None, "missing_blind_posts"

    roles = _assign_fourmax_roles(hand)
    if roles is None:
        return None, "too_many_roles"
    if hero_name not in roles:
        return None, "hero_role_unknown"

    stack_of = {s.name: s.starting_stack for s in hand.seats}
    dealt_stacks_bb = [stack_of[n] / hand.bb_amount for n in roles]
    effective_stack_bb = min(dealt_stacks_bb)
    if effective_stack_bb > MAX_PUSHFOLD_STACK_BB:
        return None, "stack_too_deep_for_pushfold"

    shoved_roles: set[str] = set()

    for action in hand.preflop_actions:
        name = action.name
        if action.action == "return":
            continue

        role = roles.get(name)
        if role is None:
            return None, "unknown_actor"

        if action.action in ("checks", "bets"):
            return None, "non_allin_action"

        is_full_shove = _is_full_shove(action, stack_of)
        is_non_allin_raise = action.action == "raises" and not is_full_shove
        is_plain_call = action.action == "calls" and not shoved_roles
        is_closing_call = action.action == "calls" and bool(shoved_roles)

        if is_non_allin_raise or is_plain_call:
            return None, "non_allin_action"

        if action.action == "folds":
            if name == hero_name:
                key = infoset_key(role, frozenset(shoved_roles))
                return PushFoldSpot(hand.hand_id, hand.timestamp, key, hand.hero_cards, effective_stack_bb, "fold"), None
            continue

        if is_full_shove or is_closing_call:
            if name == hero_name:
                key = infoset_key(role, frozenset(shoved_roles))
                return (
                    PushFoldSpot(hand.hand_id, hand.timestamp, key, hand.hero_cards, effective_stack_bb, "shove_or_call"),
                    None,
                )
            shoved_roles.add(role)
            continue

    return None, "no_gradable_decision"
