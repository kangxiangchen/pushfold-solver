"""Settlement math: the trickiest correctness surface in the whole solver.

Two distinct "net" quantities matter, and conflating them silently breaks things:

  ledger_net -- matches literal hand-history bookkeeping. EXCLUDES rakeback (it's a
      separate account credit, never touching the same-hand chip count). This is what
      the two golden fixtures (test_ev.py) check to the cent.
  ev_net = ledger_net + rakeback -- what a player actually optimizes when deciding
      shove/call, since rakeback is real, strategy-dependent money (you only earn it
      by contributing chips) even though it's paid out separately.

Every terminal node -- a genuine showdown OR any fold -- reduces to one formula once
you know (a) how many players are contesting the pot and for what equity share, and
(b) how many chips each seat actually leaves in the pot (an uncalled bet is simply
never part of it, so its owner's contribution is 0 there). `settle()` below is that
one formula; nothing else in this module hardcodes a separate showdown-vs-walk case.

For any info set with seats still left to act after it (every seat but the last),
computing a hand's EV requires knowing how the REST of the hand plays out -- some
subset of the remaining seats will also shove, in sequence, each conditioning on the
exact history realized so far. `_enumerate_realization_leaves` is the one place that
combinatorial branching happens; `fold_baseline_ev_net` and `shove_ev_net_vec` both
build directly on it, so the branching logic itself only has to be gotten right once.
"""

from __future__ import annotations

import numpy as np

import cards
import equity
import tree
from game import GameConfig
from tree import InfoSet, RangeMap


def settle(
    cfg: GameConfig, contributions: dict[str, float], equities: dict[str, float]
) -> dict[str, dict[str, float]]:
    """General terminal settlement.

    contributions[seat]: chips this seat leaves in the pot, NOT returned --
        cfg.stack_bb for a genuine all-in showdown participant; a forfeited blind_bb
        for a seat that folded; 0 for a seat whose bet/blind was never contested (the
        lone winner of a walk/steal never has anything "at risk", since an uncalled
        bet is returned before the pot is even formed -- this is what makes the
        formula below handle Fixture B without a special case).
    equities[seat]: this seat's equity share of the award, in [0, 1], summing to 1.0
        across all seats present (0 implicitly for any seat not in this dict, i.e. a
        pure folder who isn't contesting anything).

    ledger_net_i = equities.get(i, 0) * (P - drop_bb) - contributions[i]  for every
    seat i in `contributions`. This single line already reproduces: a showdown
    participant (contribution=stack_bb, equity=their real equity share); a folder
    (contribution=blind_bb, equity=0, so ledger_net = -blind_bb exactly); and an
    uncontested winner (contribution=0, equity=1.0, so ledger_net = P - drop_bb,
    i.e. exactly the spec's "D - drop" steal/walk formula, since P = D when nobody
    else has a live contribution).
    """
    total_pot = sum(contributions.values())
    result: dict[str, dict[str, float]] = {}
    for seat, contribution in contributions.items():
        award = equities.get(seat, 0.0) * (total_pot - cfg.drop_bb)
        ledger_net = award - contribution
        rakeback = cfg.rakeback_pool_bb * (contribution / total_pot) if total_pot > 0 else 0.0
        result[seat] = {
            "ledger_net": ledger_net,
            "rakeback": rakeback,
            "ev_net": ledger_net + rakeback,
        }
    return result


def _enumerate_realization_leaves(
    cfg: GameConfig, infoset: InfoSet, ranges: RangeMap, *, hero_shoves: bool
) -> list[tuple[float, frozenset[str], tuple[str, ...], dict[str, np.ndarray]]]:
    """The one place the "who among the remaining seats ends up live" branching
    happens. Returns a list of leaves, each `(probability, live_opponents,
    folded_after, opp_ranges)`:

      probability   -- P(this exact realization), given the CURRENT ranges snapshot.
      live_opponents -- frozenset of seats other than `infoset.seat` who are genuinely
          all-in in this realization (from infoset.shoved_before, fixed, plus however
          many of the seats after infoset.seat also shove along this specific path).
      folded_after  -- seats after infoset.seat who fold along this path (their
          blinds become dead money).
      opp_ranges    -- {seat: ranges[that seat's OWN info-set key]} for every seat in
          live_opponents, i.e. their actual (current-iterate) shoving range, not just
          a probability -- needed downstream for the equity computation.

    `hero_shoves` selects which of the two callers is asking: True from
    shove_ev_net_vec (infoset.seat is committing right now, so it's added to the
    realized history from the start), False from fold_baseline_ev_net (infoset.seat
    never becomes live; only the seats strictly after it matter for what follows).

    Exactness argument: shove/fold are the only two actions, seats act in a fixed
    order, and each seat's action is drawn from *that seat's own range at the info
    set induced by the specific history realized so far* -- not some unconditional
    marginal frequency, which is exactly why 14 separate info sets exist per seat in
    the first place. By the chain rule for a sequential-move game, P(one specific
    realization) = product of each seat's real conditional shove/fold probability
    along that path, and enumerating every subset of `seats_after` is an exhaustive,
    disjoint partition of the continuation space. So a probability-weighted sum over
    these leaves is EXACT given the `ranges` snapshot -- the only approximation
    anywhere in the pipeline is `multiway_equity_mc`'s finite sampling, which enters
    strictly downstream of this function, per leaf.

    Two edge cases, both required to reproduce the two hand-verified HU numbers
    (-0.44 for SB folding, -0.94 for BB folding) as a regression check:

    (1) The last seat has no decision at all if nobody has shoved by the time play
        reaches it (tree.build_infosets excludes that empty-shoved_before info set
        entirely -- an all-fold walk is uncontested, not a choice). This can only
        arise here when hero_shoves=False and infoset.shoved_before is empty --
        when hero_shoves=True, infoset.seat is in the history from the start, so the
        last seat always has *something* to face. Handled by forcing a deterministic
        non-shove (not a fold -- nothing is forfeited) and moving on.
    (2) An uncalled bet is never part of the pot. On the fold-baseline side, if only
        ONE other seat ever ends up shoving after hero folds, that lone shover is
        *also* uncontested (everyone else, hero included, folded) and contributes 0,
        not their stack. A real multiway pot -- stacks genuinely committed -- only
        exists once 2+ *other* seats are simultaneously live.
    """
    seats_after = cfg.seats[infoset.seat_pos + 1 :]
    last_seat = cfg.seats[-1]

    fixed_opp_ranges: dict[str, np.ndarray] = {}
    for seat in sorted(infoset.shoved_before, key=cfg.seats.index):
        seat_pos = cfg.seats.index(seat)
        seat_shoved_before = frozenset(
            s for s in infoset.shoved_before if cfg.seats.index(s) < seat_pos
        )
        fixed_opp_ranges[seat] = ranges[tree.infoset_key(seat, seat_shoved_before)]

    initial_shoved = set(infoset.shoved_before)
    if hero_shoves:
        initial_shoved.add(infoset.seat)

    leaves: list[tuple[float, frozenset[str], tuple[str, ...], dict[str, np.ndarray]]] = []

    def walk(
        i: int,
        prob: float,
        shoved_so_far: set[str],
        folded_after: list[str],
        opp_ranges: dict[str, np.ndarray],
    ) -> None:
        if prob <= 0.0:
            return
        if i == len(seats_after):
            live_opponents = frozenset(shoved_so_far - ({infoset.seat} if hero_shoves else set()))
            leaves.append((prob, live_opponents, tuple(folded_after), dict(opp_ranges)))
            return

        seat = seats_after[i]
        if seat == last_seat and not shoved_so_far:
            # Edge case (1): nobody has shoved yet, so the last seat has no decision
            # to make -- it wins for free if the walk ends here, with nothing forfeited.
            walk(i + 1, prob, shoved_so_far, folded_after, opp_ranges)
            return

        key = tree.infoset_key(seat, frozenset(shoved_so_far))
        p_shove = cards.range_average_weight(ranges[key])

        if p_shove > 0.0:
            shoved_branch_ranges = dict(opp_ranges)
            shoved_branch_ranges[seat] = ranges[key]
            walk(i + 1, prob * p_shove, shoved_so_far | {seat}, folded_after, shoved_branch_ranges)
        if p_shove < 1.0:
            walk(i + 1, prob * (1.0 - p_shove), shoved_so_far, folded_after + [seat], opp_ranges)

    walk(0, 1.0, initial_shoved, [], fixed_opp_ranges)
    return leaves


def fold_baseline_ev_net(cfg: GameConfig, infoset: InfoSet, ranges: RangeMap) -> float:
    """EV-net of folding at `infoset`, for ANY info-set shape (HU's two trivial cases
    and every 4-max shape alike). Folding hero has zero equity in every possible
    continuation, so ledger_net is always exactly -blind_of(seat) regardless of what
    happens afterward -- only the rakeback term varies leaf to leaf, since it depends
    on the eventual pot size (more downstream shovers -> bigger pot -> smaller
    rakeback share of hero's own forfeited blind). So this is a probability-weighted
    average of rakeback over `_enumerate_realization_leaves`'s exhaustive leaf set,
    added to the fixed -blind_of(seat) ledger term.

    Verified by hand against the two HU numbers from the spec's reconciliation table:
    SB-folds (HU) = -0.44, BB-folds (HU) = -0.94 (both single-leaf cases, one of them
    exercising edge case (1) above).
    """
    this_blind = cfg.blind_of(infoset.seat)
    ev_total = 0.0
    for prob, live_others, folded_after, _opp_ranges in _enumerate_realization_leaves(
        cfg, infoset, ranges, hero_shoves=False
    ):
        num_live_others = len(live_others)
        dead_money = infoset.dead_money_bb + this_blind + sum(cfg.blind_of(s) for s in folded_after)
        # Edge case (2): a real multiway pot (stacks actually committed) only exists
        # once 2+ OTHER seats are simultaneously live; 0 or 1 live others means
        # whoever's left (if anyone) also won uncontested, so no stack is added.
        total_pot = dead_money + (num_live_others * cfg.stack_bb if num_live_others >= 2 else 0.0)
        rakeback = cfg.rakeback_pool_bb * (this_blind / total_pot) if total_pot > 0 else 0.0
        ev_total += prob * (-this_blind + rakeback)
    return ev_total


def showdown_ev_net_vec(
    cfg: GameConfig, dead_money_bb: float, eq_vector: np.ndarray, num_opponents: int = 1
) -> np.ndarray:
    """Vectorized EV-net of a genuine showdown among hero plus `num_opponents` other
    all-in seats (every one of them contributes their full stack -- no side pots,
    since every live seat commits exactly cfg.stack_bb), for every one of the 169
    hero hands at once, given their equity against the joint opponent distribution
    (`eq_vector`) and any already-dead money from earlier folds. `num_opponents=1`
    (the default) reproduces the original heads-up-only formula exactly, so every
    existing caller is unaffected by this generalization."""
    total_pot = (1 + num_opponents) * cfg.stack_bb + dead_money_bb
    award = eq_vector * (total_pot - cfg.drop_bb)
    ledger_net = award - cfg.stack_bb
    rakeback = cfg.rakeback_pool_bb * (cfg.stack_bb / total_pot)
    return ledger_net + rakeback


def steal_ev_net(cfg: GameConfig, total_dead_money_bb: float) -> float:
    """EV-net of winning uncontested (everyone else folds) -- a scalar, since it
    doesn't depend on the winner's specific hand. No rakeback term: the winner's own
    bet was never called, so their contribution to the formed pot is 0."""
    return total_dead_money_bb - cfg.drop_bb


def shove_ev_net_vec(
    cfg: GameConfig,
    infoset: InfoSet,
    ranges: RangeMap,
    table: np.ndarray,
    evaluator: equity.Evaluator | None = None,
    mc_iterations: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """EV-net of SHOVING, for every one of the 169 hero hands, at ANY info set --
    this single function replaces the old HU-only open_shove_ev_net_vec (exactly one
    seat left to act) and call_ev_net_vec (the last seat, no one left to act): both
    were special cases of the same underlying computation, which is why they've been
    unified. The real dispatch axis was never "open vs call", it's "how many
    realizations does `_enumerate_realization_leaves` produce" -- 1 trivial leaf when
    hero is the last seat, up to 8 when hero is UTG with three seats still to act.

    EV_shove(hand) = sum over leaves of P(leaf) * EV(hand | that exact realization),
    which is exact given the CURRENT ranges snapshot by the same chain-rule argument
    as `_enumerate_realization_leaves`'s docstring. Per leaf: 0 live opponents is a
    pure steal; exactly 1 live opponent is a pairwise showdown against the exact
    2-way equity table (cheap, exact); 2+ live opponents needs a genuine joint
    equity, which pairwise numbers can't compose into, so it goes through
    `equity.multiway_equity_mc` instead (the spec's own reasoning for why multiway
    needs Monte Carlo -- see equity.py).

    This reduces to the exact old formulas on HU's two info sets: SB's open has
    exactly one seat after it (BB), producing 2 leaves -- steal (BB never calls) and
    p_call-weighted showdown (BB calls with its current range) -- matching the old
    open_shove_ev_net_vec's blend termwise. BB's call has zero seats after it,
    producing exactly 1 trivial leaf with 1 live opponent (SB), reducing to the old
    call_ev_net_vec's pure showdown-vs-range computation.
    """
    total = np.zeros(cards.NUM_CANON_HANDS)
    for prob, live_opponents, folded_after, opp_ranges in _enumerate_realization_leaves(
        cfg, infoset, ranges, hero_shoves=True
    ):
        dead_money = infoset.dead_money_bb + sum(cfg.blind_of(s) for s in folded_after)
        num_opponents = len(live_opponents)

        if num_opponents == 0:
            leaf_ev = steal_ev_net(cfg, dead_money)  # scalar broadcasts over the 169-vector sum
        elif num_opponents == 1:
            (opp_range,) = opp_ranges.values()
            eq_vec = equity.hand_vs_range_equity_vectorized(opp_range, table)
            leaf_ev = showdown_ev_net_vec(cfg, dead_money, eq_vec, num_opponents=1)
        else:
            if evaluator is None:
                raise ValueError(
                    f"{tree.infoset_key(infoset.seat, infoset.shoved_before)} needs multiway "
                    f"equity ({num_opponents} live opponents) but no evaluator was supplied"
                )
            eq_vec = equity.multiway_equity_mc(
                list(opp_ranges.values()), evaluator, mc_iterations, rng=rng
            )
            leaf_ev = showdown_ev_net_vec(cfg, dead_money, eq_vec, num_opponents=num_opponents)

        total = total + prob * leaf_ev
    return total
