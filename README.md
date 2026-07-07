# Push/Fold Solver

A game-theory-optimal (GTO) push/fold solver for 4-max all-in-or-fold No-Limit Hold'em,
under a specific real-money rake/fee structure (CoinPoker: 0.12bb rake + 0.10bb splash fee
+ 0.06bb rakeback pool, every pot). Output: per-position shove/call ranges as 13x13
hand-grid heatmaps.

**Scope: both heads-up (2-player) and full 4-max are solvable.** HU was built and validated
first, since it has published Nash charts to check against and 4-max doesn't — see
"Convergence honesty" below. The two share every module except `src/ev.py`/`src/solver.py`'s
EV computation, which is fully general over any number of live opponents and any number of
seats still to act (see `ev._enumerate_realization_leaves`), not HU-specific logic with a
4-max case bolted on.

## Why these modeling choices

**Why 8bb fixed.** 8bb effective is the deep-relative-to-blinds regime where push/fold *is*
the whole game — no postflop play to model. This is a snapshot for this stack depth, not a
claim that real stacks never move; deeper stacks would need a full postflop solver, not
this one.

**Why the splash fee is exact, not an approximation.** The 0.10bb splash fee is skimmed
from every pot and redistributed later into unrelated, random "splash" pots. That
redistribution is exogenous — it doesn't depend on any player's strategy at this table — so
it cannot change the *relative* EV of shoving vs. folding. Folding it into a flat 0.22bb
drop alongside rake is therefore an exact simplification for range-solving, not a modeling
shortcut (see `GameConfig.drop_bb` in `src/game.py`).

**Why rakeback is in the objective despite being paid separately.** Rakeback is credited to
a player's account outside the hand (it never touches the in-hand chip ledger — verified
against real hand-history fixtures in `tests/test_ev.py`), but it's proportional to
contribution, and a player only contributes by entering the pot. That makes it
strategy-dependent, so it belongs in the EV objective the solver optimizes even though it's
excluded from the "ledger" net that matches what actually appears at the table. See
`src/ev.py` for the `ledger_net` vs `ev_net` split — conflating the two is the easiest way
to silently build a solver that optimizes the wrong number.

**Why folding pays slightly more than the flat blind amount.** A folded blind isn't lost to
the void -- it's dead money left in a pot that still gets raked, so the folder earns a tiny
pro-rata rakeback rebate on it. For heads-up specifically every fold is immediately
terminal (the hand just ends), so this resolves to exact constants rather than an
approximation: `SB` folding nets **-0.44bb**, not the naive -0.5bb; `BB` folding nets
**-0.94bb**, not -1.0bb (see `ev.fold_baseline_ev_net`, derived from the same `settle()`
function the golden fixtures are checked against, not a separate hardcoded table).

**Convergence honesty.** Damped best-response iteration finds *an* equilibrium; Nash
existence is guaranteed, but uniqueness is **not** guaranteed once 3+ players are involved.
This is exactly why heads-up — which reduces to a clean two-range fixed point with a known
threshold structure — is built and validated first, and why 4-max convergence is
deliberately not attempted until this HU core has been reviewed. Two implementation details
worth knowing if you extend this: (1) a *constant* damping factor was found empirically to
lock into a stable 2-cycle around hands that are genuinely indifferent at equilibrium
(oscillating forever, e.g. ~0.44/~0.55) rather than converging -- `solver.SolverConfig.alpha`
decays over iterations for exactly this reason, the same diminishing-step-size idea behind
classical fictitious play; (2) range-change alone isn't a safe convergence test once alpha
decays, since changes shrink near zero simply because the step size did, even while the
strategy is still far from a real fixed point -- `solve()` additionally requires
exploitability (the real "does anyone want to deviate" question) below its own threshold.

**Why 4-max needs Monte Carlo.** Every info set with 2+ live opponents (e.g. BB facing three
shovers, or UTG blending over every combination of who among BTN/SB/BB also ends up shoving)
needs a genuine JOINT equity across 3-4 hands. Pairwise numbers from the exact 169x169 table
don't compose into that — they can't capture correlation between multiple opponents' hands — so
those showdowns go through `equity.multiway_equity_mc` (sampling, not full enumeration). Measured
here: a full `multiway_equity_mc` call costs ~0.13ms/iteration, and exactly 16 of the leaves
across the 14-info-set tree need one per sweep. This makes 4-max a **stochastic** solve: every
best response is noisy, unlike HU's exact-table best response. That difference is not cosmetic —
it breaks two things that work fine for HU, and the fixes are the interesting part.

**What the noise breaks, and how it's actually fixed (the honest 4-max convergence story).**
The first real 4-max run looked converged (range-change ~2e-4) but was still ~0.1bb exploitable.
Chasing that down produced a genuinely instructive result — including a wrong first guess, kept
here because the reasoning is the point:

- *First guess (wrong): "the step size froze, and the final iterate is a noisy snapshot."*
  Plausible — the decaying `alpha` had shrunk to ~1e-4 — so the fix seemed to be flooring the
  step size (`alpha_min`) and returning a **Polyak–Ruppert tail average** instead of the final
  iterate. But flooring `alpha` made it *worse* (~0.21bb: a constant floor holds a wide, biased
  orbit and breaks the averaging theory, which needs the step → 0), and averaging changed almost
  nothing: the averaged and current-iterate exploitability were identical. **That identity is the
  clue** — averaging removes *variance*; if it does nothing, the residual isn't variance, it's a
  *bias* in where the dynamics settle.
- *The real cause: quantal-response bias from per-sweep MC noise.* Each sweep's best response is
  a hard shove/fold threshold applied to a **noisy** Monte Carlo EV. The expected update is
  therefore toward `E[threshold(trueEV + noise)]` — a *smoothed* (logit-like) response, i.e. a
  quantal-response equilibrium sitting ~0.1bb off the true Nash. It's a bias, not variance, so no
  amount of averaging or step-size tuning removes it; it only shrinks as the per-sweep noise does.
- *The fix: raise the per-sweep MC count.* Measured directly by refining the frozen ranges at
  increasing per-sweep mc: exploitability fell 0.16 → 0.045 → 0.020 at mc = 500 → 2000 → 8000,
  tracking the predicted ~`1/√mc`. So the lever was never the step schedule or the estimator —
  it was equity precision inside each best response. The production run (`scripts/solve_fourmax.py`)
  warm-starts from the earlier approximate ranges (standard equilibrium refinement — cheap, since
  a few hundred high-mc sweeps suffice) and runs at high mc; tail averaging is kept but plays only
  a minor variance-cleanup role, not the starring one first assumed.
- *The metric needed fixing too.* Exploitability takes `max(ev_shove, fold)` over a noisy
  `ev_shove`, so it's upward-biased by ~`1/√mc` (Jensen). Re-measuring the *same* frozen ranges
  gave ~0.65 at mc=150, ~0.20 at mc=800, ~0.10 at mc=12000 — the low-mc "0.003 target" was never
  reachable. So the final exploitability is measured once at high mc (`final_exploitability_mc`)
  and again at 2× that, exposing the residual **noise floor** instead of trusting one number.

This is validated bottom-up: the machinery reproduces the known-good HU equilibrium (HU is
deterministic, the correctness anchor) to ~0.1% before it's trusted on 4-max. 4-max real-solve
tests are `@pytest.mark.slow` (see `pytest.ini`); `scripts/solve_fourmax.py` runs the full solve —
checkpointed and auto-resuming, since it's a multi-hour job.

*Result:* the production solve (warm-started, 600 sweeps at mc=20000, ~7.25h) reached an
exploitability of **~0.003–0.005bb** — measured at mc=80000 and again at mc=160000, where the
lower second reading confirms both numbers are near the measurement noise floor, i.e. the true
gap is ≲0.003bb. That's ~20–30× tighter than the first (frozen) run's 0.1bb, and comfortably in
"solved" territory. Notably the *aggregate* shove percentages barely moved from the frozen run —
the shape was already right; what the high-mc refinement fixed was the precise mixed weights on
the near-indifferent boundary hands, which is exactly what exploitability is sensitive to.

**One residual approximation, stated plainly.** `equity.multiway_equity_mc` evaluates one
representative combo per canonical hero hand, so there's a small systematic bias that does *not*
shrink with more MC samples (unlike the quantal-response bias above, which does). It's second-order
and left documented, not hidden — removing it would mean sampling over hero's combos too.

**The cross-cutting theme.** The solved ranges depend on the *real* drop being charged, and
come out measurably tighter than rake-free Nash (see the monotonicity check in
`tests/test_solver.py`, comparing a real-drop solve against a zero-rake solve of the same
config). The number this solver actually optimizes — chip EV under a specific fee structure
— is not the textbook number.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`eval7` (C-backed 7-card evaluator) is the primary dependency; if it fails to build on your
platform, `src/equity.py` falls back to the pure-Python `treys` evaluator behind the same
`Evaluator` interface — the rest of the code doesn't care which is active.

## Usage

```
python scripts/build_equity_cache.py     # one-time, ~2.5-4 hr: builds cache/equity_169x169_*.npy
python scripts/solve.py --config hu       # solves HU, writes heatmap PNGs to cache/ (seconds)
caffeinate -i python scripts/solve_fourmax.py   # 4-max: checkpointed, auto-resuming, multi-hour+
python scripts/export_range_viewer.py     # -> cache/range_viewer.html, interactive 14-tab viewer
```

The 4-max run is a separate, long, stochastic job (see the convergence section above); its
runner checkpoints to `cache/fourmax_checkpoint.pkl` and auto-resumes, so if it's interrupted
just rerun the same command.

The cache build computes EXACT equity for every canonical hand pair, averaged over
every valid specific-combo pairing (not one arbitrarily-chosen representative combo) --
see `equity._suit_partition_signature` for why that's tractable (a suit-relabeling
symmetry collapses ~808k raw combo-pairs to ~50k distinct computations) and
`equity.Eval7Evaluator`/`_exact_equity_one_combo` for why hole cards and the deck are
converted to evaluator-native objects once per pair rather than once per board (the
naive per-board conversion was measured to cost ~3-4x).

## Backtesting

```
python scripts/backtest.py --hand-history data/handhistory.txt
```

Parses a CoinPoker hand-history `.txt` export, detects the hands where Hero actually
faced a genuine heads-up push/fold decision (Hero at one of the two blind seats, action
reduced to exactly one live opponent), solves the HU equilibrium at each distinct
effective-stack depth encountered, grades every one of Hero's fold/shove/call decisions
against it, and reports the EV cost in bb of every deviation. Prints an aggregate summary
(skip-reason breakdown, % matching GTO exactly, total/average EV loss, worst individual
deviations) and writes a per-hand CSV (`--out-csv`, default `cache/backtest_results.csv`).

Known, deliberate limitations — see `src/pushfold_spot.py`'s module docstring for the
full reasoning:
- Ante hands are skipped (`GameConfig.ante_bb` exists but isn't consumed by `ev.py`).
- Hands where Hero shoves/faces a shove with 2+ live opponents still behind are skipped
  (not reducible to the 2-seat HU game the solver here targets, even though 4-max itself
  is solvable — see the Scope note above; extending the backtest to grade those spots
  against a real 4-max solve is a natural next step, not attempted here).
- Spots where the effective stack exceeds `pushfold_spot.MAX_PUSHFOLD_STACK_BB` (20bb) are
  skipped — a fold to the blinds at 100bb+ is an ordinary deep-stack lay-down with a full
  betting spectrum available, not a real push/fold decision, even though it's
  structurally identical to a genuine spot.

## Session tracking

```
python scripts/session.py start          # stage a live session (prompts for the client's numbers)
python scripts/session.py end            # complete it (prompts again, appends to data/sessions.csv)
python scripts/session.py add ...        # record a finished session non-interactively (see --help)
python scripts/session.py report --curve --hand-history data/export.txt
```

Sessions are **account-level** records (multitabling mixes 2-3 stakes, so a balance delta
can't be attributed per stake): main balance + the separate claimable rakeback balance at
both boundaries, plus any mid-session claims/deposits/withdrawals. That split is what
separates `table_result` from `rakeback_earned` — a bare balance delta conflates them the
moment rakeback is claimed. Optionally record the client's mission/leaderboard "rake paid"
counter at both ends: each such session then yields an estimate of the **effective rakeback
rate ρ** (rakeback earned per unit of attributed rake), the empirical input the fee model
needs. Per-stake stats (hands, net, bb/100) come exclusively from `--hand-history`, which
joins each session's time window against the export, grouped by stake; the report also
cross-checks the balance-derived table result against the HH-derived net (a data-entry and
export-coverage check). The ledger lives in `data/sessions.csv` (gitignored — personal
financial data); see `src/sessions.py` for the accounting identities.

## Layout

See `src/game.py`, `src/cards.py`, `src/equity.py`, `src/tree.py`, `src/ev.py`,
`src/solver.py`, `src/viz.py` — one module per concern, in dependency order. Every locked
game parameter (stakes, rake, splash, rakeback, seats) lives in `GameConfig`
(`src/game.py`), so changing any of them is a one-line edit, not a code change.

`src/hand_history.py` (raw CoinPoker `.txt` parsing), `src/pushfold_spot.py` (detecting
which parsed hands are gradable HU push/fold spots), and `src/backtest.py`
(depth-bucketed solving + EV-loss grading) implement the backtesting feature described
above, driven by the thin CLI in `scripts/backtest.py`.
