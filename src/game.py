"""Locked game parameters for the push/fold solver.

Every real-money rule from the spec (stakes, rake, splash fee, rakeback, seat order)
lives in one place: `GameConfig`. Nothing downstream should hardcode a literal like
0.22 or 0.06 -- it should read `cfg.drop_bb` / `cfg.rakeback_pool_bb` instead, so the
"adjustable parameters" promise stays true.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class GameConfig:
    # Ordered action sequence. ("SB", "BB") for heads-up; ("UTG", "BTN", "SB", "BB")
    # for 4-max. tree.py builds info sets purely from this tuple's length and order,
    # so adding/removing seats never requires touching tree.py itself.
    seats: tuple[str, ...]

    stack_bb: float = 8.0
    blinds: dict[str, float] = field(default_factory=lambda: {"SB": 0.5, "BB": 1.0})
    ante_bb: float = 0.0

    rake_bb: float = 0.12
    splash_fee_bb: float = 0.10
    rakeback_frac: float = 0.5  # fraction of rake_bb returned as the rakeback pool

    def __post_init__(self) -> None:
        if not self.seats:
            raise ValueError("seats must be non-empty")
        if len(set(self.seats)) != len(self.seats):
            raise ValueError(f"seats must be unique, got {self.seats}")
        unknown_blind_seats = set(self.blinds) - set(self.seats)
        if unknown_blind_seats:
            raise ValueError(f"blinds reference seats not in seats: {unknown_blind_seats}")
        if any(b < 0 for b in self.blinds.values()):
            raise ValueError(f"blinds must be non-negative, got {self.blinds}")
        if self.stack_bb <= max(self.blinds.values(), default=0.0):
            raise ValueError("stack_bb must exceed every blind")
        if not 0.0 <= self.rakeback_frac <= 1.0:
            raise ValueError(f"rakeback_frac must be in [0, 1], got {self.rakeback_frac}")
        if self.rake_bb < 0 or self.splash_fee_bb < 0:
            raise ValueError("rake_bb and splash_fee_bb must be non-negative")

    @property
    def drop_bb(self) -> float:
        """Flat amount removed from every pot before award.

        The splash fee's redistribution is exogenous (independent of any player's
        strategy at this table), so it cannot change the *relative* EV of shoving vs.
        folding. Folding it into rake as one flat drop is an EXACT simplification for
        range-solving, not an approximation.
        """
        return self.rake_bb + self.splash_fee_bb

    @property
    def rakeback_pool_bb(self) -> float:
        """Total rakeback paid out per pot, split pro-rata by contribution (see ev.py)."""
        return self.rakeback_frac * self.rake_bb

    def blind_of(self, seat: str) -> float:
        """0 for seats that post no blind (e.g. UTG/BTN), else the posted blind."""
        return self.blinds.get(seat, 0.0)

    def rake_free(self) -> "GameConfig":
        """This config with rake/splash/rakeback all zeroed out -- used as the baseline
        for the Gate 6 directional sanity check (real-drop ranges should come out
        monotonically tighter than this)."""
        return replace(self, rake_bb=0.0, splash_fee_bb=0.0, rakeback_frac=0.0)


HU_CONFIG = GameConfig(seats=("SB", "BB"))

# Built/enumerated this session (tree.py), not solved or validated -- see README scope note.
FOURMAX_CONFIG = GameConfig(seats=("UTG", "BTN", "SB", "BB"))
