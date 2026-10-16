"""Draft state: who owns each pick, what has been picked, and every team's roster."""

from __future__ import annotations

from dataclasses import dataclass, field

from draftai.draft.order import pick_owners, picks_for


@dataclass
class DraftState:
    teams: int
    rounds: int
    owners: list[int]  # owning slot of each overall pick
    my_slot: int
    picks: list[str] = field(default_factory=list)  # Sleeper player ids, in pick order
    pick_slots: list[int] = field(default_factory=list)  # slot that made each pick

    @classmethod
    def new(cls, teams: int, rounds: int, my_slot: int, draft_type: str = "snake", reversal_round: int = 0,
            traded: list[tuple[int, int, int]] | None = None) -> "DraftState":
        return cls(teams, rounds, pick_owners(teams, rounds, draft_type, reversal_round, traded or []), my_slot)

    # --- progress ------------------------------------------------------------------
    @property
    def version(self) -> int:
        """Changes whenever a pick is made; a recommendation for an older version is stale."""
        return len(self.picks)

    @property
    def current_pick(self) -> int:
        return len(self.picks) + 1

    @property
    def total_picks(self) -> int:
        return len(self.owners)

    @property
    def complete(self) -> bool:
        return self.current_pick > self.total_picks

    @property
    def on_the_clock(self) -> int | None:
        return None if self.complete else self.owners[self.current_pick - 1]

    @property
    def my_turn(self) -> bool:
        return self.on_the_clock == self.my_slot

    def my_picks(self) -> list[int]:
        return picks_for(self.owners, self.my_slot)

    def my_upcoming(self) -> list[int]:
        return [p for p in self.my_picks() if p >= self.current_pick]

    def next_pick_after(self, pick: int, slot: int | None = None) -> int | None:
        slot = self.my_slot if slot is None else slot
        later = [p for p in picks_for(self.owners, slot) if p > pick]
        return later[0] if later else None

    def picks_until_mine(self) -> int | None:
        up = self.my_upcoming()
        return None if not up else up[0] - self.current_pick

    # --- rosters -------------------------------------------------------------------
    def roster(self, slot: int | None = None) -> list[str]:
        slot = self.my_slot if slot is None else slot
        return [p for p, s in zip(self.picks, self.pick_slots) if s == slot]

    def rosters(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {s: [] for s in range(1, self.teams + 1)}
        for p, s in zip(self.picks, self.pick_slots):
            out.setdefault(s, []).append(p)
        return out

    @property
    def drafted(self) -> set[str]:
        return set(self.picks)

    def add_pick(self, player_id: str, slot: int | None = None) -> None:
        if self.complete:
            raise ValueError("draft is complete")
        if player_id in self.drafted:
            raise ValueError(f"player {player_id} already drafted")
        owner = self.owners[self.current_pick - 1] if slot is None else slot
        self.picks.append(player_id)
        self.pick_slots.append(owner)

    def undo(self) -> None:
        if self.picks:
            self.picks.pop()
            self.pick_slots.pop()

    def copy(self) -> "DraftState":
        return DraftState(self.teams, self.rounds, list(self.owners), self.my_slot, list(self.picks), list(self.pick_slots))
