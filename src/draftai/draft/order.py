"""Pick order from the draft's settings, never from a hard-coded snake formula.

Teams are identified by draft slot (1..teams). Sleeper's ``reversal_round`` (3 for
third-round reversal) flips the direction of that round and every round after it, and
traded picks reassign individual (round, original slot) picks to a new owner.
"""

from __future__ import annotations

from typing import Iterable


def round_reversed(rnd: int, draft_type: str = "snake", reversal_round: int = 0) -> bool:
    if draft_type == "linear":
        return False
    rev = rnd % 2 == 0
    if reversal_round and rnd >= reversal_round:
        rev = not rev
    return rev


def pick_order(teams: int, rounds: int, draft_type: str = "snake", reversal_round: int = 0) -> list[tuple[int, int]]:
    """(round, original slot) for every overall pick, in order."""
    out = []
    for rnd in range(1, rounds + 1):
        slots = range(teams, 0, -1) if round_reversed(rnd, draft_type, reversal_round) else range(1, teams + 1)
        out.extend((rnd, s) for s in slots)
    return out


def pick_owners(
    teams: int,
    rounds: int,
    draft_type: str = "snake",
    reversal_round: int = 0,
    traded: Iterable[tuple[int, int, int]] = (),
) -> list[int]:
    """Owning slot for each overall pick. ``traded`` holds (round, original slot, new slot)."""
    moves = {(r, s): new for r, s, new in traded}
    return [moves.get((r, s), s) for r, s in pick_order(teams, rounds, draft_type, reversal_round)]


def picks_for(owners: list[int], slot: int) -> list[int]:
    """Overall pick numbers (1-based) owned by ``slot``."""
    return [i + 1 for i, o in enumerate(owners) if o == slot]


def windows(picks: list[int]) -> list[int]:
    """Opponent picks between consecutive picks of one team."""
    return [b - a - 1 for a, b in zip(picks, picks[1:])]
