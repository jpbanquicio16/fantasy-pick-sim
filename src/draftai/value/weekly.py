"""Expected Lock-In value per player per Sleeper week.

    E[L_pk] = a_p * LockIn_p(team's games in week k, per-game miss probabilities)

a_p is the chance he is not on a multi-game absence; the per-game miss probability is
his short-absence (rest) rate, raised on the second night of a back-to-back. Lock-In
values are positively homogeneous (scaling every game by c scales the stopping value by
c), which lets season-level uncertainty be applied afterwards as a multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from draftai.value.lockin import EmpiricalScores, lockin_value


@dataclass(frozen=True)
class Availability:
    a: float  # probability of being available in a given week
    q: float  # per-game short-absence probability
    q_b2b: float  # on the second night of a back-to-back

    @classmethod
    def from_share(cls, share: float, q_short: float, b2b_mult: float, b2b_share: float = 0.17) -> "Availability":
        share = float(np.clip(share, 0.0, 1.0))
        q = float(np.clip(q_short, 0.0, 0.5))
        a = share / (1.0 - q) if q < 1 else 0.0
        if a > 1.0:  # projected to play more than his rest rate allows
            a, q = 1.0, 1.0 - share
        q_other = q / (b2b_share * b2b_mult + (1 - b2b_share))
        return cls(a=float(a), q=float(q_other), q_b2b=float(min(0.95, q_other * b2b_mult)))

    def game_q(self, pattern: tuple[bool, ...]) -> list[float]:
        return [self.q_b2b if b else self.q for b in pattern]


def weekly_values(
    dist: EmpiricalScores,
    avail: Availability,
    patterns: list[tuple[bool, ...]],
    cache: dict | None = None,
) -> np.ndarray:
    """Expected locked score in each week, given the team's game pattern per week."""
    cache = {} if cache is None else cache
    out = np.zeros(len(patterns))
    for k, pat in enumerate(patterns):
        if not pat:
            continue
        if pat not in cache:
            cache[pat] = lockin_value(dist, avail.game_q(pat))
        out[k] = avail.a * cache[pat]
    return out


def lockin_profile(dist: EmpiricalScores, max_games: int = 5) -> dict[int, float]:
    """Expected locked score for n = 1..max_games games and no missed games."""
    return {n: lockin_value(dist, [0.0] * n) for n in range(1, max_games + 1)}
