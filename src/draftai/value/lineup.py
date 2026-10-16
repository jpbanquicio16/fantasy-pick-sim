"""Lineup optimiser: assign players to starting slots to maximise value.

An assignment problem (players x slots) solved exactly with
``scipy.optimize.linear_sum_assignment``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

# Which Sleeper positions may fill each starting slot.
SLOT_ELIGIBILITY: dict[str, set[str]] = {
    "PG": {"PG"},
    "SG": {"SG"},
    "SF": {"SF"},
    "PF": {"PF"},
    "C": {"C"},
    "G": {"PG", "SG", "G"},
    "F": {"SF", "PF", "F"},
    "UTIL": {"PG", "SG", "SF", "PF", "C", "G", "F"},
}
NON_STARTING = {"BN", "IL", "IR", "TAXI"}
_NEG = -1e9


def starting_slots(roster_positions: Sequence[str]) -> list[str]:
    return [s for s in roster_positions if s not in NON_STARTING]


def eligible(positions: Sequence[str], slot: str) -> bool:
    return bool(SLOT_ELIGIBILITY.get(slot, {slot}) & set(positions))


def eligibility_matrix(positions: Sequence[Sequence[str]], slots: Sequence[str]) -> np.ndarray:
    return np.array([[eligible(p, s) for s in slots] for p in positions], dtype=bool)


def optimise(values: np.ndarray, elig: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """Best total value and (player, slot) pairs. ``values`` is per player (n,) and
    ``elig`` is (n, slots). Negative values are never started."""
    if values.size == 0 or elig.size == 0:
        return 0.0, []
    gain = np.where(elig, np.maximum(values, 0.0)[:, None], _NEG)
    rows, cols = linear_sum_assignment(gain, maximize=True)
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if gain[r, c] > _NEG / 2]
    return float(sum(gain[r, c] for r, c in pairs)), pairs


def season_value(weekly: np.ndarray, elig: np.ndarray, weights: np.ndarray) -> float:
    """Sum over weeks of the best lineup's value. ``weekly`` is (players, weeks)."""
    if weekly.shape[0] == 0:
        return 0.0
    total = 0.0
    for k in range(weekly.shape[1]):
        if weights[k] == 0:
            continue
        v, _ = optimise(weekly[:, k], elig)
        total += weights[k] * v
    return total
