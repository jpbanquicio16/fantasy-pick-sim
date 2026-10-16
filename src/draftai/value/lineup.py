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


def slot_specificity(slots: Sequence[str]) -> np.ndarray:
    """How many positions each slot accepts (PG: 1, G: 3, UTIL: 7); fewer is more specific."""
    return np.array([len(SLOT_ELIGIBILITY.get(s, {s})) for s in slots], dtype=float)


def greedy_lineups(values: np.ndarray, elig: np.ndarray, specificity: np.ndarray) -> np.ndarray:
    """Lineup value for many rosters and weeks at once: (m, players, weeks) -> (m, weeks).

    Players are placed in descending value, each into the most specific open slot he is
    eligible for; between equally specific slots, the one fewer of the remaining players
    can fill. With flex slots (G, F, UTIL) this is within about half a percent of the exact
    assignment (tests/unit/test_season.py), and it vectorises across simulated worlds where
    ``optimise`` needs one solver call per roster-week. Values <= 0 are never started.
    ``elig`` is (m, players, slots).
    """
    m, n, k = values.shape
    s = elig.shape[2]
    order = np.argsort(-values, axis=1)
    open_ = np.ones((m, k, s), dtype=bool)
    total = np.zeros((m, k))
    mi = np.arange(m)[:, None]
    positive = (values > 0)[..., None] & elig[:, :, None, :]  # (m, n, k, s)
    demand = positive.sum(axis=1).astype(float)  # (m, k, s) players who could still use each slot
    tiebreak = np.arange(s) * 1e-6  # then the earlier slot
    for r in range(n):
        idx = order[:, r, :]  # (m, k)
        v = np.take_along_axis(values, idx[:, None, :], axis=1)[:, 0, :]
        e = elig[mi, idx]  # (m, k, s)
        demand -= e & (v > 0)[..., None]
        cand = e & open_ & (v > 0)[..., None]
        score = np.where(cand, specificity * 1e3 + demand + tiebreak, np.inf)
        best = score.argmin(axis=2)
        ok = np.isfinite(score.min(axis=2))
        total += np.where(ok, v, 0.0)
        a, b = np.nonzero(ok)
        open_[a, b, best[a, b]] = False
    return total
