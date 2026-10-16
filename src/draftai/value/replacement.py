"""Value over replacement on Lock-In weekly value, with slot-level replacement levels.

Replacement levels come from filling every team's starting lineup at once: an
assignment of the pool to (teams x starting slots) maximising total value, so
multi-eligible players go where they add most. The last player used at each slot type
defines that slot's replacement level; a player is measured against his cheapest
eligible slot, which is why positional flexibility has value.

    VOR_p = sum_k w_k * (L_pk - R_{s*(p),k}),   s*(p) = argmin_{s in elig(p)} R_s

L_pk already includes the availability factor, so weeks he misses contribute
(0 - R), i.e. the cost of filling the slot from replacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from draftai.value.lineup import eligible, starting_slots


@dataclass
class ReplacementLevels:
    slot_types: list[str]
    player_index: dict[str, int]  # slot type -> row of the replacement player
    season_value: dict[str, float]  # slot type -> replacement season value
    weekly: dict[str, np.ndarray]  # slot type -> replacement weekly values
    starters: np.ndarray  # rows used as starters league-wide

    def cheapest_slot(self, positions: list[str]) -> str | None:
        options = [s for s in self.slot_types if eligible(positions, s)]
        return min(options, key=lambda s: self.season_value[s]) if options else None


def replacement_levels(
    weekly: np.ndarray,
    weights: np.ndarray,
    positions: list[list[str]],
    roster_positions: list[str],
    teams: int,
    pool_size: int | None = None,
) -> ReplacementLevels:
    season = weekly @ weights
    slots = starting_slots(roster_positions)
    pool_size = pool_size or min(len(season), teams * len(slots) * 3)
    order = np.argsort(-season)[:pool_size]
    instances = [s for s in slots for _ in range(teams)]
    gain = np.full((len(order), len(instances)), -1e9)
    for i, p in enumerate(order):
        for j, s in enumerate(instances):
            if eligible(positions[p], s):
                gain[i, j] = season[p]
    rows, cols = linear_sum_assignment(gain, maximize=True)
    chosen: dict[str, list[int]] = {}
    for r, c in zip(rows, cols):
        if gain[r, c] > -1e8:
            chosen.setdefault(instances[c], []).append(int(order[r]))
    types = sorted(set(slots), key=slots.index)
    idx, val, wk = {}, {}, {}
    for s in types:
        members = chosen.get(s, [])
        if not members:
            idx[s], val[s], wk[s] = -1, 0.0, np.zeros(weekly.shape[1])
            continue
        last = min(members, key=lambda p: season[p])
        idx[s], val[s], wk[s] = last, float(season[last]), weekly[last]
    starters = np.array(sorted(p for m in chosen.values() for p in m))
    return ReplacementLevels(types, idx, val, wk, starters)


def vor(weekly: np.ndarray, weights: np.ndarray, positions: list[list[str]], levels: ReplacementLevels) -> np.ndarray:
    out = np.zeros(weekly.shape[0])
    for p in range(weekly.shape[0]):
        s = levels.cheapest_slot(positions[p])
        if s is None:
            out[p] = -np.inf
            continue
        out[p] = float((weekly[p] - levels.weekly[s]) @ weights)
    return out
