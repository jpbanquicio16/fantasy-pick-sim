"""Play a drafted league through a real season.

Every team follows the same rules, so differences come from the draft:

* Lineups: each week, the best assignment of the healthy roster by in-season expected
  value (the projection updated with games already played), so a breakout gets started
  and a bust benched, without hindsight about the week itself.
* Scoring: a starter's week is his realised locked score under the pre-season lock rule.
* Waivers: a team that cannot fill a starting slot claims the best healthy free agent for
  it (up to ``claims`` per week; each free agent goes to one team a week, in an order
  that rotates weekly).
* Standings: random head-to-head schedules (averaged, since schedule luck is not a
  property of the draft), then the bracket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from draftai.backtest.history import LeagueRules, Realised
from draftai.value.lineup import optimise
from draftai.value.season import SeasonOutcome, season_outcomes


@dataclass
class LeagueResult:
    weekly: np.ndarray  # (teams, active weeks) realised scores
    outcome: SeasonOutcome

    def regular_per_week(self, playoff: np.ndarray) -> np.ndarray:
        return self.weekly[:, ~playoff].mean(axis=1)


def team_weeks(rows: list[int], R: Realised, elig: np.ndarray, free: np.ndarray | None = None,
               claims: int = 0) -> np.ndarray:
    """Weekly realised score of one roster (no competition for free agents)."""
    out = np.zeros(R.weights.size)
    for k in range(R.weights.size):
        out[k] = _week(rows, k, R, elig, free, claims, set())[0]
    return out


def _week(rows: list[int], k: int, R: Realised, elig: np.ndarray, free: np.ndarray | None, claims: int,
          taken: set[int]) -> tuple[float, list[int]]:
    ros = [r for r in rows if R.healthy[r, k]]
    added: list[int] = []
    total, pairs = optimise(R.expected[ros, k], elig[ros]) if ros else (0.0, [])
    n_slots = elig.shape[1]
    while free is not None and len(pairs) < n_slots and len(added) < claims:
        used = {s for _, s in pairs}
        open_slots = [s for s in range(n_slots) if s not in used]
        pick = next((int(f) for f in free if f not in taken and R.healthy[f, k] and elig[f, open_slots].any()), None)
        if pick is None:
            break
        added.append(pick)
        taken.add(pick)
        ros = ros + [pick]
        total, pairs = optimise(R.expected[ros, k], elig[ros])
    return float(sum(R.locked[ros[i], k] for i, _ in pairs)), added


def play(rosters: list[list[int]], R: Realised, elig: np.ndarray, rules: LeagueRules, rng: np.random.Generator,
         schedules: int = 40, claims: int = 2) -> LeagueResult:
    teams, K = len(rosters), R.weights.size
    drafted = {r for ros in rosters for r in ros}
    undrafted = np.array([p for p in range(R.locked.shape[0]) if p not in drafted], dtype=np.int64)
    weekly = np.zeros((teams, K))
    for k in range(K):
        free = undrafted[np.argsort(-R.expected[undrafted, k], kind="stable")][:60] if undrafted.size else None
        taken: set[int] = set()
        for i in range(teams):
            team = (i + k) % teams
            weekly[team, k], _ = _week(rosters[team], k, R, elig, free, claims, taken)
    outcome = season_outcomes(weekly[:, ~R.playoff], weekly[:, R.playoff], min(rules.playoff_teams, teams), schedules,
                              rng, rules.playoff_round_type)
    return LeagueResult(weekly, outcome)
