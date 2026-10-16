"""Head-to-head seasons: schedules, standings and the playoff bracket.

Pure functions on (teams, weeks) score matrices, shared by the draft backtest (real box
scores) and the end-of-mock projection (simulated worlds). The bracket is a fixed,
standard-seeded bracket with byes for the top seeds, which is Sleeper's: with six playoff
teams, 3 v 6 and 4 v 5 in the first round, then 1 and 2 meet the winners.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def round_robin(teams: int, weeks: int, rng: np.random.Generator) -> np.ndarray:
    """(weeks, teams) opponent of each team each week; -1 is a bye (odd team counts).

    Circle method on a random ordering of the teams, cycled for as many weeks as needed.
    """
    n = teams + (teams % 2)
    team_at = rng.permutation(n)  # circle position -> team (team id ``teams`` is the odd-count dummy)
    ring = list(range(n))
    rounds = []
    for _ in range(n - 1):
        opp = np.full(n, -1)
        for i in range(n // 2):
            a, b = team_at[ring[i]], team_at[ring[n - 1 - i]]
            opp[a], opp[b] = b, a
        rounds.append(opp[:teams])
        ring = [ring[0], ring[-1], *ring[1:-1]]
    out = np.empty((weeks, teams), dtype=np.int64)
    for w in range(weeks):
        opp = rounds[w % len(rounds)]
        out[w] = np.where(opp >= teams, -1, opp)
    return out


def standings(scores: np.ndarray, schedule: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wins (ties count half) and points for. ``scores`` is (teams, weeks)."""
    t, w = scores.shape
    wins = np.zeros(t)
    for k in range(w):
        opp = schedule[k]
        has = opp >= 0
        mine, theirs = scores[:, k], scores[np.maximum(opp, 0), k]
        wins += np.where(has, (mine > theirs) + 0.5 * (mine == theirs), 0.0)
    return wins, scores.sum(axis=1)


def seed_order(wins: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Teams from first seed to last: wins, then points for."""
    return np.lexsort((-points, -wins))


def bracket_positions(size: int) -> list[int]:
    """Standard seeding: [1, 8, 4, 5, 2, 7, 3, 6] for eight; adjacent pairs meet."""
    pos = [1, 2]
    while len(pos) < size:
        n = 2 * len(pos)
        pos = [x for s in pos for x in (s, n + 1 - s)]
    return pos


def playoff_champion(seeds: np.ndarray, playoff_teams: int, round_scores: np.ndarray) -> int:
    """Winner of the bracket. ``seeds`` lists teams by seed; ``round_scores`` is
    (teams, rounds), each round's score summed over its weeks. Ties go to the better seed."""
    size = 1
    while size < playoff_teams:
        size *= 2
    field = [int(seeds[s - 1]) if s <= playoff_teams else -1 for s in bracket_positions(size)]
    seed_of = {int(t): i for i, t in enumerate(seeds)}
    r = 0
    while len(field) > 1:
        nxt = []
        for a, b in zip(field[::2], field[1::2]):
            if a < 0 or b < 0:
                nxt.append(max(a, b))
                continue
            sa, sb = round_scores[a, r], round_scores[b, r]
            nxt.append(a if sa > sb or (sa == sb and seed_of[a] < seed_of[b]) else b)
        field = nxt
        r += 1
    return field[0]


def round_spans(playoff_teams: int, round_type: int = 0) -> list[int]:
    """Weeks per playoff round (Sleeper's playoff_round_type: 1 = two-week final,
    2 = two weeks per round)."""
    rounds = 1
    while 2**rounds < playoff_teams:
        rounds += 1
    return [2 if round_type == 2 or (round_type == 1 and r == rounds - 1) else 1 for r in range(rounds)]


def group_rounds(po_scores: np.ndarray, playoff_teams: int, round_type: int = 0) -> np.ndarray:
    """(teams, playoff weeks) -> (teams, rounds) by summing each round's weeks."""
    spans = round_spans(playoff_teams, round_type)
    cols, w = [], 0
    for s in spans:
        cols.append(po_scores[:, w: w + s].sum(axis=1) if w < po_scores.shape[1] else np.zeros(po_scores.shape[0]))
        w += s
    return np.stack(cols, axis=1)


@dataclass
class SeasonOutcome:
    p_playoffs: np.ndarray  # (teams,)
    p_title: np.ndarray
    win_rate: np.ndarray  # head-to-head weekly win rate
    mean_seed: np.ndarray


def season_outcomes(regular: np.ndarray, playoff: np.ndarray, playoff_teams: int, schedules: int,
                    rng: np.random.Generator, round_type: int = 0) -> SeasonOutcome:
    """Average over random schedules (schedule luck is not a property of the roster).

    ``regular`` (teams, weeks) are regular-season scores; ``playoff`` (teams, weeks) the
    playoff weeks' scores; either can be (schedules, teams, weeks) to give every schedule
    its own draw (simulated seasons).
    """
    teams = regular.shape[-2]
    made, title, wr, seed_sum = np.zeros(teams), np.zeros(teams), np.zeros(teams), np.zeros(teams)
    for i in range(schedules):
        reg = regular[i] if regular.ndim == 3 else regular
        po = playoff[i] if playoff.ndim == 3 else playoff
        sched = round_robin(teams, reg.shape[1], rng)
        wins, pf = standings(reg, sched)
        order = seed_order(wins, pf)
        k = min(playoff_teams, teams)
        made[order[:k]] += 1
        seed_sum[order] += np.arange(1, teams + 1)
        games = (sched >= 0).sum(axis=0)
        wr += wins / np.maximum(games, 1)
        if po.shape[1]:
            title[playoff_champion(order, k, group_rounds(po, k, round_type))] += 1
    return SeasonOutcome(made / schedules, title / schedules, wr / schedules, seed_sum / schedules)
