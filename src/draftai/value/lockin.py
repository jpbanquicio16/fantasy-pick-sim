"""Lock-In as optimal stopping: the expected score of the game you would actually lock.

Backward induction over a player's games in a week (q_k = probability he misses game k):

    V_n = (1 - q_n) * mu                      the last game is forced (zero if he sits)
    V_k = (1 - q_k) * E[max(X_k, V_{k+1})] + q_k * V_{k+1}
    lock game k  <=>  X_k >= V_{k+1}

E[max(X1..Xn)] would assume you can pick the best game in hindsight; Sleeper requires
the decision before the next game, so that "prophet" value is only an upper bound.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy import integrate
from scipy.stats import norm


# --- normal approximation --------------------------------------------------------
def emax_normal(mu: float, sigma: float, c: float) -> float:
    """E[max(X, c)] for X ~ N(mu, sigma^2)."""
    if sigma <= 0:
        return max(mu, c)
    z = (c - mu) / sigma
    return c * norm.cdf(z) + mu * norm.sf(z) + sigma * norm.pdf(z)


def stopping_values_normal(mu: float, sigma: float, q: Sequence[float]) -> list[float]:
    """[V_1, ..., V_n] for normal game scores and per-game miss probabilities ``q``."""
    n = len(q)
    if n == 0:
        return []
    v = [0.0] * n
    v[-1] = (1 - q[-1]) * mu
    for k in range(n - 2, -1, -1):
        v[k] = (1 - q[k]) * emax_normal(mu, sigma, v[k + 1]) + q[k] * v[k + 1]
    return v


def lockin_normal(mu: float, sigma: float, n_games: int, q: float = 0.0) -> float:
    """Expected locked score per slot-week under the normal approximation."""
    if n_games <= 0:
        return 0.0
    return stopping_values_normal(mu, sigma, [q] * n_games)[0]


@lru_cache(maxsize=None)
def kappa(n_games: int) -> float:
    """Lockable premium in SDs: V_1 = mu + kappa(n) * sigma for standard normal games."""
    return lockin_normal(0.0, 1.0, n_games)


@lru_cache(maxsize=None)
def prophet_kappa(n_games: int) -> float:
    """E[max of n standard normals]: the hindsight upper bound."""
    if n_games <= 1:
        return 0.0
    f = lambda x: x * n_games * norm.pdf(x) * norm.cdf(x) ** (n_games - 1)
    return float(integrate.quad(f, -12, 12)[0])


# --- empirical distributions (the simulator's game scores) ----------------------
class EmpiricalScores:
    """Sorted sample of simulated game scores with O(log n) E[max(X, c)]."""

    def __init__(self, samples: np.ndarray):
        x = np.sort(np.asarray(samples, dtype=float))
        self.x = x
        self.n = x.size
        # tail_sum[i] = sum of x[i:]
        self.tail_sum = np.concatenate([np.cumsum(x[::-1])[::-1], [0.0]])
        self.mean = float(x.mean()) if self.n else 0.0
        self.sd = float(x.std()) if self.n else 0.0

    def emax(self, c: float) -> float:
        i = int(np.searchsorted(self.x, c, side="right"))
        return (c * i + self.tail_sum[i]) / self.n

    def tail_prob(self, c: float) -> float:
        return float(1.0 - np.searchsorted(self.x, c, side="left") / self.n)


def stopping_values(dist: EmpiricalScores, q: Sequence[float]) -> list[float]:
    n = len(q)
    if n == 0:
        return []
    v = [0.0] * n
    v[-1] = (1 - q[-1]) * dist.mean
    for k in range(n - 2, -1, -1):
        v[k] = (1 - q[k]) * dist.emax(v[k + 1]) + q[k] * v[k + 1]
    return v


def lockin_value(dist: EmpiricalScores, q: Sequence[float]) -> float:
    v = stopping_values(dist, q)
    return v[0] if v else 0.0


def thresholds(dist: EmpiricalScores, q: Sequence[float]) -> list[float]:
    """Lock threshold for each game: V_{k+1}, and -inf for the forced last game."""
    v = stopping_values(dist, q)
    return [*v[1:], -np.inf] if v else []


def realised_lock(scores: Sequence[float | None], thr: Sequence[float]) -> float:
    """Apply a threshold rule to an actual week; ``None`` marks a missed game.

    An unlocked starter takes his team's final game of the week, zero if he sits.
    """
    n = len(scores)
    for k, (x, t) in enumerate(zip(scores, thr)):
        if x is None:
            if k == n - 1:
                return 0.0
            continue
        if k == n - 1 or x >= t:
            return float(x)
    return 0.0
