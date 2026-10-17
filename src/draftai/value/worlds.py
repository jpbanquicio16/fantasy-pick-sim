"""Simulated season outcomes ("worlds") for every board player.

A world fixes, for each player, how good his season is and which weeks he misses:

    value[i, p, k] = healthy[p, k] * mult[i, p] * 1[week k not in p's absence in world i]

``healthy`` is his expected locked score in week k when available (the board's weekly value
divided by his availability). ``mult`` is a season-level scoring multiplier drawn from the
calibrated FP/G quantiles (Lock-In values scale with game scores). Absences come from the
games-share draw as one contiguous block of weeks at a random point in the season, so
bench depth and waiver cover are valued in the weeks that actually need them.

Teammates share minutes. When a player sits, his minutes go to his teammates (the team
minutes budget applied week by week, by the same elasticity as models/budget.py), so their
values rise in exactly the weeks he is out:

    value[i, p, k] *= 1 + c_p * (missing minutes on p's team in week k of world i - expected)

with c_p = e(m_p) / (m_p * sum_j share_j e(m_j)) over his team, and values taken to move in
proportion to minutes. Measured against the expected absences, this leaves every player's
average unchanged; it makes teammates' seasons negatively correlated, as they are.

Worlds are drawn once per board and shared by every candidate and every pick: common
random numbers, so comparisons between candidates reflect the candidates, not luck.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from draftai.features.roles import elasticity
from draftai.value.board import Board, _share_draws, _split_normal_draws

BOOST_RANGE = (0.5, 2.0)  # bounds on a week's teammate-absence multiplier


@dataclass
class Worlds:
    healthy: np.ndarray  # (players, active weeks) float32
    weights: np.ndarray  # (active weeks,)
    mult: np.ndarray  # (worlds, players) float32
    out_start: np.ndarray  # (worlds, players) int16
    out_len: np.ndarray  # (worlds, players) int16
    # Teammates sharing minutes (None: independent players).
    team_idx: np.ndarray | None = None  # (players,) team index
    mins_dev: np.ndarray | None = None  # (worlds, teams, weeks) missing minutes minus their mean
    coef: np.ndarray | None = None  # (players,) value change per missing minute on his team
    own: np.ndarray | None = None  # (players, weeks) his own expected missing minutes

    @property
    def n(self) -> int:
        return int(self.mult.shape[0])

    @property
    def weeks(self) -> int:
        return int(self.healthy.shape[1])

    @classmethod
    def draw(cls, board: Board, n: int, seed: int = 0, teammates: bool = True) -> Worlds:
        p = board.players
        active = board.weights > 0
        a = p["avail_a"].fill_null(0.0).to_numpy()
        healthy = np.where(a[:, None] > 1e-6, board.weekly[:, active] / np.maximum(a, 1e-6)[:, None], 0.0)
        k = int(active.sum())
        rng = np.random.default_rng([seed, 1])
        players = p.height
        mult = np.empty((n, players), dtype=np.float32)
        start = np.empty((n, players), dtype=np.int16)
        length = np.empty((n, players), dtype=np.int16)
        fp50 = p["fp_p50"].fill_null(0.0).to_numpy()
        fp_mean = p["fp_pg"].to_numpy()
        p10, p90 = p["fp_p10"].fill_null(0.0).to_numpy(), p["fp_p90"].fill_null(0.0).to_numpy()
        s10, s50, s90 = (p[c].fill_null(0.0).to_numpy() for c in ("share_p10", "share_p50", "share_p90"))
        q = np.clip(p["q_short_hat"].fill_null(0.0).to_numpy(), 0.0, 0.5)
        z, u, us = rng.standard_normal((n, players)), rng.random((n, players)), rng.random((n, players))
        for j in range(players):
            mid = fp50[j] if fp50[j] > 1e-6 else fp_mean[j]
            if mid <= 1e-6:
                mult[:, j] = 1.0
            else:
                mult[:, j] = _split_normal_draws((p10[j] or mid) / mid, (p90[j] or mid) / mid, z[:, j])
            share = _share_draws(s10[j], s50[j], s90[j], u[:, j])
            avail_frac = np.clip(share / (1.0 - q[j]), 0.0, 1.0)
            out = np.floor((1.0 - avail_frac) * k + us[:, j]).astype(np.int64)
            out = np.clip(out, 0, k)
            length[:, j] = out
            start[:, j] = np.floor(rng.random(n) * (k - out + 1)).astype(np.int64)
        w = cls(healthy.astype(np.float32), board.weights[active], mult, start, length)
        if teammates and {"team_now", "mpg_hat", "share_hat"} <= set(p.columns):
            w._share_minutes(p["team_now"].to_list(), p["mpg_hat"].fill_null(0.0).to_numpy(),
                             p["share_hat"].fill_null(0.0).to_numpy())
        return w

    def _share_minutes(self, teams: list, mpg: np.ndarray, share: np.ndarray) -> None:
        names = sorted({t for t in teams if t})
        t_of = {t: i for i, t in enumerate(names)}
        idx = np.array([t_of.get(t, -1) for t in teams])
        e = elasticity(mpg) * np.clip(share, 0.0, 1.0)
        W = np.bincount(np.maximum(idx, 0), weights=np.where(idx >= 0, e, 0.0), minlength=len(names))
        ok = (idx >= 0) & (mpg > 1e-6)
        denom = np.where(ok, mpg * W[np.maximum(idx, 0)], 1.0)
        coef = np.where(ok & (denom > 1e-9), elasticity(mpg) / np.maximum(denom, 1e-9), 0.0)
        wk = np.arange(self.weeks)
        out = (wk >= self.out_start[..., None]) & (wk < (self.out_start + self.out_len)[..., None])  # (n, P, K)
        missing = np.zeros((self.n, len(names), self.weeks), dtype=np.float32)
        for t in range(len(names)):
            members = np.flatnonzero(idx == t)
            if members.size:
                missing[:, t] = np.einsum("ipk,p->ik", out[:, members], mpg[members])
        self.team_idx = np.maximum(idx, 0)
        self.mins_dev = (missing - missing.mean(axis=0, keepdims=True)).astype(np.float32)
        self.coef = coef.astype(np.float32)
        self.own = (mpg[:, None] * out.mean(axis=0)).astype(np.float32)

    def values(self, worlds: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """(m,) world indices and (m, r) board rows (-1 = empty) -> (m, r, weeks) values."""
        safe = np.maximum(rows, 0)
        w = worlds[:, None]
        base = self.healthy[safe] * self.mult[w, safe][..., None]
        if self.coef is not None:
            dev = self.mins_dev[w, self.team_idx[safe]] + self.own[safe]  # others' missing minutes vs expected
            base = base * np.clip(1.0 + self.coef[safe][..., None] * dev, *BOOST_RANGE)
        wk = np.arange(self.weeks)
        s, ln = self.out_start[w, safe][..., None], self.out_len[w, safe][..., None]
        out = (wk >= s) & (wk < s + ln)
        return np.where(out | (rows[..., None] < 0), 0.0, base)

    def expected(self) -> np.ndarray:
        """Mean value over worlds per player and week (a check against the board)."""
        n = self.n
        rows = np.arange(self.healthy.shape[0])
        acc = np.zeros(self.healthy.shape)
        for i in range(n):
            acc += self.values(np.array([i]), rows[None, :])[0]
        return acc / n
