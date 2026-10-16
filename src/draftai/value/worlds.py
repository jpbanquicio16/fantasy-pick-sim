"""Simulated season outcomes ("worlds") for every board player.

A world fixes, for each player, how good his season is and which weeks he misses:

    value[i, p, k] = healthy[p, k] * mult[i, p] * 1[week k not in p's absence in world i]

``healthy`` is his expected locked score in week k when available (the board's weekly value
divided by his availability). ``mult`` is a season-level scoring multiplier drawn from the
calibrated FP/G quantiles (Lock-In values scale with game scores). Absences come from the
games-share draw as one contiguous block of weeks at a random point in the season, so
bench depth and waiver cover are valued in the weeks that actually need them.

Worlds are drawn once per board and shared by every candidate and every pick: common
random numbers, so comparisons between candidates reflect the candidates, not luck.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from draftai.value.board import Board, _share_draws, _split_normal_draws


@dataclass
class Worlds:
    healthy: np.ndarray  # (players, active weeks) float32
    weights: np.ndarray  # (active weeks,)
    mult: np.ndarray  # (worlds, players) float32
    out_start: np.ndarray  # (worlds, players) int16
    out_len: np.ndarray  # (worlds, players) int16

    @property
    def n(self) -> int:
        return int(self.mult.shape[0])

    @property
    def weeks(self) -> int:
        return int(self.healthy.shape[1])

    @classmethod
    def draw(cls, board: Board, n: int, seed: int = 0) -> "Worlds":
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
        return cls(healthy.astype(np.float32), board.weights[active], mult, start, length)

    def values(self, worlds: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """(m,) world indices and (m, r) board rows (-1 = empty) -> (m, r, weeks) values."""
        safe = np.maximum(rows, 0)
        w = worlds[:, None]
        base = self.healthy[safe] * self.mult[w, safe][..., None]
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
