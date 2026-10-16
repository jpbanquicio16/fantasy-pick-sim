"""Game-level simulation: whole box scores drawn around a player's projection.

Each simulated game resamples one real game (the player's own, or a similar player's
when his sample is thin), keeps its minutes ratio and its standardised residual vector,
and re-centres them on the projected minutes and per-minute rates:

    z_c   = (x_c - r_c * m) / sqrt(r_c * m + 1)      from the donor game
    m'    = m * mpg_hat / mpg_donor
    x'_c  = r_hat_c * m' + z_c * sqrt(r_hat_c * m' + 1)   (stochastically rounded, >= 0)

When the projection equals the donor's own season, the donor game is reproduced
exactly. Resampling whole residual vectors keeps the within-game correlation between
minutes, points, threes and assists without fitting a copula; every simulated box
score is then scored by the deterministic engine, so bonus probabilities are exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from draftai.features.build import COMPONENTS
from draftai.scoring import ScoringSettings, score_arrays

# Box columns not projected directly scale with a projected component.
EXTRA_SCALE = {"fgm": "pts", "fga": "pts", "ftm": "pts", "fta": "pts", "fg3a": "fg3m", "oreb": "reb",
               "dreb": "reb", "pf": None}
RECENCY = {1: 1.0, 2: 0.6}


def simulated_stats(settings: ScoringSettings) -> list[str]:
    """Box columns the simulator must draw: the projected components plus any extra
    column the league scores (e.g. FGA for missed-shot penalties)."""
    from draftai.scoring.settings import MISS_KEYS, STAT_KEYS

    need = set(COMPONENTS)
    for key in settings.weights:
        if key in STAT_KEYS and STAT_KEYS[key] in EXTRA_SCALE:
            need.add(STAT_KEYS[key])
        if key in MISS_KEYS:
            need.update(MISS_KEYS[key])
    return [c for c in [*COMPONENTS, *EXTRA_SCALE] if c in need]


@dataclass
class DonorBank:
    """Standardised game residuals for every player-season in the recent history."""

    player_id: np.ndarray  # per game
    season_lag: np.ndarray
    minutes: np.ndarray
    mpg: np.ndarray  # donor's season minutes per game
    z: dict[str, np.ndarray]  # per stat, per game
    index: dict[int, np.ndarray]  # player_id -> game indices
    profile: pl.DataFrame  # player_id, games, mpg, per-minute rates (for neighbours)

    @classmethod
    def build(cls, history: pl.DataFrame, target_season: int, stats: list[str]) -> "DonorBank":
        h = history.filter(
            (pl.col("season") >= target_season - 2) & (pl.col("season") < target_season) & (pl.col("min") >= 1)
        )
        tot = h.group_by("player_id", "season").agg(
            pl.col("min").sum().alias("smin"), pl.len().alias("sgp"), *[pl.col(c).sum().alias(f"s_{c}") for c in stats]
        )
        h = h.join(tot, on=["player_id", "season"]).sort("player_id", "game_date")
        m = h["min"].to_numpy().astype(float)
        smin = h["smin"].to_numpy().astype(float)
        z = {}
        for c in stats:
            rate = h[f"s_{c}"].to_numpy() / smin
            e = rate * m
            z[c] = (h[c].to_numpy() - e) / np.sqrt(e + 1.0)
        pid = h["player_id"].to_numpy()
        index: dict[int, np.ndarray] = {}
        order = np.argsort(pid, kind="stable")
        uniq, starts = np.unique(pid[order], return_index=True)
        for u, s, e in zip(uniq, starts, [*starts[1:], len(order)]):
            index[int(u)] = order[s:e]
        prof = (
            tot.group_by("player_id")
            .agg(pl.col("sgp").sum().alias("games"), pl.col("smin").sum().alias("min"),
                 *[pl.col(f"s_{c}").sum().alias(c) for c in COMPONENTS])
            .with_columns(
                (pl.col("min") / pl.col("games")).alias("mpg"),
                *[(pl.col(c) / pl.col("min")).alias(f"r_{c}") for c in COMPONENTS],
            )
        )
        return cls(
            player_id=pid,
            season_lag=target_season - h["season"].to_numpy(),
            minutes=m,
            mpg=smin / h["sgp"].to_numpy(),
            z=z,
            index=index,
            profile=prof,
        )

    def neighbours(self, mpg: float, rates: dict[str, float], k: int, exclude: int | None = None,
                   min_games: int = 40) -> list[int]:
        prof = self.profile.filter(pl.col("games") >= min_games)
        if exclude is not None:
            prof = prof.filter(pl.col("player_id") != exclude)
        cols = ["mpg", *[f"r_{c}" for c in COMPONENTS]]
        x = prof.select(cols).to_numpy()
        target = np.array([mpg, *[rates[c] for c in COMPONENTS]])
        scale = x.std(axis=0) + 1e-9
        d = np.sqrt((((x - target) / scale) ** 2).sum(axis=1))
        return prof["player_id"].to_numpy()[np.argsort(d)[:k]].tolist()


@dataclass
class GameProjection:
    """What the simulator needs for one player (per-minute rates, minutes per game)."""

    player_id: int
    mpg: float
    rates: dict[str, float]
    donors: list[int] | None = None  # explicit donor players (rookies: their pick bucket)


class GameSimulator:
    def __init__(self, bank: DonorBank, settings: ScoringSettings, min_donor_games: int = 40,
                 neighbours: int = 8, seed: int = 0):
        self.bank = bank
        self.settings = settings
        self.min_donor_games = min_donor_games
        self.k = neighbours
        self.seed = seed

    def _pool(self, proj: GameProjection) -> tuple[np.ndarray, np.ndarray]:
        """Donor game indices and sampling weights."""
        b = self.bank
        own = b.index.get(proj.player_id, np.array([], dtype=int)) if proj.donors is None else np.array([], dtype=int)
        w_own = np.array([RECENCY.get(int(l), 0.3) for l in b.season_lag[own]]) if own.size else np.array([])
        n_eff = w_own.sum()
        if n_eff >= self.min_donor_games:
            return own, w_own
        donors = proj.donors or b.neighbours(proj.mpg, proj.rates, self.k, exclude=proj.player_id)
        idx = [b.index[d] for d in donors if d in b.index]
        if not idx:
            if own.size:
                return own, w_own
            raise ValueError(f"no donor games for player {proj.player_id}")
        other = np.concatenate(idx)
        w_other = np.array([RECENCY.get(int(l), 0.3) for l in b.season_lag[other]])
        w_other *= (self.min_donor_games - n_eff) / w_other.sum()
        return np.concatenate([own, other]), np.concatenate([w_own, w_other])

    def sample_box(self, proj: GameProjection, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        b = self.bank
        pool, w = self._pool(proj)
        g = rng.choice(pool, size=n, p=w / w.sum())
        m_new = b.minutes[g] * (proj.mpg / b.mpg[g])
        box: dict[str, np.ndarray] = {}
        for c in b.z:
            r = proj.rates[c] if c in proj.rates else proj.rates.get(f"_{c}", 0.0)
            e = r * m_new
            x = np.maximum(e + b.z[c][g] * np.sqrt(e + 1.0), 0.0)
            box[c] = np.floor(x + rng.random(n)).astype(np.int64)
        box["min"] = m_new
        return box

    def fp_samples(self, proj: GameProjection, n: int, seed_offset: int = 0) -> np.ndarray:
        rng = np.random.default_rng([self.seed, proj.player_id % (2**31), seed_offset])
        return score_arrays(self.sample_box(proj, n, rng), self.settings)


def extra_rates(history: pl.DataFrame, player_id: int, rates: dict[str, float], target_season: int) -> dict[str, float]:
    """Per-minute rates for box columns the models do not project, scaled like their
    anchor component (e.g. FGA moves with points). Stored under a leading underscore."""
    h = history.filter((pl.col("player_id") == player_id) & (pl.col("season") >= target_season - 2))
    out = {}
    if h.height == 0:
        return out
    tot = h.select([pl.col(c).sum() for c in [*COMPONENTS, *EXTRA_SCALE, "min"]]).row(0, named=True)
    for c, anchor in EXTRA_SCALE.items():
        base = tot[c] / max(tot["min"], 1.0)
        if anchor is not None and tot[anchor] > 0:
            base *= rates[anchor] / (tot[anchor] / max(tot["min"], 1.0))
        out[f"_{c}"] = base
    return out
