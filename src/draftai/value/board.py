"""The draft board: every draftable player's game distribution, weekly Lock-In values,
season value, its uncertainty, and value over replacement.

The board is the only thing the draft engine reads. It is built offline (seconds) and
saved as a Parquet table plus a NumPy weekly-value matrix aligned by row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from draftai.features.build import COMPONENTS
from draftai.scoring import ScoringSettings
from draftai.value.games import (
    DonorBank,
    GameProjection,
    GameSimulator,
    extra_rates,
    simulated_stats,
)
from draftai.value.lineup import starting_slots
from draftai.value.lockin import EmpiricalScores, kappa, thresholds
from draftai.value.replacement import ReplacementLevels, replacement_levels, vor
from draftai.value.schedule import WeekPlan, team_week_patterns
from draftai.value.weekly import Availability, lockin_profile, weekly_values

# Pre-draft Sleeper injury status -> games-share multiplier. Deliberately mild: the
# status says nothing about duration. Use configs/availability_overrides.csv for specifics.
INJURY_SHARE_MULT = {"Out": 0.85, "Doubtful": 0.95, "Questionable": 0.98, "DTD": 0.98, "Day-To-Day": 0.98}
SEASON_DRAWS = 4000
QUANTILE_GRID = np.linspace(0.0, 1.0, 201)  # stored game-score distribution per player


@dataclass
class Board:
    players: pl.DataFrame
    weekly: np.ndarray  # (players, weeks) expected locked score
    weeks: list[int]
    weights: np.ndarray  # per week
    replacement: ReplacementLevels
    roster_positions: list[str]
    teams: int
    quantiles: np.ndarray | None = None  # (players, 201) game-score quantiles; None on old boards

    @property
    def slots(self) -> list[str]:
        return starting_slots(self.roster_positions)

    def positions(self) -> list[list[str]]:
        return self.players["positions"].to_list()

    def game_scores(self, row: int) -> EmpiricalScores:
        """A player's simulated game-score distribution (from the stored quantile grid)."""
        if self.quantiles is None:
            raise ValueError("this board predates stored game distributions; rerun `draftai project`")
        return EmpiricalScores(self.quantiles[row])

    def with_teams(self, teams: int) -> "Board":
        """The same players for a league of a different size: replacement levels and VOR
        depend on how many lineups the pool must fill."""
        positions = self.positions()
        levels = replacement_levels(self.weekly, self.weights, positions, self.roster_positions, teams)
        v = vor(self.weekly, self.weights, positions, levels)
        players = self.players.with_columns(
            pl.Series("vor", v),
            pl.Series("vor_per_week", v / max(self.weights.sum(), 1e-9)),
            pl.Series("repl_slot", [levels.cheapest_slot(p) for p in positions]),
        )
        return Board(players, self.weekly, self.weeks, self.weights, levels, self.roster_positions, teams,
                     self.quantiles)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.players.write_parquet(directory / "board.parquet")
        np.save(directory / "weekly.npy", self.weekly)
        if self.quantiles is not None:
            np.save(directory / "game_quantiles.npy", self.quantiles)
        meta = {
            "weeks": self.weeks,
            "weights": self.weights.tolist(),
            "roster_positions": self.roster_positions,
            "teams": self.teams,
            "replacement": {
                "slot_types": self.replacement.slot_types,
                "player_index": self.replacement.player_index,
                "season_value": self.replacement.season_value,
                "weekly": {k: v.tolist() for k, v in self.replacement.weekly.items()},
                "starters": self.replacement.starters.tolist(),
            },
        }
        (directory / "board_meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, directory: Path) -> "Board":
        meta = json.loads((directory / "board_meta.json").read_text())
        r = meta["replacement"]
        rep = ReplacementLevels(
            r["slot_types"], {k: int(v) for k, v in r["player_index"].items()}, r["season_value"],
            {k: np.array(v) for k, v in r["weekly"].items()}, np.array(r["starters"], dtype=int),
        )
        return cls(
            players=pl.read_parquet(directory / "board.parquet"),
            weekly=np.load(directory / "weekly.npy"),
            weeks=meta["weeks"],
            weights=np.array(meta["weights"]),
            replacement=rep,
            roster_positions=meta["roster_positions"],
            teams=meta["teams"],
            quantiles=np.load(directory / "game_quantiles.npy") if (directory / "game_quantiles.npy").exists() else None,
        )


def _split_normal_draws(lo_rel: float, hi_rel: float, z: np.ndarray) -> np.ndarray:
    """Season-level multiplier with median 1 and the calibrated P10/P90 as quantiles."""
    s_lo = max(1e-6, (1 - lo_rel) / 1.2816)
    s_hi = max(1e-6, (hi_rel - 1) / 1.2816)
    return np.maximum(0.0, 1 + np.where(z < 0, s_lo, s_hi) * z)


def _share_draws(p10: float, p50: float, p90: float, u: np.ndarray) -> np.ndarray:
    lo0 = max(0.0, p10 - 0.5 * (p50 - p10))
    hi1 = min(1.0, p90 + 0.5 * (p90 - p50))
    return np.interp(u, [0.0, 0.1, 0.5, 0.9, 1.0], [lo0, p10, p50, p90, hi1])


def _locked_sd(fp: np.ndarray, dist: EmpiricalScores, rng: np.random.Generator, n: int = 2000) -> float:
    """Spread of the locked score in a three-game week under the optimal lock rule: the
    week-to-week noise a head-to-head matchup sees from one lineup slot."""
    t = thresholds(dist, [0.0, 0.0, 0.0])
    x = rng.choice(fp, size=(n, 3))
    locked = np.where(x[:, 0] >= t[0], x[:, 0], np.where(x[:, 1] >= t[1], x[:, 1], x[:, 2]))
    return float(locked.std())


def build_board(
    proj: pl.DataFrame,
    history: pl.DataFrame,
    schedule: pl.DataFrame,
    plan: WeekPlan,
    settings: ScoringSettings,
    roster_positions: list[str],
    teams: int,
    b2b_mult: float,
    target_season: int,
    game_samples: int = 4000,
    min_donor_games: int = 40,
    neighbours: int = 8,
    seed: int = 0,
    log=lambda *_: None,
) -> Board:
    """``proj`` needs: player_id, team_now, positions, mpg_hat, rate_*_hat, share_hat,
    share_p10/p50/p90, fp_p10/p50/p90, q_short_hat, rookie, donor_ids (rookies)."""
    stats = simulated_stats(settings)
    bank = DonorBank.build(history, target_season, stats)
    sim = GameSimulator(bank, settings, min_donor_games, neighbours, seed)
    patterns = team_week_patterns(schedule)
    weeks = plan.weeks
    weights = np.array([plan.weights[w] for w in weeks])
    rng = np.random.default_rng(seed)
    z_season, u_share = rng.standard_normal(SEASON_DRAWS), rng.random(SEASON_DRAWS)

    proj = proj.with_columns(
        pl.col("injury_status").replace_strict(INJURY_SHARE_MULT, default=1.0, return_dtype=pl.Float64).alias("inj_mult")
        if "injury_status" in proj.columns else pl.lit(1.0).alias("inj_mult")
    ).with_columns(
        (pl.col("share_hat") * pl.col("inj_mult")).alias("share_hat"),
        (pl.col("share_p10") * pl.col("inj_mult")).alias("share_p10"),
        (pl.col("share_p50") * pl.col("inj_mult")).alias("share_p50"),
        (pl.col("share_p90") * pl.col("inj_mult")).alias("share_p90"),
    )
    rows, weekly, quantiles = [], [], []
    for r in proj.iter_rows(named=True):
        rates = {c: float(r[f"rate_{c}_hat"] or 0.0) for c in COMPONENTS}
        if len(stats) > len(COMPONENTS):
            rates.update(extra_rates(history, r["player_id"], rates, target_season))
        donors = list(r["donor_ids"]) if r.get("rookie") and r.get("donor_ids") is not None else None
        gp = GameProjection(int(r["player_id"]), float(r["mpg_hat"] or 0.0), rates, donors=donors)
        try:
            fp = sim.fp_samples(gp, game_samples)
        except ValueError:
            continue
        # Technical and flagrant fouls: a small per-game expectation (not in box scores).
        fp = fp + float(r.get("foul_fp_pg") or 0.0)
        dist = EmpiricalScores(fp)
        avail = Availability.from_share(r["share_hat"] or 0.0, r["q_short_hat"] or 0.0, b2b_mult)
        team_pats = [patterns.get((r["team_now"], w), ()) for w in weeks]
        wk = weekly_values(dist, avail, team_pats)
        prof = lockin_profile(dist, 4)
        lock_sd = _locked_sd(fp, dist, rng)
        season = float(wk @ weights)
        # Season-value uncertainty: calibrated FP/G and games-share quantiles.
        p50 = max(r["fp_p50"] or dist.mean, 1e-6)
        mult = _split_normal_draws((r["fp_p10"] or p50) / p50, (r["fp_p90"] or p50) / p50, z_season)
        sh = _share_draws(r["share_p10"], r["share_p50"], r["share_p90"], u_share)
        q_raw = min(float(r["q_short_hat"] or 0.0), 0.5)
        a_ratio = np.clip(sh / (1.0 - q_raw), 0.0, 1.0) / max(avail.a, 1e-6)
        season_draws = season * mult * a_ratio
        rows.append(
            {
                **{k: r[k] for k in r if k not in ("donor_ids",)},
                "fp_pg": dist.mean,
                "fp_sd_game": dist.sd,
                "p_game_50": dist.tail_prob(50.0),
                "p_game_40": dist.tail_prob(40.0),
                "lockin_2": prof[2],
                "lockin_3": prof[3],
                "lockin_4": prof[4],
                "lockin_premium_3": prof[3] - dist.mean,
                "lockin_normal_3": dist.mean + kappa(3) * dist.sd,
                "lockin_sd_3": lock_sd,
                "avail_a": avail.a,
                "avail_q": avail.q,
                "games_proj": float(sum(len(p) for p in team_pats) * (r["share_hat"] or 0.0)),
                "season_value": season,
                "season_p10": float(np.quantile(season_draws, 0.1)),
                "season_p90": float(np.quantile(season_draws, 0.9)),
                "season_sd": float(season_draws.std()),
                "weekly_value": season / max(weights.sum(), 1e-9),
            }
        )
        weekly.append(wk)
        quantiles.append(np.quantile(fp, QUANTILE_GRID))
    players = pl.DataFrame(rows, infer_schema_length=None)
    W = np.vstack(weekly) if weekly else np.zeros((0, len(weeks)))
    positions = players["positions"].to_list()
    levels = replacement_levels(W, weights, positions, roster_positions, teams)
    v = vor(W, weights, positions, levels)
    players = players.with_columns(
        pl.Series("vor", v),
        pl.Series("vor_per_week", v / max(weights.sum(), 1e-9)),
        pl.Series("repl_slot", [levels.cheapest_slot(p) for p in positions]),
    )
    log(f"board: {players.height} players, {len(weeks)} weeks; replacement levels "
        + ", ".join(f"{s}={levels.season_value[s]:.0f}" for s in levels.slot_types))
    Q = np.vstack(quantiles) if quantiles else np.zeros((0, QUANTILE_GRID.size))
    return Board(players, W, weeks, weights, levels, roster_positions, teams, Q)
