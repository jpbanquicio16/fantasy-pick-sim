"""The as-of contract: a feature function sees only rows strictly before its cut-off.

Feature builders never receive the full game log. They receive an ``AsOf`` view, which
filters once at construction and refuses to hand out anything dated on or after the
cut-off. The perturbation test (tests/unit/test_features.py) checks the contract
mechanically: scrambling every post-cut-off row must leave the features unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class AsOf:
    as_of: date
    player_games: pl.DataFrame
    team_games: pl.DataFrame

    @classmethod
    def view(cls, player_games: pl.DataFrame, team_games: pl.DataFrame, as_of: date) -> "AsOf":
        pg = player_games.filter(pl.col("game_date") < as_of)
        tg = team_games.filter(pl.col("game_date") < as_of)
        return cls(as_of=as_of, player_games=pg, team_games=tg)

    def __post_init__(self) -> None:
        for name, df in (("player_games", self.player_games), ("team_games", self.team_games)):
            if df.height and df["game_date"].max() >= self.as_of:
                raise LeakageError(f"{name} contains rows dated on or after {self.as_of}")


def season_cutoff(team_games: pl.DataFrame, season: int) -> date:
    """Opening night of ``season``: features may use only games before this date."""
    opener = team_games.filter(pl.col("season") == season)["game_date"].min()
    if opener is None:
        raise ValueError(f"no games for season {season}")
    return opener


def live_cutoff(draft_date: date, schedule: pl.DataFrame | None = None) -> date:
    """For a live draft the cut-off is the draft date (or opening night, if earlier)."""
    if schedule is not None and schedule.height:
        return min(draft_date + timedelta(days=1), schedule["game_date"].min())
    return draft_date + timedelta(days=1)
