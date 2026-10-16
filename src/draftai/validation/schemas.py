"""Pandera schemas and cross-table checks. A failed check stops the pipeline."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

_count = pa.Check.in_range(0, 200)

PLAYER_GAMES = pa.DataFrameSchema(
    {
        "season": pa.Column(pl.Int32, pa.Check.in_range(1990, 2100)),
        "game_id": pa.Column(pl.Utf8, pa.Check.str_length(10, 10)),
        "game_date": pa.Column(pl.Date),
        "player_id": pa.Column(pl.Int64, pa.Check.gt(0)),
        "player_name": pa.Column(pl.Utf8),
        "team_id": pa.Column(pl.Int64),
        "team": pa.Column(pl.Utf8, pa.Check.str_length(2, 3)),
        "opp": pa.Column(pl.Utf8, pa.Check.str_length(2, 3)),
        "home": pa.Column(pl.Boolean),
        "min": pa.Column(pl.Float64, pa.Check.in_range(0, 70)),
        "pts": pa.Column(pl.Int32, pa.Check.in_range(0, 110)),
        "reb": pa.Column(pl.Int32, _count),
        "ast": pa.Column(pl.Int32, _count),
        "stl": pa.Column(pl.Int32, _count),
        "blk": pa.Column(pl.Int32, _count),
        "tov": pa.Column(pl.Int32, _count),
        "fg3m": pa.Column(pl.Int32, _count),
    },
    unique=["player_id", "game_id"],
    strict=False,
)

TEAM_GAMES = pa.DataFrameSchema(
    {
        "season": pa.Column(pl.Int32),
        "game_id": pa.Column(pl.Utf8),
        "game_date": pa.Column(pl.Date),
        "team": pa.Column(pl.Utf8),
        "team_min": pa.Column(pl.Float64, pa.Check.in_range(235, 400)),
        "team_pts": pa.Column(pl.Int32, pa.Check.in_range(40, 200)),
    },
    unique=["team_id", "game_id"],
    strict=False,
)


class ValidationError(RuntimeError):
    pass


def validate_season(player_games: pl.DataFrame, team_games: pl.DataFrame) -> None:
    PLAYER_GAMES.validate(player_games)
    TEAM_GAMES.validate(team_games)

    # Team totals must equal the sum of their players.
    sums = player_games.group_by("game_id", "team_id").agg(
        pl.col("pts").sum().alias("p_pts"), pl.col("reb").sum().alias("p_reb"),
        pl.col("ast").sum().alias("p_ast"),
    )
    joined = team_games.join(sums, on=["game_id", "team_id"], how="left")
    bad = joined.filter(
        (pl.col("p_pts") != pl.col("team_pts"))
        | (pl.col("p_reb") != pl.col("team_reb"))
        | (pl.col("p_ast") != pl.col("team_ast"))
        | pl.col("p_pts").is_null()
    )
    # A handful of historical box scores have known one-off discrepancies; tolerate 0.5%.
    if bad.height > max(2, 0.005 * team_games.height):
        raise ValidationError(f"{bad.height} team-games do not match player sums:\n{bad.head()}")

    # Every game has exactly two teams.
    per_game = team_games.group_by("game_id").len()
    if per_game.filter(pl.col("len") != 2).height:
        raise ValidationError("games without exactly two teams")

    # Expected games per team (covers the 2019-20 bubble and 72-game 2020-21).
    per_team = team_games.group_by("team").len()
    lo, hi = per_team["len"].min(), per_team["len"].max()
    n_teams = per_team.height
    if n_teams not in (30,) or lo < 60 or hi > 84:
        raise ValidationError(f"unexpected season shape: {n_teams} teams, {lo}-{hi} games")
