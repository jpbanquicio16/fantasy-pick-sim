"""Raw responses -> typed tables, stored as Parquet (one file per table and season).

Tables
------
player_games  one row per player per game played (absence from a box score is a DNP)
team_games    one row per team per game
player_bio    one row per player per season (age, height, draft slot)
draft_class   one row per drafted player per draft year
schedule      the target season's regular-season games, one row per team per game
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from draftai.config import Paths
from draftai.ingestion.nba import NbaClient

STAT_RENAME = {
    "MIN": "min", "FGM": "fgm", "FGA": "fga", "FG3M": "fg3m", "FG3A": "fg3a", "FTM": "ftm",
    "FTA": "fta", "OREB": "oreb", "DREB": "dreb", "REB": "reb", "AST": "ast", "STL": "stl",
    "BLK": "blk", "TOV": "tov", "PF": "pf", "PTS": "pts",
}
INT_STATS = [c for c in STAT_RENAME.values() if c != "min"]


def _matchup(df: pl.DataFrame) -> pl.DataFrame:
    """'OKC vs. HOU' is a home game against HOU; 'HOU @ OKC' is away at OKC."""
    return df.with_columns(
        pl.col("MATCHUP").str.contains(" vs. ").alias("home"),
        pl.col("MATCHUP").str.split(" ").list.last().alias("opp"),
    )


def normalise_player_games(raw: pl.DataFrame, season: int) -> pl.DataFrame:
    df = _matchup(raw).rename(STAT_RENAME)
    return df.select(
        pl.lit(season, dtype=pl.Int32).alias("season"),
        pl.col("GAME_ID").cast(pl.Utf8).alias("game_id"),
        pl.col("GAME_DATE").str.to_date().alias("game_date"),
        pl.col("PLAYER_ID").cast(pl.Int64).alias("player_id"),
        pl.col("PLAYER_NAME").alias("player_name"),
        pl.col("TEAM_ID").cast(pl.Int64).alias("team_id"),
        pl.col("TEAM_ABBREVIATION").alias("team"),
        pl.col("opp"),
        pl.col("home"),
        pl.col("min").cast(pl.Float64).fill_null(0.0),
        *[pl.col(c).cast(pl.Int32).fill_null(0) for c in INT_STATS],
    ).sort("game_date", "game_id", "team", "player_id")


def normalise_team_games(raw: pl.DataFrame, season: int) -> pl.DataFrame:
    df = _matchup(raw).rename(STAT_RENAME)
    return df.select(
        pl.lit(season, dtype=pl.Int32).alias("season"),
        pl.col("GAME_ID").cast(pl.Utf8).alias("game_id"),
        pl.col("GAME_DATE").str.to_date().alias("game_date"),
        pl.col("TEAM_ID").cast(pl.Int64).alias("team_id"),
        pl.col("TEAM_ABBREVIATION").alias("team"),
        pl.col("opp"),
        pl.col("home"),
        pl.col("min").cast(pl.Float64).alias("team_min"),
        *[pl.col(c).cast(pl.Int32).fill_null(0).alias(f"team_{c}") for c in INT_STATS],
    ).sort("game_date", "game_id", "team")


def _num(col: str) -> pl.Expr:
    """'Undrafted' and blanks become null."""
    return pl.col(col).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


def normalise_bio(raw: pl.DataFrame, season: int) -> pl.DataFrame:
    return raw.select(
        pl.lit(season, dtype=pl.Int32).alias("season"),
        pl.col("PLAYER_ID").cast(pl.Int64).alias("player_id"),
        pl.col("PLAYER_NAME").alias("player_name"),
        _num("AGE").alias("age"),
        _num("PLAYER_HEIGHT_INCHES").alias("height_in"),
        _num("PLAYER_WEIGHT").alias("weight_lb"),
        _num("DRAFT_YEAR").alias("draft_year"),
        _num("DRAFT_NUMBER").alias("draft_number"),
    ).unique("player_id", keep="first")


def normalise_draft(raw: pl.DataFrame) -> pl.DataFrame:
    return raw.select(
        pl.col("SEASON").cast(pl.Int32).alias("draft_year"),
        pl.col("PERSON_ID").cast(pl.Int64).alias("player_id"),
        pl.col("PLAYER_NAME").alias("player_name"),
        pl.col("OVERALL_PICK").cast(pl.Int32).alias("overall_pick"),
        pl.col("TEAM_ABBREVIATION").alias("team"),
    )


def normalise_schedule(raw: pl.DataFrame, season: int, include_cup_final: bool = False) -> pl.DataFrame:
    """Regular-season games (prefix 002), optionally the NBA Cup final (prefix 006).

    The Cup final does not count toward regular-season statistics; whether Sleeper scores
    it is a league question, so it is opt-in.
    """
    keep = ["002", "006"] if include_cup_final else ["002"]
    games = raw.filter(pl.col("game_id").str.slice(0, 3).is_in(keep)).with_columns(
        pl.col("game_date_est").str.to_date().alias("game_date")
    )
    home = games.select("game_id", "game_date", pl.col("home").alias("team"), pl.col("away").alias("opp"),
                        pl.lit(True).alias("home"))
    away = games.select("game_id", "game_date", pl.col("away").alias("team"), pl.col("home").alias("opp"),
                        pl.lit(False).alias("home"))
    return (
        pl.concat([home, away])
        .with_columns(pl.lit(season, dtype=pl.Int32).alias("season"))
        .sort("game_date", "team")
    )


def schedule_from_team_games(team_games: pl.DataFrame) -> pl.DataFrame:
    """Historical schedules come straight from team game logs."""
    return team_games.select("game_id", "game_date", "team", "opp", "home", "season").sort(
        "game_date", "team"
    )


# --- storage ---------------------------------------------------------------------
def table_path(paths: Paths, table: str, season: int | None = None) -> Path:
    name = f"{table}.parquet" if season is None else f"{table}_{season}.parquet"
    return paths.normalised / table / name


def write_table(df: pl.DataFrame, paths: Paths, table: str, season: int | None = None) -> Path:
    p = table_path(paths, table, season)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)
    return p


def load_table(table: str, paths: Paths | None = None, seasons: list[int] | None = None) -> pl.DataFrame:
    paths = paths or Paths()
    d = paths.normalised / table
    files = sorted(d.glob(f"{table}*.parquet"))
    if seasons is not None:
        wanted = {f"{table}_{s}.parquet" for s in seasons}
        files = [f for f in files if f.name in wanted]
    if not files:
        raise FileNotFoundError(f"no normalised {table!r} tables under {d}; run `draftai ingest` first")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def ingest_seasons(
    seasons: list[int],
    target_season: int,
    client: NbaClient | None = None,
    paths: Paths | None = None,
    include_cup_final: bool = False,
    log=print,
) -> None:
    """Fetch (or read from the raw cache) and normalise every table."""
    from draftai.validation.schemas import validate_season

    paths = (paths or Paths()).ensure()
    client = client or NbaClient()
    for s in seasons:
        pg = normalise_player_games(client.player_game_logs(s), s)
        tg = normalise_team_games(client.team_game_logs(s), s)
        validate_season(pg, tg)
        write_table(pg, paths, "player_games", s)
        write_table(tg, paths, "team_games", s)
        write_table(normalise_bio(client.player_bio(s), s), paths, "player_bio", s)
        log(f"season {s}: {pg.height} player-games, {tg['game_id'].n_unique()} games")
    drafts = [normalise_draft(client.draft_history(y)) for y in range(min(seasons), target_season + 1)]
    write_table(pl.concat(drafts), paths, "draft_class")
    sched = normalise_schedule(client.schedule(target_season), target_season, include_cup_final)
    write_table(sched, paths, "schedule", target_season)
    log(f"schedule {target_season}: {sched['game_id'].n_unique()} games")
