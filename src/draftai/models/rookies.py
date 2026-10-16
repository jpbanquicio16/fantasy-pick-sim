"""Draft-slot baseline for rookies (the separate rookie model is a later version).

Historical first-year outcomes by draft-pick bucket: FP/G, minutes, games share and
per-minute rates, plus the bucket's own spread for uncertainty.
"""

from __future__ import annotations

import polars as pl

from draftai.features.build import COMPONENTS, availability_stats, season_stats
from draftai.scoring import ScoringSettings

PICK_BUCKETS = [(1, 3), (4, 7), (8, 14), (15, 30), (31, 60)]


def pick_bucket(pick: pl.Expr) -> pl.Expr:
    expr = pl.lit("undrafted")
    for lo, hi in reversed(PICK_BUCKETS):
        expr = pl.when(pick.is_between(lo, hi)).then(pl.lit(f"{lo}-{hi}")).otherwise(expr)
    return pl.when(pick.is_null()).then(pl.lit("undrafted")).otherwise(expr)


def rookie_baseline(
    player_games: pl.DataFrame,
    team_games: pl.DataFrame,
    draft: pl.DataFrame,
    settings: ScoringSettings,
    before_season: int,
) -> pl.DataFrame:
    """First-year outcomes by pick bucket for draft classes before ``before_season``."""
    classes = draft.filter(pl.col("draft_year") < before_season)
    seasons = classes["draft_year"].unique().to_list()
    pg = player_games.filter(pl.col("season").is_in(seasons))
    tg = team_games.filter(pl.col("season").is_in(seasons))
    ss = season_stats(pg, tg, settings).join(availability_stats(pg, tg), on=["player_id", "season"], how="left")
    team_gp = tg.group_by("season", "team").len().group_by("season").agg(pl.col("len").median().alias("team_gp"))
    rows = (
        classes.join(ss, left_on=["player_id", "draft_year"], right_on=["player_id", "season"], how="left")
        .join(team_gp, left_on="draft_year", right_on="season", how="left")
        .with_columns(
            pick_bucket(pl.col("overall_pick")).alias("bucket"),
            pl.col("gp").fill_null(0),
            pl.col("min").fill_null(0.0),
        )
        .with_columns((pl.col("gp") / pl.col("team_gp")).clip(0, 1).alias("share"))
    )
    played = pl.col("gp") >= 10
    return rows.group_by("bucket").agg(
        pl.len().alias("n"),
        pl.col("share").mean().alias("share"),
        pl.col("share").quantile(0.1).alias("share_p10"),
        pl.col("share").quantile(0.9).alias("share_p90"),
        (pl.col("min").filter(played).sum() / pl.col("gp").filter(played).sum()).alias("mpg"),
        pl.col("fp_pg").filter(played).mean().alias("fp_pg"),
        pl.col("fp_pg").filter(played).quantile(0.1).alias("fp_pg_p10"),
        pl.col("fp_pg").filter(played).quantile(0.5).alias("fp_pg_p50"),
        pl.col("fp_pg").filter(played).quantile(0.9).alias("fp_pg_p90"),
        (pl.col("fp_bonus").filter(played).sum() / pl.col("gp").filter(played).sum()).alias("bonus_pg"),
        *[
            (pl.col(c).filter(played).sum() / pl.col("min").filter(played).sum()).alias(f"rate_{c}")
            for c in COMPONENTS
        ],
        pl.col("player_id").filter(played).alias("donor_ids"),
    ).sort("bucket")
