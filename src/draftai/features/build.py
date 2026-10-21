"""Player-season feature rows, computed as of a cut-off.

One training row per (player, target season Y). Every feature is computed from an
``AsOf`` view (games strictly before Y's opener) plus a declared roster snapshot (each
player's team at the cut-off). Targets come from season Y and are joined afterwards.

Component features (minutes, per-minute rates, availability) do not depend on the
league's scoring, so the component models are portable across scoring settings; the
fantasy-point features feed only the direct FP/G and quantile models.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from draftai.features.asof import AsOf, season_cutoff
from draftai.features.roles import newcomer_minutes, opportunity, team_minutes_target
from draftai.scoring import ScoringSettings, score_frame

COMPONENTS = ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m"]
LAG_WEIGHTS = {1: 5.0, 2: 3.0, 3: 2.0}
# Height buckets used as the empirical-Bayes prior for per-minute rates.
HEIGHT_EDGES = (77.5, 80.5)
SHORT_RUN = 2  # missed runs of at most this many games count as rest or minor knocks


# --- per-season aggregates ---------------------------------------------------------
def _b2b_flags(team_games: pl.DataFrame) -> pl.DataFrame:
    """Second night of a back-to-back for every team game."""
    return team_games.sort("season", "team", "game_date").with_columns(
        ((pl.col("game_date") - pl.col("game_date").shift(1).over("season", "team")).dt.total_days() == 1)
        .fill_null(False)
        .alias("b2b2")
    )


def season_stats(player_games: pl.DataFrame, team_games: pl.DataFrame, settings: ScoringSettings) -> pl.DataFrame:
    """Totals, per-game distribution summaries and usage, one row per player-season."""
    pg = score_frame(player_games, settings)
    team = team_games.select(
        "game_id", "team_id", "team_min",
        (pl.col("team_fga") + 0.44 * pl.col("team_fta") + pl.col("team_tov")).alias("team_used"),
    )
    pg = pg.join(team, on=["game_id", "team_id"], how="left").with_columns(
        (pl.col("fga") + 0.44 * pl.col("fta") + pl.col("tov")).alias("used"),
        (pl.col("team_used") * pl.col("min") / (pl.col("team_min") / 5.0)).alias("team_used_share"),
        (
            sum((pl.col(c) >= 10).cast(pl.Int32) for c in ["pts", "reb", "ast", "stl", "blk"]) >= 2
        ).alias("dd"),
    )
    stats = COMPONENTS + ["fga", "fta", "fgm", "ftm", "oreb", "dreb"]
    return (
        pg.sort("game_date")
        .group_by("player_id", "season")
        .agg(
            pl.len().alias("gp"),
            pl.col("min").sum().alias("min"),
            *[pl.col(c).sum().alias(c) for c in stats],
            pl.col("fp").sum().alias("fp"),
            pl.col("fp_bonus").sum().alias("fp_bonus"),
            pl.col("fp").std().alias("fp_sd"),
            pl.col("fp").quantile(0.5).alias("fp_p50"),
            pl.col("fp").quantile(0.9).alias("fp_p90"),
            pl.col("dd").mean().alias("dd_rate"),
            (pl.col("fp") >= 50).mean().alias("fp50_rate"),
            (pl.col("used").sum() / pl.col("team_used_share").sum()).alias("usage"),
            pl.col("min").tail(20).mean().alias("min_last20"),
            pl.col("team").first().alias("first_team"),
            pl.col("team").last().alias("last_team"),
            pl.col("player_name").last(),
        )
        .with_columns(
            (pl.col("min") / pl.col("gp")).alias("mpg"),
            (pl.col("fp") / pl.col("gp")).alias("fp_pg"),
        )
    )


def availability_stats(player_games: pl.DataFrame, team_games: pl.DataFrame) -> pl.DataFrame:
    """Games missed, longest absence and short-absence (rest) rate per player-season.

    A player's schedule runs from his first team's opener (or his first game, for later
    stints) to his last team's final game. Missing from a box score is a DNP.
    """
    tg = _b2b_flags(team_games)
    bounds = tg.group_by("season", "team").agg(
        pl.col("game_date").min().alias("t_first"), pl.col("game_date").max().alias("t_last")
    )
    stints = (
        player_games.group_by("player_id", "season", "team")
        .agg(pl.col("game_date").min().alias("first"), pl.col("game_date").max().alias("last"))
        .sort("player_id", "season", "first")
        .with_columns(
            (pl.col("first").rank("ordinal").over("player_id", "season") == 1).alias("is_first"),
            (pl.col("first").rank("ordinal", descending=True).over("player_id", "season") == 1).alias("is_last"),
        )
        .join(bounds, on=["season", "team"], how="left")
        .with_columns(
            pl.when(pl.col("is_first")).then(pl.col("t_first")).otherwise(pl.col("first")).alias("w_start"),
            pl.when(pl.col("is_last")).then(pl.col("t_last")).otherwise(pl.col("last")).alias("w_end"),
        )
    )
    sched = (
        stints.select("player_id", "season", "team", "w_start", "w_end")
        .join(tg.select("season", "team", "game_id", "game_date", "b2b2"), on=["season", "team"])
        .filter(pl.col("game_date").is_between(pl.col("w_start"), pl.col("w_end")))
        .unique(["player_id", "season", "game_date"], keep="first")
    )
    played = player_games.select("player_id", "game_id").with_columns(pl.lit(True).alias("played"))
    sched = (
        sched.join(played, on=["player_id", "game_id"], how="left")
        .with_columns(pl.col("played").fill_null(False))
        .sort("player_id", "season", "game_date")
        .with_columns((~pl.col("played")).alias("missed"))
        .with_columns(
            (pl.col("missed") != pl.col("missed").shift(1).over("player_id", "season"))
            .fill_null(True)
            .cast(pl.Int32)
            .cum_sum()
            .over("player_id", "season")
            .alias("run")
        )
        .with_columns(pl.len().over("player_id", "season", "run").alias("run_len"))
        .with_columns(
            (pl.col("missed") & (pl.col("run_len") > SHORT_RUN)).alias("long_miss"),
            (pl.col("missed") & (pl.col("run_len") <= SHORT_RUN)).alias("short_miss"),
        )
    )
    return sched.group_by("player_id", "season").agg(
        pl.len().alias("n_sched"),
        pl.when(pl.col("missed")).then(pl.col("run_len")).otherwise(0).max().alias("longest_absence"),
        pl.col("long_miss").sum().alias("long_missed"),
        pl.col("short_miss").sum().alias("short_missed"),
        (pl.col("short_miss") & pl.col("b2b2")).sum().alias("short_missed_b2b"),
        (pl.col("b2b2") & ~pl.col("long_miss")).sum().alias("b2b_avail"),
    ).with_columns(
        (pl.col("short_missed") / (pl.col("n_sched") - pl.col("long_missed")).clip(1)).alias("q_short")
    )


def team_season_stats(team_games: pl.DataFrame) -> pl.DataFrame:
    return (
        team_games.with_columns(
            (pl.col("team_fga") + 0.44 * pl.col("team_fta") - pl.col("team_oreb") + pl.col("team_tov")).alias("poss")
        )
        .group_by("season", "team")
        .agg(
            pl.len().alias("team_gp"),
            (48.0 * pl.col("poss").sum() / (pl.col("team_min").sum() / 5.0)).alias("pace"),
        )
    )


def league_b2b_multiplier(avail: pl.DataFrame) -> float:
    """How much more likely a short absence is on the second night of a back-to-back."""
    s = avail.select(
        pl.col("short_missed_b2b").sum().alias("mb"),
        pl.col("b2b_avail").sum().alias("nb"),
        (pl.col("short_missed") - pl.col("short_missed_b2b")).sum().alias("mo"),
        (pl.col("n_sched") - pl.col("long_missed") - pl.col("b2b_avail")).sum().alias("no"),
    ).row(0, named=True)
    if not s["nb"] or not s["mo"]:
        return 1.0
    return float((s["mb"] / s["nb"]) / (s["mo"] / s["no"]))


# --- empirical-Bayes shrinkage -------------------------------------------------------
def height_bucket(height: pl.Expr) -> pl.Expr:
    return (
        pl.when(height.is_null()).then(pl.lit("unk"))
        .when(height < HEIGHT_EDGES[0]).then(pl.lit("guard"))
        .when(height < HEIGHT_EDGES[1]).then(pl.lit("wing"))
        .otherwise(pl.lit("big"))
    )


def eb_prior(counts: np.ndarray, minutes: np.ndarray) -> tuple[float, float]:
    """Method-of-moments prior for per-minute Poisson rates: (mean, strength k in minutes).

    Observed rate variance = true variance + sampling variance (mean / minutes); the
    prior strength is mean / true variance, so noisier stats (steals, blocks) get pulled
    harder toward the prior.
    """
    ok = minutes > 0
    c, m = counts[ok], minutes[ok]
    if c.size < 5:
        return (float(c.sum() / max(m.sum(), 1.0)), 500.0)
    mu = c.sum() / m.sum()
    r = c / m
    var_obs = np.average((r - mu) ** 2, weights=m)
    var_samp = np.average(mu / m, weights=m)
    tau2 = max(var_obs - var_samp, 1e-3 * mu**2 + 1e-12)
    return float(mu), float(min(mu / tau2, 5000.0))


# --- feature rows --------------------------------------------------------------------
def _bucket_rates(ss: pl.DataFrame, rows: pl.DataFrame) -> list[pl.Expr]:
    """Empirical-Bayes shrinkage of weighted per-minute rates toward the height bucket."""
    hist = ss.join(rows.select("player_id", "hbucket"), on="player_id", how="left").with_columns(
        pl.col("hbucket").fill_null("unk")
    )
    shrunk = []
    for c in COMPONENTS:
        exprs = []
        for hb in ["guard", "wing", "big", "unk"]:
            h = hist.filter(pl.col("hbucket") == hb) if hb != "unk" else hist
            mu, k = eb_prior(h[c].to_numpy().astype(float), h["min"].to_numpy().astype(float))
            exprs.append((hb, mu, k))
        expr = pl.lit(None, dtype=pl.Float64)
        for hb, mu, k in exprs:
            expr = pl.when(pl.col("hbucket") == hb).then(
                (pl.col(f"{c}_wsum") + k * mu) / (pl.col("min_w") + k)
            ).otherwise(expr)
        shrunk.append(expr.alias(f"rate_{c}"))
    return shrunk


def _lagged(seasons: pl.DataFrame, target: int, lag: int, cols: list[str]) -> pl.DataFrame:
    return seasons.filter(pl.col("season") == target - lag).select(
        "player_id", *[pl.col(c).alias(f"{c}_{lag}") for c in cols]
    )


def build_features(
    view: AsOf,
    target_season: int,
    roster: pl.DataFrame,
    bio: pl.DataFrame,
    draft: pl.DataFrame,
    settings: ScoringSettings,
    pool: pl.DataFrame | None = None,
    newcomers: pl.DataFrame | None = None,
    rate_prior: str = "bucket",
) -> pl.DataFrame:
    """Feature rows for ``target_season`` from games strictly before ``view.as_of``.

    ``roster`` (player_id, team) is the team each player is on at the cut-off. ``pool``
    (player_id) restricts the rows; by default every player with a game in either of the
    two prior seasons, including those who would go on to miss the season. ``newcomers``
    (player_id, team, mpg, share) are the rookies and other players without recent NBA
    games joining each team, at their draft-slot expected minutes (``newcomer_minutes``).
    """
    Y = target_season
    pg, tg = view.player_games, view.team_games
    pg = pg.filter(pl.col("season") >= Y - 3)
    tg = tg.filter(pl.col("season") >= Y - 3)
    ss = season_stats(pg, tg, settings)
    av = availability_stats(pg, tg)
    ss = ss.join(av, on=["player_id", "season"], how="left").with_columns(
        (pl.col("gp") / pl.col("n_sched").clip(1)).clip(0, 1).alias("share")
    )
    first_season = pg.group_by("player_id").agg(pl.col("season").min().alias("first_season"))

    if pool is None:
        pool = ss.filter(pl.col("season") >= Y - 2).select("player_id").unique()
    rows = pool.select("player_id").unique().join(first_season, on="player_id", how="left")

    lag_cols = ["gp", "min", "mpg", "share", "fp_pg", "fp", "fp_bonus", "usage", *COMPONENTS, "fga", "fta"]
    for lag in LAG_WEIGHTS:
        rows = rows.join(_lagged(ss, Y, lag, lag_cols), on="player_id", how="left")
        # In the league but no games that season: zero games, not unknown.
        in_league = pl.col("first_season") <= Y - lag
        rows = rows.with_columns(
            *[
                pl.when(pl.col(f"{c}_{lag}").is_null() & in_league).then(0.0).otherwise(pl.col(f"{c}_{lag}")).alias(f"{c}_{lag}")
                for c in ["gp", "min", "share", "fp", "fp_bonus", *COMPONENTS, "fga", "fta"]
            ]
        )
    last = ss.filter(pl.col("season") == Y - 1).select(
        "player_id", "fp_sd", "fp_p50", "fp_p90", "dd_rate", "fp50_rate", "min_last20", "longest_absence",
        "q_short", "n_sched",
    )
    last_team = ss.sort("season").group_by("player_id").agg(pl.col("last_team").last())
    rows = rows.join(last, on="player_id", how="left").join(last_team, on="player_id", how="left")

    # Marcel-style weighted aggregates (5/3/2 by season).
    def wsum(col: str) -> pl.Expr:
        return sum(LAG_WEIGHTS[l] * pl.col(f"{col}_{l}").fill_null(0.0) for l in LAG_WEIGHTS)

    w_in = sum(
        LAG_WEIGHTS[l] * pl.col(f"share_{l}").is_not_null().cast(pl.Float64) for l in LAG_WEIGHTS
    )
    rows = rows.with_columns(
        wsum("min").alias("min_w"),
        wsum("gp").alias("gp_w"),
        (wsum("share") / w_in.clip(1e-9)).alias("share_w"),
        *[wsum(c).alias(f"{c}_wsum") for c in [*COMPONENTS, "fga", "fta", "fp", "fp_bonus"]],
    ).with_columns(
        (pl.col("min_w") / pl.col("gp_w").clip(1e-9)).alias("mpg_w"),
        (pl.col("fp_wsum") / pl.col("gp_w").clip(1e-9)).alias("fp_pg_w"),
        (pl.col("fp_bonus_wsum") / pl.col("gp_w").clip(1e-9)).alias("bonus_pg_w"),
        (36.0 * pl.col("fp_wsum") / pl.col("min_w").clip(1e-9)).alias("fp36_w"),
        ((pl.col("fga_wsum") + 0.44 * pl.col("fta_wsum") + pl.col("tov_wsum")) / pl.col("min_w").clip(1e-9)).alias("used_per_min_w"),
        (pl.col("mpg_1") - pl.col("mpg_2")).alias("mpg_trend"),
        (pl.col("min_last20") - pl.col("mpg_1")).alias("last20_minus_mpg"),
        (pl.col("fp_sd") / pl.col("fp_pg_1").clip(1.0)).alias("fp_cv_1"),
        (pl.col("fp_p90") - pl.col("fp_p50")).alias("fp_tail_1"),
    )

    # Bio: only seasons before Y (a bio row for Y would reveal that he played in Y).
    b = (
        bio.filter(pl.col("season") < Y)
        .sort("season")
        .group_by("player_id")
        .agg(
            pl.col("age").drop_nulls().last().alias("age_last"),
            pl.col("season").filter(pl.col("age").is_not_null()).last().alias("age_season"),
            pl.col("height_in").drop_nulls().last().alias("height_in"),
            pl.col("draft_year").drop_nulls().last().alias("bio_draft_year"),
            pl.col("draft_number").drop_nulls().last().alias("bio_draft_pick"),
        )
    )
    d = draft.filter(pl.col("draft_year") <= Y).group_by("player_id").agg(
        pl.col("overall_pick").min().alias("draft_pick"), pl.col("draft_year").min()
    )
    rows = (
        rows.join(b, on="player_id", how="left")
        .join(d, on="player_id", how="left")
        .with_columns(
            pl.coalesce("draft_year", "bio_draft_year").alias("draft_year"),
            pl.coalesce("draft_pick", "bio_draft_pick").alias("draft_pick"),
        )
        .with_columns(
            (pl.col("age_last") + (Y - pl.col("age_season"))).alias("age"),
            (Y - pl.coalesce(pl.col("draft_year"), pl.col("first_season"))).alias("experience"),
            pl.col("draft_pick").fill_null(75).log().alias("log_draft_pick"),
            height_bucket(pl.col("height_in")).alias("hbucket"),
        )
        .with_columns((pl.col("age") ** 2).alias("age2"))
    )

    if rate_prior.startswith("hierarchical"):
        # Shrinkage toward a regression prior: size, age, experience, draft slot (features/rates.py).
        from draftai.features.rates import hierarchical_rates

        people = (ss.select("player_id").unique().join(b, on="player_id", how="left").join(d, on="player_id", how="left")
                  .join(first_season, on="player_id", how="left")
                  .with_columns(pl.coalesce("draft_year", "bio_draft_year").alias("draft_year"),
                                pl.coalesce("draft_pick", "bio_draft_pick").alias("draft_pick")))
        hist = ss.select("player_id", "season", "min", *COMPONENTS).join(people, on="player_id", how="left").with_columns(
            (pl.col("age_last") + (pl.col("season") - pl.col("age_season"))).alias("age"),
            (pl.col("season") - pl.coalesce(pl.col("draft_year"), pl.col("first_season"))).alias("experience"),
            height_bucket(pl.col("height_in")).alias("hbucket"))
        rows = rows.with_columns(hierarchical_rates(rows, hist, COMPONENTS, Y, scaled=rate_prior == "hierarchical_scaled"))
    else:
        rows = rows.with_columns(_bucket_rates(ss, rows))

    # Team context at the cut-off.
    ts = team_season_stats(tg)
    rows = rows.join(roster.select("player_id", pl.col("team").alias("team_now")), on="player_id", how="left")
    rows = rows.with_columns(pl.coalesce("team_now", "last_team").alias("team_now"))
    rows = rows.with_columns(
        (pl.col("last_team").is_not_null() & (pl.col("team_now") != pl.col("last_team"))).cast(pl.Int8).alias("new_team")
    )
    rows = rows.join(
        ts.filter(pl.col("season") == Y - 1).select(pl.col("team").alias("team_now"), pl.col("pace").alias("team_pace_1"),
                                                    pl.col("team_gp").alias("team_gp_1")),
        on="team_now", how="left",
    )
    # Role change: minutes his team opened up or filled since last season (features/roles.py).
    rows = opportunity(rows, pg.filter(pl.col("season") == Y - 1), newcomers).with_columns(
        pl.lit(team_minutes_target(tg)).alias("team_min_target"))
    # Ratios over zero minutes (players logged with 0:00) are unknown, not infinite.
    floats = [c for c, t in rows.schema.items() if t in (pl.Float64, pl.Float32)]
    rows = rows.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in floats])
    # Deterministic row order: LightGBM bagging depends on it.
    return rows.with_columns(pl.lit(Y, dtype=pl.Int32).alias("season"), pl.lit(view.as_of).alias("as_of")).sort("player_id")


# Feature lists for the models. Component models avoid fantasy-point features.
COMPONENT_FEATURES = [
    "gp_1", "gp_2", "gp_3", "mpg_1", "mpg_2", "mpg_3", "share_1", "share_2", "share_3",
    "min_w", "mpg_w", "share_w", "mpg_trend", "min_last20", "last20_minus_mpg", "usage_1", "usage_2",
    "used_per_min_w", "longest_absence", "q_short", "age", "age2", "experience", "log_draft_pick",
    "height_in", "new_team", "team_pace_1", "opp_team_mpg", "opp_pos_mpg", "opp_own_mpg",
    *[f"rate_{c}" for c in COMPONENTS],
]
FP_FEATURES = COMPONENT_FEATURES + [
    "fp_pg_1", "fp_pg_2", "fp_pg_3", "fp_pg_w", "fp36_w", "bonus_pg_w", "fp_sd", "fp_cv_1", "fp_tail_1",
    "dd_rate", "fp50_rate",
]


# --- targets ---------------------------------------------------------------------------
def build_targets(player_games: pl.DataFrame, team_games: pl.DataFrame, season: int,
                  settings: ScoringSettings) -> tuple[pl.DataFrame, float]:
    """Season-Y outcomes per player, and the median team games that season."""
    pg = player_games.filter(pl.col("season") == season)
    tg = team_games.filter(pl.col("season") == season)
    ss = season_stats(pg, tg, settings).join(availability_stats(pg, tg), on=["player_id", "season"], how="left")
    team_gp = float(tg.group_by("team").len()["len"].median())
    return ss.select(
        "player_id",
        pl.col("gp").alias("y_gp"),
        pl.col("min").alias("y_min"),
        (pl.col("gp") / pl.col("n_sched").clip(1)).clip(0, 1).alias("y_share"),
        pl.col("mpg").alias("y_mpg"),
        pl.col("fp_pg").alias("y_fp_pg"),
        pl.col("fp").alias("y_fp"),
        pl.col("fp_sd").alias("y_fp_sd"),
        (pl.col("fp_bonus") / pl.col("gp")).alias("y_bonus_pg"),
        *[(pl.col(c) / pl.col("min").clip(1e-9)).alias(f"y_rate_{c}") for c in COMPONENTS],
        pl.col("first_team").alias("y_first_team"),
    ), team_gp


def roster_from_first_game(player_games: pl.DataFrame, season: int) -> pl.DataFrame:
    """Historical stand-in for the roster at the cut-off: each player's first team in the
    season. Rosters are set by opening night, so this differs from the true cut-off team
    only for players traded before their first game; it is an explicit, separate input,
    never read from the feature builder's game history."""
    return (
        player_games.filter(pl.col("season") == season)
        .sort("game_date")
        .group_by("player_id")
        .agg(pl.col("team").first())
    )


def build_training_set(
    player_games: pl.DataFrame,
    team_games: pl.DataFrame,
    bio: pl.DataFrame,
    draft: pl.DataFrame,
    settings: ScoringSettings,
    seasons: list[int],
    log=lambda *_: None,
    rate_prior: str = "bucket",
) -> pl.DataFrame:
    frames = []
    for Y in seasons:
        cut = season_cutoff(team_games, Y)
        view = AsOf.view(player_games, team_games, cut)
        roster = roster_from_first_game(player_games, Y)
        feats = build_features(view, Y, roster, bio, draft, settings,
                               newcomers=draft_class_newcomers(view, draft, Y, roster, settings), rate_prior=rate_prior)
        targets, team_gp = build_targets(player_games, team_games, Y, settings)
        rows = feats.join(targets, on="player_id", how="left").with_columns(
            pl.col("y_gp").fill_null(0),
            pl.col("y_min").fill_null(0.0),
            pl.col("y_share").fill_null(0.0),
            pl.lit(team_gp).alias("y_team_gp"),
        )
        frames.append(rows)
        log(f"features {Y}: {rows.height} rows (as of {cut})")
    return pl.concat(frames, how="diagonal_relaxed").sort("season", "player_id")


def draft_class_newcomers(view: AsOf, draft: pl.DataFrame, target_season: int, roster: pl.DataFrame,
                          settings: ScoringSettings) -> pl.DataFrame:
    """The season's draft class at their draft-slot expected minutes (from classes before
    it), on their team at the cut-off (else their draft team)."""
    from draftai.models.rookies import rookie_baseline

    base = rookie_baseline(view.player_games, view.team_games, draft, settings, before_season=target_season)
    rk = draft.filter(pl.col("draft_year") == target_season).select(
        "player_id", pl.col("overall_pick").alias("draft_pick"), pl.col("team").alias("draft_team")
    ).join(roster.select("player_id", "team"), on="player_id", how="left")
    return newcomer_minutes(rk.select("player_id", pl.coalesce("team", "draft_team").alias("team"), "draft_pick"), base)


def rookie_rows(draft: pl.DataFrame, target_season: int, roster: pl.DataFrame | None = None) -> pl.DataFrame:
    """Players drafted into ``target_season`` (no NBA history)."""
    r = draft.filter(pl.col("draft_year") == target_season).select(
        "player_id", "player_name", pl.col("overall_pick").alias("draft_pick"), pl.col("team").alias("draft_team")
    )
    if roster is not None:
        r = r.join(roster.select("player_id", "team"), on="player_id", how="left")
    return r.with_columns(pl.lit(target_season, dtype=pl.Int32).alias("season"))


def today_cutoff(team_games: pl.DataFrame, target_season: int, today: date) -> date:
    """Cut-off for a live projection: the earlier of today and the target season's opener."""
    opener = team_games.filter(pl.col("season") == target_season)["game_date"].min()
    return min(today, opener) if opener is not None else today
