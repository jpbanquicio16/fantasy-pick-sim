"""Turn fitted models into a projections table for one season (live or historical)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from draftai.features.asof import AsOf
from draftai.features.build import COMPONENTS, build_features, rookie_rows
from draftai.models.rookies import pick_bucket, rookie_baseline
from draftai.scoring import ScoringSettings

Q_PRIOR_GAMES = 20.0  # shrinkage of the short-absence rate toward the league mean, in games


def _quantile_cols(prefix: str, q: np.ndarray) -> list[pl.Series]:
    return [pl.Series(f"{prefix}_p10", q[:, 0]), pl.Series(f"{prefix}_p50", q[:, 1]), pl.Series(f"{prefix}_p90", q[:, 2])]


def project_rows(feats: pl.DataFrame, models, settings: ScoringSettings) -> pl.DataFrame:
    """Point projections, calibrated quantiles and a shrunk rest rate per feature row."""
    pred = models.projection.predict(feats, settings)
    fpq = models.fp_q.predict(feats, point=pred["fp_pg_hat"].to_numpy())
    shq = models.share_q.predict(feats)
    played = feats.filter(pl.col("gp_1") > 0)
    q_lg = float((played["q_short"].fill_null(0) * played["gp_1"]).sum() / max(played["gp_1"].sum(), 1))
    q = feats.select(
        ((pl.col("q_short").fill_null(q_lg) * pl.col("gp_1").fill_null(0) + Q_PRIOR_GAMES * q_lg)
         / (pl.col("gp_1").fill_null(0) + Q_PRIOR_GAMES)).alias("q_short_hat")
    )
    return pl.concat(
        [
            pred,
            pl.DataFrame(_quantile_cols("fp", fpq) + _quantile_cols("share", shq)),
            q,
            feats.select("team_now", "age", "fp_pg_1", "share_1", "gp_1", "mpg_1"),
        ],
        how="horizontal",
    ).with_columns(pl.lit(False).alias("rookie"), pl.lit(q_lg).alias("q_league"))


def rookie_projections(
    rookies: pl.DataFrame, baseline: pl.DataFrame, q_league: float, settings: ScoringSettings
) -> pl.DataFrame:
    """Draft-slot baseline: each rookie gets his pick bucket's first-year outcomes."""
    from draftai.models.components import linear_fp

    known = set(baseline["bucket"].to_list())
    # Undrafted signings have no draft-class history; the second round is the closest proxy.
    r = rookies.with_columns(pick_bucket(pl.col("draft_pick")).alias("bucket")).with_columns(
        pl.when(pl.col("bucket").is_in(list(known))).then(pl.col("bucket")).otherwise(pl.lit("31-60")).alias("bucket")
    ).join(baseline, on="bucket", how="left")
    rates = {c: r[f"rate_{c}"].fill_null(0).to_numpy() for c in COMPONENTS}
    mpg = r["mpg"].fill_null(10.0).to_numpy()
    lin = linear_fp(rates, mpg, settings) + r["bonus_pg"].fill_null(0).to_numpy()
    return r.select(
        "player_id", "season",
        pl.col("mpg").alias("mpg_hat"),
        pl.col("share").alias("share_hat"),
        *[pl.col(f"rate_{c}").alias(f"rate_{c}_hat") for c in COMPONENTS],
        pl.col("bonus_pg").alias("bonus_hat"),
        pl.Series("fp_pg_hat", lin),
        pl.Series("fp_direct_hat", lin),
        pl.col("fp_pg_p10").alias("fp_p10"), pl.col("fp_pg_p50").alias("fp_p50"), pl.col("fp_pg_p90").alias("fp_p90"),
        pl.col("share_p10"), pl.col("share").alias("share_p50"), pl.col("share_p90"),
        pl.lit(q_league).alias("q_short_hat"),
        pl.coalesce("team", "draft_team").alias("team_now") if "team" in r.columns else pl.col("draft_team").alias("team_now"),
        pl.lit(None, dtype=pl.Float64).alias("age"),
        pl.lit(None, dtype=pl.Float64).alias("fp_pg_1"),
        pl.lit(None, dtype=pl.Float64).alias("share_1"),
        pl.lit(None, dtype=pl.Float64).alias("gp_1"),
        pl.lit(None, dtype=pl.Float64).alias("mpg_1"),
        pl.lit(True).alias("rookie"),
        pl.lit(q_league).alias("q_league"),
        pl.col("donor_ids"),
        pl.col("bucket"),
    )


def season_projections(
    player_games: pl.DataFrame,
    team_games: pl.DataFrame,
    bio: pl.DataFrame,
    draft: pl.DataFrame,
    models,
    settings: ScoringSettings,
    target_season: int,
    as_of: date,
    roster: pl.DataFrame,
    pool: pl.DataFrame | None = None,
    extra_rookies: pl.DataFrame | None = None,
    explain: bool = False,
) -> pl.DataFrame:
    """Veterans from the models, rookies from the draft-slot baseline.

    ``roster`` (player_id, team) is the team at the cut-off. ``extra_rookies``
    (player_id, team) adds players with no NBA history who are not in the draft class
    (undrafted signings, international players); they get the undrafted baseline.
    """
    view = AsOf.view(player_games, team_games, as_of)
    recent = view.player_games.filter(pl.col("season") >= target_season - 3).select("player_id").unique()
    vet_pool = None if pool is None else pool.join(recent, on="player_id", how="semi")
    feats = build_features(view, target_season, roster, bio, draft, settings, pool=vet_pool)
    vets = project_rows(feats, models, settings)
    if explain and vets.height:
        from draftai.models.explain import projection_drivers

        vets = vets.join(projection_drivers(models, feats), on="player_id", how="left")
    q_lg = float(vets["q_league"][0]) if vets.height else 0.06
    base = rookie_baseline(view.player_games, view.team_games, draft, settings, before_season=target_season)
    # Newcomers: this year's draft class, plus pool players with no recent NBA games.
    if pool is None:
        new_ids = rookie_rows(draft, target_season).select("player_id")
    else:
        new_ids = pool.join(recent, on="player_id", how="anti").select("player_id")
    if extra_rookies is not None and extra_rookies.height:
        new_ids = pl.concat([new_ids, extra_rookies.select("player_id")]).unique()
    picks = draft.filter(pl.col("draft_year") <= target_season).group_by("player_id").agg(
        pl.col("overall_pick").min().alias("draft_pick"), pl.col("team").last().alias("draft_team"))
    teams = roster.select("player_id", "team")
    if extra_rookies is not None and extra_rookies.height:
        teams = pl.concat([teams, extra_rookies.select("player_id", "team")]).unique("player_id")
    rk = new_ids.join(picks, on="player_id", how="left").join(teams, on="player_id", how="left").with_columns(
        pl.lit(target_season, dtype=pl.Int32).alias("season"))
    rooks = rookie_projections(rk, base, q_lg, settings) if rk.height else None
    out = vets if rooks is None else pl.concat([vets, rooks], how="diagonal_relaxed")
    return out.with_columns(pl.lit(as_of).alias("as_of"))


def apply_availability_overrides(proj: pl.DataFrame, path: Path, id_col: str = "sleeper_id") -> pl.DataFrame:
    """Manual games-share overrides (e.g. a known season-ending injury, or a return).

    CSV columns: sleeper_id, share (absolute, 0-1) or multiplier, note.
    """
    if not path.exists():
        return proj
    o = pl.read_csv(path, schema_overrides={id_col: pl.Utf8, "share": pl.Float64, "multiplier": pl.Float64})
    if o.height == 0:
        return proj
    p = proj.join(o.select(id_col, "share", "multiplier"), on=id_col, how="left")
    new = (
        pl.when(pl.col("share").is_not_null()).then(pl.col("share"))
        .when(pl.col("multiplier").is_not_null()).then(pl.col("share_hat") * pl.col("multiplier"))
        .otherwise(pl.col("share_hat")).clip(0, 1)
    )
    ratio = (new / pl.col("share_hat").clip(1e-6))
    return p.with_columns(
        new.alias("share_hat"),
        (pl.col("share_p10") * ratio).clip(0, 1).alias("share_p10"),
        (pl.col("share_p50") * ratio).clip(0, 1).alias("share_p50"),
        (pl.col("share_p90") * ratio).clip(0, 1).alias("share_p90"),
        (pl.col("share").is_not_null() | pl.col("multiplier").is_not_null()).alias("availability_override"),
    ).drop("share", "multiplier")
