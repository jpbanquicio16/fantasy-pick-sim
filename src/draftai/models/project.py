"""Turn fitted models into a projections table for one season (live or historical)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from draftai.features.asof import AsOf
from draftai.features.build import build_features, rookie_rows
from draftai.features.roles import newcomer_minutes, team_minutes_target
from draftai.models.budget import apply_budget, apply_overrides
from draftai.models.rookies import (
    bio_attributes,
    rookie_baseline,
    rookie_context,
    rookie_features,
    rookie_model_projections,
    rookie_projections,
)
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
            feats.select("team_now", "age", "fp_pg_1", "share_1", "gp_1", "mpg_1", "fp_pg_w", "share_w", "last_team",
                         "new_team", "opp_team_mpg", "opp_pos_mpg", "opp_dep", "opp_arr", "opp_new", "opp_ret"),
        ],
        how="horizontal",
    ).with_columns(pl.lit(False).alias("rookie"), pl.lit(q_lg).alias("q_league"))


def model_rookie_features(rk: pl.DataFrame, view: AsOf, roster: pl.DataFrame, bio: pl.DataFrame, draft: pl.DataFrame,
                          target_season: int, attrs: pl.DataFrame | None = None) -> pl.DataFrame:
    """The rookie model's inputs for ``rk`` (player_id, draft_pick, draft_team, team,
    season): background from the draft history, attributes from ``attrs`` and then the
    bio, team context from the as-of view and every player's team at the cut-off."""
    org = (draft.filter(pl.col("draft_year") <= target_season).sort("draft_year").group_by("player_id")
           .agg((pl.col("org_type").last() if "org_type" in draft.columns else pl.lit(None, dtype=pl.Utf8))
                .alias("org_type"), pl.col("player_name").last()))
    known = bio_attributes(bio, target_season)
    if attrs is not None and attrs.height:
        known = pl.concat([attrs.select("player_id", pl.col("age").cast(pl.Float64), pl.col("height_in").cast(pl.Float64),
                                        pl.col("weight_lb").cast(pl.Float64)), known], how="vertical_relaxed")
        known = known.group_by("player_id", maintain_order=True).agg(
            pl.col("age").drop_nulls().first(), pl.col("height_in").drop_nulls().first(),
            pl.col("weight_lb").drop_nulls().first())
    base = rk.select("player_id", "season", "draft_pick", pl.coalesce("team", "draft_team").alias("team")) \
        .join(org, on="player_id", how="left").join(known, on="player_id", how="left")
    ctx = rookie_context(view.player_games, view.team_games, roster, bio, target_season)
    return rookie_features(base, ctx)


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
    rookie_attrs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Veterans from the models, rookies from the rookie model (``models.rookies``; the
    draft-slot baseline when there is none), then the team minutes budget over both
    (``models.minutes_budget`` is its strength; 0 turns it off).

    ``roster`` (player_id, team) is the team at the cut-off. ``extra_rookies``
    (player_id, team) adds players with no NBA history who are not in the draft class
    (undrafted signings, international players); they are projected as undrafted.
    ``rookie_attrs`` (player_id, age, height_in, weight_lb) are the rookies' attributes
    when the NBA's bio has none yet (live: from Sleeper).
    """
    view = AsOf.view(player_games, team_games, as_of)
    recent = view.player_games.filter(pl.col("season") >= target_season - 3).select("player_id").unique()
    vet_pool = None if pool is None else pool.join(recent, on="player_id", how="semi")
    base = rookie_baseline(view.player_games, view.team_games, draft, settings, before_season=target_season)
    # Newcomers first: their minutes are arrivals in the veterans' role features.
    # This year's draft class, plus pool players with no recent NBA games.
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
    newcomers = newcomer_minutes(rk.select("player_id", pl.coalesce("team", "draft_team").alias("team"), "draft_pick"),
                                 base) if rk.height else None

    feats = build_features(view, target_season, roster, bio, draft, settings, pool=vet_pool, newcomers=newcomers,
                           rate_prior=getattr(models, "rate_prior", "bucket"))
    vets = project_rows(feats, models, settings)
    if explain and vets.height:
        from draftai.models.explain import projection_drivers

        vets = vets.join(projection_drivers(models, feats), on="player_id", how="left")
    q_lg = float(vets["q_league"][0]) if vets.height else 0.06
    rooks = None
    if rk.height:
        model = getattr(models, "rookies", None)
        if model is None:
            rooks = rookie_projections(rk, base, q_lg, settings)
        else:
            rf = model_rookie_features(rk, view, teams, bio, draft, target_season, rookie_attrs)
            rooks = rookie_model_projections(rf, model, q_lg, settings)
    out = vets if rooks is None else pl.concat([vets, rooks], how="diagonal_relaxed")
    out = apply_budget(out, team_minutes_target(view.team_games), getattr(models, "minutes_budget", 0.0), settings)
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


def apply_minutes_overrides(proj: pl.DataFrame, path: Path, settings: ScoringSettings,
                            log_dir: Path | None = None, log=print) -> pl.DataFrame:
    """Manual minutes per game for preseason news the model cannot know (a starting job
    announced, a move to the bench). Applied after the model and the team budget, so the
    model and its evaluation are untouched; teammates absorb the change (models/budget.py).

    CSV columns: sleeper_id, mpg, note (optional ``name``, for readability only). Every
    application is logged to ``log_dir/minutes_overrides_log.csv``.
    """
    flags = [pl.lit(False).alias("minutes_override"), pl.lit(None, dtype=pl.Utf8).alias("minutes_note"),
             pl.lit(0.0).alias("override_delta")]
    if not path.exists():
        return proj.with_columns(flags)
    o = pl.read_csv(path, schema_overrides={"sleeper_id": pl.Utf8, "mpg": pl.Float64, "note": pl.Utf8})
    if o.height == 0 or "mpg" not in o.columns:
        return proj.with_columns(flags)
    out, changes = apply_overrides(proj, o, settings)
    for r in changes.iter_rows(named=True):
        if r["status"] != "applied":
            log(f"minutes override for {r['sleeper_id']}: {r['status']}")
            continue
        log(f"minutes override: {r['name']} ({r['team']}) {r['before_mpg']:.1f} -> {r['override_mpg']:.1f} min/game "
            f"(model {r['model_mpg']:.1f}); {r['teammates']} teammates adjusted" + (f"; {r['note']}" if r["note"] else ""))
    if log_dir is not None:
        from datetime import datetime

        log_dir.mkdir(parents=True, exist_ok=True)
        f = log_dir / "minutes_overrides_log.csv"
        stamped = changes.with_columns(pl.lit(datetime.now().isoformat(timespec="seconds")).alias("applied_at"))
        if f.exists():
            stamped = pl.concat([pl.read_csv(f, schema_overrides=stamped.schema), stamped], how="vertical_relaxed")
        stamped.write_csv(f)
    return out
