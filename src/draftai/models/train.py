"""Training pipeline: rolling-origin validation, one test-season evaluation, final fit.

1. For each validation season V: fit on target seasons before V, predict V.
2. Choose blend weights, the team minutes budget's strength and conformal corrections
   from those out-of-fold predictions.
3. Freeze everything; fit on seasons before the test season; evaluate the test season once.
4. Fit on every season through the test season; these models project the draft season.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from draftai.config import TrainConfig
from draftai.features.build import COMPONENT_FEATURES, COMPONENTS, FP_FEATURES
from draftai.models import baselines
from draftai.models.budget import STRENGTH_GRID, allocate
from draftai.models.components import ProjectionModel, blended_from_parts, linear_fp
from draftai.models.evaluate import (
    choose_blends,
    draftable_mask,
    mae,
    paired_bootstrap,
    rmse,
    spearman,
)
from draftai.models.quantiles import QuantileModel, coverage, crps_from_quantiles, pinball
from draftai.scoring import ScoringSettings

FP_MIN_GP = 10  # FP/G is evaluated on players with at least this many games


@dataclass
class FoldResult:
    season: int
    rows: pl.DataFrame
    parts: dict[str, tuple[np.ndarray, np.ndarray]]
    fp_q_raw: np.ndarray
    share_q_raw: np.ndarray
    ridge_base: np.ndarray


@dataclass
class Models:
    projection: ProjectionModel
    fp_q: QuantileModel
    share_q: QuantileModel
    minutes_budget: float = 0.0  # strength of the team minutes budget (models/budget.py)


@dataclass
class TrainResult:
    blends: dict[str, float]
    fp_adj: tuple[float, float]
    share_adj: tuple[float, float]
    validation: pl.DataFrame
    test: pl.DataFrame
    uncertainty: pl.DataFrame
    test_rows: pl.DataFrame
    models_test: Models
    models_final: Models
    budget_strength: float = 0.0
    budget_grid: pl.DataFrame | None = None
    cohorts: pl.DataFrame | None = None
    summary: dict = field(default_factory=dict)


def _new_models(cfg: TrainConfig) -> Models:
    q = tuple(cfg.quantiles)
    return Models(
        projection=ProjectionModel(cfg.lgbm, cfg.ridge_alpha, cfg.recency_half_life_seasons, cfg.seed),
        fp_q=QuantileModel("y_fp_pg", FP_FEATURES, cfg.lgbm, cfg.recency_half_life_seasons, cfg.seed, q,
                           min_gp=FP_MIN_GP, clip=(0.0, 90.0), anchor="fp_pg_w"),
        share_q=QuantileModel("y_share", COMPONENT_FEATURES, cfg.lgbm, cfg.recency_half_life_seasons, cfg.seed, q,
                              clip=(0.0, 1.0)),
    )


def _fit(models: Models, train: pl.DataFrame) -> Models:
    models.projection.fit(train)
    models.fp_q.fit(train)
    models.share_q.fit(train)
    return models


def run_fold(ts: pl.DataFrame, season: int, cfg: TrainConfig) -> FoldResult:
    train = ts.filter((pl.col("season") < season) & (pl.col("season") >= cfg.first_target_season))
    test = ts.filter(pl.col("season") == season)
    m = _fit(_new_models(cfg), train)
    rb = baselines.RidgeBaseline(cfg.ridge_alpha).fit(train)
    return FoldResult(
        season=season,
        rows=test,
        parts=m.projection.predict_parts(test),
        fp_q_raw=m.fp_q.predict_raw(test),
        share_q_raw=m.share_q.predict_raw(test),
        ridge_base=rb.predict(test),
    )


def _concat_parts(folds: list[FoldResult]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    names = folds[0].parts.keys()
    return {n: (np.concatenate([f.parts[n][0] for f in folds]), np.concatenate([f.parts[n][1] for f in folds])) for n in names}


def budgeted(rows: pl.DataFrame, p: dict[str, np.ndarray], strength: float,
             settings: ScoringSettings) -> tuple[np.ndarray, np.ndarray]:
    """(minutes, FP/G) after the team minutes budget, for one season's rows.

    The rows are veterans; each team's rookies share the budget through the
    ``team_new_emin``/``team_new_wt`` sums the feature builder attached.
    """
    if strength <= 0.0 or "team_min_target" not in rows.columns:
        return p["mpg"], p["fp_pg_hat"]
    teams = rows.group_by("team_now").agg(pl.col("team_new_emin").first(), pl.col("team_new_wt").first())
    extra_e = dict(zip(teams["team_now"].to_list(), teams["team_new_emin"].to_list()))
    extra_w = dict(zip(teams["team_now"].to_list(), teams["team_new_wt"].to_list()))
    new = allocate(np.array(rows["team_now"].to_list(), dtype=object), p["mpg"], p["share"],
                   float(rows["team_min_target"][0]), strength, extra_e, extra_w)
    ratio = np.where(p["mpg"] > 1e-6, new / np.maximum(p["mpg"], 1e-6), 1.0)
    fp = linear_fp({c: p[f"rate_{c}"] for c in COMPONENTS}, new, settings) + p["bonus"] * ratio
    return new, fp


def point_predictions(fold: FoldResult, blends: dict[str, float], settings: ScoringSettings,
                      strength: float = 0.0) -> dict[str, dict[str, np.ndarray]]:
    """FP/G, games-share and minutes predictions for every model being compared.
    ``components_blend`` is what the system projects (with the team minutes budget);
    ``components_no_budget`` is the same models without it."""
    d = fold.rows
    share_1 = d["share_1"].fill_null(0).to_numpy()
    share_w = d["share_w"].fill_null(0).to_numpy()
    mpg_1 = d["mpg_1"].fill_null(0).to_numpy()
    mpg_w = d["mpg_w"].fill_null(0).to_numpy()
    out = {
        "last_season": {"fp": baselines.last_season(d), "share": share_1, "mpg": mpg_1},
        "marcel": {"fp": baselines.marcel(d), "share": share_w, "mpg": mpg_w},
        "ridge_baseline": {"fp": fold.ridge_base, "share": share_1, "mpg": mpg_1},
    }
    for label, b in (("lgbm", 1.0), ("ridge", 0.0)):
        p = blended_from_parts(fold.parts, {k: b for k in blends}, settings)
        out[f"components_{label}"] = {"fp": p["fp_pg_hat"], "share": p["share"], "mpg": p["mpg"]}
    p = blended_from_parts(fold.parts, blends, settings)
    out["components_no_budget"] = {"fp": p["fp_pg_hat"], "share": p["share"], "mpg": p["mpg"]}
    mpg_b, fp_b = budgeted(d, p, strength, settings)
    out["components_blend"] = {"fp": fp_b, "share": p["share"], "mpg": mpg_b}
    out["direct_fp_blend"] = {"fp": p["fp_direct"], "share": p["share"], "mpg": p["mpg"]}
    return out


def choose_budget(folds: list[FoldResult], blends: dict[str, float],
                  settings: ScoringSettings) -> tuple[float, pl.DataFrame]:
    """Strength of the team minutes budget: the grid value with the lowest FP/G MAE on the
    top 200 draftable players, averaged over the validation folds (ties: the weaker)."""
    rows = []
    for f in folds:
        d = f.rows
        p = blended_from_parts(f.parts, blends, settings)
        m = (d["y_gp"].to_numpy() >= FP_MIN_GP) & draftable_mask(d)
        y_fp, y_mpg = d["y_fp_pg"].fill_null(0).to_numpy(), d["y_mpg"].fill_null(0).to_numpy()
        for lam in STRENGTH_GRID:
            mpg, fp = budgeted(d, p, lam, settings)
            rows.append({"season": f.season, "strength": lam, "fpg_mae_top200": mae(fp[m], y_fp[m]),
                         "mpg_mae_top200": mae(mpg[m], y_mpg[m])})
    grid = pl.DataFrame(rows)
    mean = grid.group_by("strength").agg(pl.col("fpg_mae_top200").mean()).sort("strength")
    best = float(mean.sort(["fpg_mae_top200", "strength"])["strength"][0])
    return best, grid


OPP_COHORT = 15.0  # minutes per team game: about one rotation player's minutes


def cohorts(d: pl.DataFrame) -> dict[str, np.ndarray]:
    """Role-change cohorts (design section 24), reported separately because an average
    over everyone can hide a model that fails exactly where roles change. "Minutes fell"
    is defined by the outcome: a diagnostic, not something the model could know."""
    def col(c: str) -> np.ndarray:
        return d[c].fill_null(0).to_numpy()

    long_absent = (col("longest_absence") >= 15) | ((col("gp_1") == 0) & (col("gp_2") > 0))
    return {
        "all": np.ones(d.height, dtype=bool),
        "changed teams": col("new_team") == 1,
        "minutes fell 5+ (outcome)": (col("gp_1") >= 20) & (col("y_mpg") < col("mpg_1") - 5),
        "returning from a long absence": long_absent,
        f"team filled {OPP_COHORT:.0f}+ min/game at his positions": col("opp_pos_mpg") <= -OPP_COHORT,
        f"team opened {OPP_COHORT:.0f}+ min/game at his positions": col("opp_pos_mpg") >= OPP_COHORT,
    }


COHORT_MODELS = ("last_season", "marcel", "components_no_budget", "components_blend")


def cohort_metrics(fold: FoldResult, preds: dict[str, dict[str, np.ndarray]],
                   fp_interval: np.ndarray | None = None) -> list[dict]:
    """FP/G and minutes error (and bias: projected minus actual) per cohort, players with
    10+ games; 80% interval coverage for the system's projection."""
    d = fold.rows
    played = d["y_gp"].to_numpy() >= FP_MIN_GP
    y_fp, y_mpg = d["y_fp_pg"].fill_null(0).to_numpy(), d["y_mpg"].fill_null(0).to_numpy()
    rows = []
    for name, mask in cohorts(d).items():
        m = mask & played
        if m.sum() == 0:
            continue
        for model in COHORT_MODELS:
            p = preds[model]
            cov = None
            if model == "components_blend" and fp_interval is not None:
                cov = coverage(fp_interval[m], y_fp[m])
            rows.append({"season": fold.season, "cohort": name, "model": model, "n": int(m.sum()),
                         "fpg_mae": mae(p["fp"][m], y_fp[m]), "fpg_bias": float(np.mean(p["fp"][m] - y_fp[m])),
                         "mpg_mae": mae(p["mpg"][m], y_mpg[m]), "mpg_bias": float(np.mean(p["mpg"][m] - y_mpg[m])),
                         "fpg_cov80": cov})
    return rows


def fold_metrics(fold: FoldResult, preds: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    d = fold.rows
    team_gp = d["y_team_gp"].to_numpy()
    y_fp = d["y_fp_pg"].fill_null(0).to_numpy()
    y_tot = d["y_fp"].fill_null(0).to_numpy()
    y_share = d["y_share"].to_numpy()
    played = d["y_gp"].to_numpy() >= FP_MIN_GP
    top = draftable_mask(d)
    rows = []
    ref = preds["marcel"]["fp"]
    for name, p in preds.items():
        tot = p["fp"] * p["share"] * team_gp
        m_all, m_top = played, played & top
        diff, lo, hi = paired_bootstrap(ref[m_top] - y_fp[m_top], p["fp"][m_top] - y_fp[m_top], seed=fold.season)
        rows.append(
            {
                "season": fold.season,
                "model": name,
                "fpg_mae_top200": mae(p["fp"][m_top], y_fp[m_top]),
                "fpg_rmse_top200": rmse(p["fp"][m_top], y_fp[m_top]),
                "fpg_mae_all": mae(p["fp"][m_all], y_fp[m_all]),
                "fpg_spearman_top200": spearman(p["fp"][m_top], y_fp[m_top]),
                "season_mae_top200": mae(tot[top], y_tot[top]),
                "season_spearman_top200": spearman(tot[top], y_tot[top]),
                "share_mae_top200": mae(p["share"][top], y_share[top]),
                "gain_vs_marcel": diff,
                "gain_lo": lo,
                "gain_hi": hi,
                "n_top200_fpg": int(m_top.sum()),
            }
        )
    return rows


def uncertainty_metrics(label: str, season: int, fp_q: np.ndarray, sh_q: np.ndarray, d: pl.DataFrame,
                        quantiles: tuple[float, ...]) -> dict:
    played = d["y_gp"].to_numpy() >= FP_MIN_GP
    y_fp = d["y_fp_pg"].fill_null(0).to_numpy()
    y_sh = d["y_share"].to_numpy()
    top = draftable_mask(d)
    return {
        "split": label,
        "season": season,
        "fpg_cov80_all": coverage(fp_q[played], y_fp[played]),
        "fpg_cov80_top200": coverage(fp_q[played & top], y_fp[played & top]),
        "fpg_pinball": pinball(fp_q[played], y_fp[played], quantiles),
        "fpg_crps": crps_from_quantiles(fp_q[played], y_fp[played], quantiles),
        "share_cov80_all": coverage(sh_q, y_sh),
        "share_cov80_top200": coverage(sh_q[top], y_sh[top]),
        "share_pinball": pinball(sh_q, y_sh, quantiles),
    }


def _centred(proto: Models, fold: FoldResult, blends: dict[str, float], settings: ScoringSettings,
             strength: float = 0.0) -> np.ndarray:
    """FP/G intervals centred on the fold's point projection (after the minutes budget)."""
    point = budgeted(fold.rows, blended_from_parts(fold.parts, blends, settings), strength, settings)[1]
    return proto.fp_q.centre(fold.fp_q_raw, point)


def _adjusted(raw: np.ndarray, adj: tuple[float, float], clip: tuple[float, float]) -> np.ndarray:
    p = raw.copy()
    p[:, 0] -= adj[0]
    p[:, -1] += adj[1]
    return np.clip(np.sort(p, axis=1), *clip)


def train_all(ts: pl.DataFrame, settings: ScoringSettings, cfg: TrainConfig, log=print) -> TrainResult:
    q = tuple(cfg.quantiles)
    # 1. Rolling origin.
    folds = []
    for V in cfg.validation_seasons:
        folds.append(run_fold(ts, V, cfg))
        log(f"validation fold {V}: trained on {cfg.first_target_season}-{V - 1}")
    oof = pl.concat([f.rows for f in folds], how="diagonal_relaxed")
    blends = choose_blends(oof, _concat_parts(folds))
    log(f"blend weights on LightGBM: {blends}")
    strength, budget_grid = choose_budget(folds, blends, settings)
    log(f"team minutes budget strength: {strength}")

    # 2. Conformal corrections: cross-conformal for reporting, pooled for the frozen model.
    proto = _new_models(cfg)
    fp_c = [_centred(proto, f, blends, settings, strength) for f in folds]
    played = [f.rows["y_gp"].to_numpy() >= FP_MIN_GP for f in folds]
    unc_rows, val_rows, cohort_rows = [], [], []
    for i, f in enumerate(folds):
        others = [j for j in range(len(folds)) if j != i]
        fp_raw = np.vstack([fp_c[j][played[j]] for j in others])
        fp_y = np.concatenate([folds[j].rows["y_fp_pg"].to_numpy()[played[j]] for j in others])
        sh_raw = np.vstack([folds[j].share_q_raw for j in others])
        sh_y = np.concatenate([folds[j].rows["y_share"].to_numpy() for j in others])
        fa = proto.fp_q.calibrate(fp_raw, fp_y)
        sa = proto.share_q.calibrate(sh_raw, sh_y)
        fp_iv = _adjusted(fp_c[i], fa, (0, 90))
        unc_rows.append(uncertainty_metrics("validation", f.season, fp_iv,
                                            _adjusted(f.share_q_raw, sa, (0, 1)), f.rows, q))
        unc_rows.append(uncertainty_metrics("validation_uncalibrated", f.season, np.clip(fp_c[i], 0, 90),
                                            f.share_q_raw, f.rows, q))
        preds = point_predictions(f, blends, settings, strength)
        val_rows += fold_metrics(f, preds)
        cohort_rows += [{"split": "validation", **r} for r in cohort_metrics(f, preds, fp_iv)]
    fp_adj = proto.fp_q.calibrate(
        np.vstack([c[p] for c, p in zip(fp_c, played)]),
        np.concatenate([f.rows["y_fp_pg"].to_numpy()[p] for f, p in zip(folds, played)]),
    )
    share_adj = proto.share_q.calibrate(
        np.vstack([f.share_q_raw for f in folds]), np.concatenate([f.rows["y_share"].to_numpy() for f in folds])
    )

    # 3. Test season, touched once with everything frozen.
    test_fold = run_fold(ts, cfg.test_season, cfg)
    test_preds = point_predictions(test_fold, blends, settings, strength)
    test_rows = fold_metrics(test_fold, test_preds)
    test_c = _centred(proto, test_fold, blends, settings, strength)
    test_iv = _adjusted(test_c, fp_adj, (0, 90))
    unc_rows.append(uncertainty_metrics("test", cfg.test_season, test_iv,
                                        _adjusted(test_fold.share_q_raw, share_adj, (0, 1)), test_fold.rows, q))
    cohort_rows += [{"split": "test", **r} for r in cohort_metrics(test_fold, test_preds, test_iv)]
    log(f"test season {cfg.test_season} evaluated")

    models_test = _fit(_new_models(cfg), ts.filter(
        (pl.col("season") < cfg.test_season) & (pl.col("season") >= cfg.first_target_season)))
    # 4. Final models on every season through the test season.
    models_final = _fit(_new_models(cfg), ts.filter(pl.col("season") >= cfg.first_target_season))
    for m in (models_test, models_final):
        m.minutes_budget = strength
        m.projection.set_blends(blends)
        m.fp_q.adj_lo, m.fp_q.adj_hi = fp_adj
        m.share_q.adj_lo, m.share_q.adj_hi = share_adj
    log("final models fitted")

    validation = pl.DataFrame(val_rows)
    test = pl.DataFrame(test_rows)
    result = TrainResult(
        blends=blends, fp_adj=fp_adj, share_adj=share_adj, validation=validation, test=test,
        uncertainty=pl.DataFrame(unc_rows), test_rows=test_fold.rows, models_test=models_test,
        models_final=models_final, budget_strength=strength, budget_grid=budget_grid,
        cohorts=pl.DataFrame(cohort_rows, infer_schema_length=None),
    )
    result.summary = gate_summary(result)
    return result


def gate_summary(r: TrainResult) -> dict:
    """V1 acceptance test 1: beat Marcel on FP/G MAE in at least 3 of 4 validation seasons,
    with a paired bootstrap interval that excludes zero."""
    v = r.validation.filter(pl.col("model") == "components_blend").sort("season")
    wins = v.filter(pl.col("gain_lo") > 0).height
    better = v.filter(pl.col("gain_vs_marcel") > 0).height
    t = r.test.filter(pl.col("model") == "components_blend").row(0, named=True)
    u = r.uncertainty.filter(pl.col("split") == "test").row(0, named=True)
    return {
        "validation_seasons_better_than_marcel": better,
        "validation_seasons_better_with_ci_excluding_zero": wins,
        "gate_v1_projection_passed": wins >= 3,
        "test_gain_vs_marcel": [t["gain_vs_marcel"], t["gain_lo"], t["gain_hi"]],
        "test_fpg_mae_top200": t["fpg_mae_top200"],
        "test_fpg_cov80_top200": u["fpg_cov80_top200"],
        "test_share_cov80_top200": u["share_cov80_top200"],
        "blends": r.blends,
        "minutes_budget_strength": r.budget_strength,
        "test_gain_from_budget": _budget_gain(r),
        "fp_conformal_adj": list(r.fp_adj),
        "share_conformal_adj": list(r.share_adj),
    }


def _budget_gain(r: TrainResult) -> float:
    """Test-season FP/G MAE (top 200) without the budget minus with it (positive = helps)."""
    t = {row["model"]: row["fpg_mae_top200"] for row in r.test.iter_rows(named=True)}
    return float(t.get("components_no_budget", np.nan) - t.get("components_blend", np.nan))


def write_report(r: TrainResult, out_dir: Path) -> Path:
    """Per-season tables via DuckDB over the metric frames, plus a JSON summary."""
    import duckdb

    out_dir.mkdir(parents=True, exist_ok=True)
    r.validation.write_parquet(out_dir / "validation_metrics.parquet")
    r.test.write_parquet(out_dir / "test_metrics.parquet")
    unc = r.uncertainty.with_columns(
        pl.col("fpg_pinball").list.eval(pl.element().round(3)).cast(pl.List(pl.Utf8)).list.join(" / "),
        pl.col("share_pinball").list.eval(pl.element().round(4)).cast(pl.List(pl.Utf8)).list.join(" / "),
    )
    con = duckdb.connect()
    con.register("val", r.validation.to_arrow())
    con.register("test", r.test.to_arrow())
    summary_sql = """
        SELECT model,
               round(avg(fpg_mae_top200), 3) AS fpg_mae_top200,
               round(avg(fpg_mae_all), 3) AS fpg_mae_all,
               round(avg(fpg_spearman_top200), 3) AS fpg_spearman,
               round(avg(season_mae_top200), 1) AS season_mae_top200,
               round(avg(season_spearman_top200), 3) AS season_spearman,
               round(avg(share_mae_top200), 4) AS share_mae_top200,
               sum(CASE WHEN gain_lo > 0 THEN 1 ELSE 0 END) AS seasons_beat_marcel_ci
        FROM {t} GROUP BY model ORDER BY fpg_mae_top200
    """
    val_summary = pl.from_arrow(con.execute(summary_sql.format(t="val")).arrow())
    test_summary = pl.from_arrow(con.execute(summary_sql.format(t="test")).arrow())
    per_season = pl.from_arrow(con.execute(
        "SELECT season, model, round(fpg_mae_top200, 3) AS mae, round(gain_vs_marcel, 3) AS gain, "
        "round(gain_lo, 3) AS lo, round(gain_hi, 3) AS hi FROM val "
        "WHERE model IN ('components_blend', 'components_no_budget', 'components_lgbm', 'components_ridge', "
        "'direct_fp_blend') "
        "ORDER BY season, model").arrow())

    budget_table = (
        r.budget_grid.group_by("strength").agg(pl.col("fpg_mae_top200").mean(), pl.col("mpg_mae_top200").mean())
        .sort("strength")
        .join(r.budget_grid.sort("season").group_by("strength", maintain_order=True)
              .agg(pl.col("fpg_mae_top200").round(3).cast(pl.Utf8).str.join(" / ").alias("fpg_by_season")),
              on="strength")
        if r.budget_grid is not None else pl.DataFrame()
    )

    def cohort_table(split: str) -> pl.DataFrame:
        c = r.cohorts.filter(pl.col("split") == split) if r.cohorts is not None else pl.DataFrame()
        if c.height == 0:
            return pl.DataFrame()
        order = {m: i for i, m in enumerate(COHORT_MODELS)}
        names = list(dict.fromkeys(c["cohort"].to_list()))
        return (c.group_by("cohort", "model").agg(pl.col("n").mean().round(0).cast(pl.Int64),
                                                  *[pl.col(x).mean() for x in ("fpg_mae", "fpg_bias", "mpg_mae",
                                                                               "mpg_bias", "fpg_cov80")])
                .with_columns(pl.col("cohort").replace_strict({n: i for i, n in enumerate(names)}).alias("_c"),
                              pl.col("model").replace_strict(order).alias("_m"))
                .sort("_c", "_m").drop("_c", "_m"))

    cohort_val, cohort_test = cohort_table("validation"), cohort_table("test")
    if r.cohorts is not None:
        r.cohorts.write_parquet(out_dir / "cohort_metrics.parquet")

    def md(df: pl.DataFrame) -> str:
        cols = df.columns
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for row in df.iter_rows():
            lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else "" if v is None else str(v)
                                           for v in row) + " |")
        return "\n".join(lines)

    text = [
        "# Projection report",
        "",
        "FP/G metrics use players with 10+ games; `top200` is the top 200 per season by a",
        "pre-season value proxy (weighted FP/G x games share), known at the cut-off.",
        "`gain` is Marcel's absolute error minus the model's (positive = model better), averaged",
        "over players, with a paired 95% bootstrap interval.",
        "",
        "## Validation (rolling origin), averaged over seasons",
        md(val_summary),
        "",
        "## Validation per season",
        md(per_season),
        "",
        "## Test season (evaluated once)",
        md(test_summary),
        "",
        "## Team minutes budget",
        "Strength chosen on the validation seasons (0 = off, 1 = every team's expected minutes",
        f"pulled all the way to the league's minutes per team game). Chosen: **{r.budget_strength}**.",
        "FP/G and minutes MAE, top 200, by strength:",
        md(budget_table),
        "",
        "## Role-change cohorts",
        "Players with 10+ games. `bias` is projected minus actual (positive = too high);",
        "`fpg_cov80` is the 80% FP/G interval's coverage. \"Minutes fell 5+\" is defined by the",
        "outcome (a diagnostic). Opportunity is `opp_pos_mpg`: minutes per team game his team",
        "opened at his positions since last season (departures minus arrivals, rookies and",
        "teammates returning from long absences), relative to the league's average team.",
        "",
        "### Validation (mean over seasons)",
        md(cohort_val),
        "",
        "### Test season",
        md(cohort_test),
        "",
        "## Uncertainty (80% intervals: P10-P90)",
        md(unc.select("split", "season", "fpg_cov80_all", "fpg_cov80_top200", "fpg_crps", "share_cov80_all",
                      "share_cov80_top200", "fpg_pinball", "share_pinball")),
        "",
        "## Gate summary",
        "```json",
        json.dumps(r.summary, indent=2),
        "```",
    ]
    path = out_dir / "projection_report.md"
    path.write_text("\n".join(text))
    (out_dir / "projection_summary.json").write_text(json.dumps(r.summary, indent=2))
    return path
