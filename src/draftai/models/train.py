"""Training pipeline: rolling-origin validation, one test-season evaluation, final fit.

1. For each validation season V: fit on target seasons before V, predict V.
2. Choose blend weights and conformal corrections from those out-of-fold predictions.
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
from draftai.features.build import COMPONENT_FEATURES, FP_FEATURES
from draftai.models import baselines
from draftai.models.components import ProjectionModel, blended_from_parts
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


def point_predictions(fold: FoldResult, blends: dict[str, float], settings: ScoringSettings) -> dict[str, dict[str, np.ndarray]]:
    """FP/G and games-share predictions for every model being compared."""
    d = fold.rows
    share_1 = d["share_1"].fill_null(0).to_numpy()
    share_w = d["share_w"].fill_null(0).to_numpy()
    out = {
        "last_season": {"fp": baselines.last_season(d), "share": share_1},
        "marcel": {"fp": baselines.marcel(d), "share": share_w},
        "ridge_baseline": {"fp": fold.ridge_base, "share": share_1},
    }
    for label, b in (("lgbm", 1.0), ("ridge", 0.0)):
        p = blended_from_parts(fold.parts, {k: b for k in blends}, settings)
        out[f"components_{label}"] = {"fp": p["fp_pg_hat"], "share": p["share"]}
    p = blended_from_parts(fold.parts, blends, settings)
    out["components_blend"] = {"fp": p["fp_pg_hat"], "share": p["share"]}
    out["direct_fp_blend"] = {"fp": p["fp_direct"], "share": p["share"]}
    return out


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


def _centred(proto: Models, fold: FoldResult, blends: dict[str, float], settings: ScoringSettings) -> np.ndarray:
    """FP/G intervals centred on the fold's blended point projection."""
    point = blended_from_parts(fold.parts, blends, settings)["fp_pg_hat"]
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

    # 2. Conformal corrections: cross-conformal for reporting, pooled for the frozen model.
    proto = _new_models(cfg)
    fp_c = [_centred(proto, f, blends, settings) for f in folds]
    played = [f.rows["y_gp"].to_numpy() >= FP_MIN_GP for f in folds]
    unc_rows, val_rows = [], []
    for i, f in enumerate(folds):
        others = [j for j in range(len(folds)) if j != i]
        fp_raw = np.vstack([fp_c[j][played[j]] for j in others])
        fp_y = np.concatenate([folds[j].rows["y_fp_pg"].to_numpy()[played[j]] for j in others])
        sh_raw = np.vstack([folds[j].share_q_raw for j in others])
        sh_y = np.concatenate([folds[j].rows["y_share"].to_numpy() for j in others])
        fa = proto.fp_q.calibrate(fp_raw, fp_y)
        sa = proto.share_q.calibrate(sh_raw, sh_y)
        unc_rows.append(uncertainty_metrics("validation", f.season, _adjusted(fp_c[i], fa, (0, 90)),
                                            _adjusted(f.share_q_raw, sa, (0, 1)), f.rows, q))
        unc_rows.append(uncertainty_metrics("validation_uncalibrated", f.season, np.clip(fp_c[i], 0, 90),
                                            f.share_q_raw, f.rows, q))
        val_rows += fold_metrics(f, point_predictions(f, blends, settings))
    fp_adj = proto.fp_q.calibrate(
        np.vstack([c[p] for c, p in zip(fp_c, played)]),
        np.concatenate([f.rows["y_fp_pg"].to_numpy()[p] for f, p in zip(folds, played)]),
    )
    share_adj = proto.share_q.calibrate(
        np.vstack([f.share_q_raw for f in folds]), np.concatenate([f.rows["y_share"].to_numpy() for f in folds])
    )

    # 3. Test season, touched once with everything frozen.
    test_fold = run_fold(ts, cfg.test_season, cfg)
    test_rows = fold_metrics(test_fold, point_predictions(test_fold, blends, settings))
    test_c = _centred(proto, test_fold, blends, settings)
    unc_rows.append(uncertainty_metrics("test", cfg.test_season, _adjusted(test_c, fp_adj, (0, 90)),
                                        _adjusted(test_fold.share_q_raw, share_adj, (0, 1)), test_fold.rows, q))
    log(f"test season {cfg.test_season} evaluated")

    models_test = _fit(_new_models(cfg), ts.filter(
        (pl.col("season") < cfg.test_season) & (pl.col("season") >= cfg.first_target_season)))
    # 4. Final models on every season through the test season.
    models_final = _fit(_new_models(cfg), ts.filter(pl.col("season") >= cfg.first_target_season))
    for m in (models_test, models_final):
        m.projection.set_blends(blends)
        m.fp_q.adj_lo, m.fp_q.adj_hi = fp_adj
        m.share_q.adj_lo, m.share_q.adj_hi = share_adj
    log("final models fitted")

    validation = pl.DataFrame(val_rows)
    test = pl.DataFrame(test_rows)
    result = TrainResult(
        blends=blends, fp_adj=fp_adj, share_adj=share_adj, validation=validation, test=test,
        uncertainty=pl.DataFrame(unc_rows), test_rows=test_fold.rows, models_test=models_test,
        models_final=models_final,
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
        "fp_conformal_adj": list(r.fp_adj),
        "share_conformal_adj": list(r.share_adj),
    }


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
        "WHERE model IN ('components_blend', 'components_lgbm', 'components_ridge', 'direct_fp_blend') "
        "ORDER BY season, model").arrow())

    def md(df: pl.DataFrame) -> str:
        cols = df.columns
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for row in df.iter_rows():
            lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
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
