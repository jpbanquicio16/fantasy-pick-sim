"""Which per-minute rate prior? The height-bucket prior against the hierarchical one (V5).

Two tests, both on the validation seasons only (the test season is left for ``draftai
train`` once the choice is made):

1. Direct: how well each prior's shrunk rate predicts next season's per-minute rate, by
   stat, weighted by next season's minutes, overall and where shrinkage matters most
   (players with few minutes, second-year players).
2. Through the whole model: rolling-origin validation FP/G MAE with each prior's features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from draftai.config import TrainConfig
from draftai.features.build import COMPONENTS, build_training_set
from draftai.features.rates import PRIORS
from draftai.scoring import ScoringSettings

MIN_NEXT = 200.0  # next-season minutes for a rate to be scored
LOW_MINUTES = 1500.0  # weighted minutes (5/3/2) below which shrinkage matters


def direct_errors(ts: pl.DataFrame, prior: str, seasons: list[int]) -> list[dict]:
    d = ts.filter(pl.col("season").is_in(seasons) & (pl.col("y_min") >= MIN_NEXT) & pl.col("min_w").is_not_null())
    groups = {
        "all": np.ones(d.height, dtype=bool),
        f"weighted minutes < {LOW_MINUTES:.0f}": d["min_w"].to_numpy() < LOW_MINUTES,
        "second season": d["experience"].fill_null(-1).to_numpy() == 1,
    }
    w = d["y_min"].to_numpy()
    out = []
    for g, m in groups.items():
        for c in COMPONENTS:
            p, y = d[f"rate_{c}"].to_numpy()[m], d[f"y_rate_{c}"].to_numpy()[m]
            ok = np.isfinite(p) & np.isfinite(y)
            # Error per 36 minutes, so stats are on a readable scale.
            err = 36.0 * (p[ok] - y[ok])
            out.append({"prior": prior, "group": g, "stat": c, "n": int(ok.sum()),
                        "mae_per36": float(np.average(np.abs(err), weights=w[m][ok])),
                        "bias_per36": float(np.average(err, weights=w[m][ok]))})
    return out


def validation_metrics(ts: pl.DataFrame, settings: ScoringSettings, cfg: TrainConfig) -> pl.DataFrame:
    """The rolling-origin part of ``train_all``: folds, blends, budget, metrics."""
    from draftai.models.train import (
        _concat_parts,
        choose_blends,
        choose_budget,
        cohort_metrics,
        fold_metrics,
        point_predictions,
        run_fold,
    )

    folds = [run_fold(ts, V, cfg) for V in cfg.validation_seasons]
    oof = pl.concat([f.rows for f in folds], how="diagonal_relaxed")
    blends = choose_blends(oof, _concat_parts(folds))
    strength, _ = choose_budget(folds, blends, settings)
    rows, cohorts = [], []
    for f in folds:
        preds = point_predictions(f, blends, settings, strength)
        rows += fold_metrics(f, preds)
        cohorts += cohort_metrics(f, preds)
    return pl.DataFrame(rows), pl.DataFrame(cohorts, infer_schema_length=None), strength


def run(pg, tg, bio, dr, settings: ScoringSettings, cfg: TrainConfig, out_dir: Path, log=print) -> dict:
    seasons = list(range(cfg.first_target_season, cfg.test_season + 1))
    direct, val, coh, strength = [], {}, {}, {}
    for prior in PRIORS:
        ts = build_training_set(pg, tg, bio, dr, settings, seasons, rate_prior=prior)
        direct += direct_errors(ts, prior, cfg.validation_seasons)
        v, c, s = validation_metrics(ts, settings, cfg)
        val[prior], coh[prior], strength[prior] = v, c, s
        log(f"rate prior {prior}: validation done")
    direct_df = pl.DataFrame(direct)
    summary = {}
    for prior in PRIORS:
        v = val[prior].filter(pl.col("model") == "components_blend").sort("season")
        summary[prior] = {"fpg_mae_top200": float(v["fpg_mae_top200"].mean()),
                          "by_season": [round(x, 4) for x in v["fpg_mae_top200"].to_list()],
                          "fpg_mae_all": float(v["fpg_mae_all"].mean()),
                          "season_spearman": float(v["season_spearman_top200"].mean()),
                          "budget_strength": strength[prior]}
    best = min(PRIORS, key=lambda p: summary[p]["fpg_mae_top200"])
    summary["chosen"] = best
    write(direct_df, val, coh, summary, out_dir)
    return summary


def write(direct: pl.DataFrame, val: dict, coh: dict, summary: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    def md(df: pl.DataFrame) -> str:
        lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        for row in df.iter_rows():
            lines.append("| " + " | ".join("" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
                                           for v in row) + " |")
        return "\n".join(lines)

    others = [p for p in PRIORS if p != "bucket"]
    wide = direct.pivot("prior", index=["group", "stat"], values="mae_per36").with_columns(
        *[(pl.col("bucket") - pl.col(p)).alias(f"gain_{p}") for p in others])
    n = direct.filter(pl.col("prior") == "bucket").select("group", "stat", "n")
    wide = wide.join(n, on=["group", "stat"], how="left")
    tot = direct.group_by("prior", "group").agg(pl.col("mae_per36").mean()).pivot(
        "prior", index="group", values="mae_per36").with_columns(
        *[(pl.col("bucket") - pl.col(p)).alias(f"gain_{p}") for p in others])
    per = pl.concat([v.filter(pl.col("model") == "components_blend").select(
        "season", pl.lit(p).alias("prior"), "fpg_mae_top200", "fpg_mae_all", "season_spearman_top200")
        for p, v in val.items()]).sort("season", "prior")
    cohort = pl.concat([c.filter(pl.col("model") == "components_blend").group_by("cohort").agg(
        pl.col("n").mean().round(0), pl.col("fpg_mae").mean(), pl.col("fpg_bias").mean()).with_columns(
        pl.lit(p).alias("prior")) for p, c in coh.items()]).sort("cohort", "prior")
    text = [
        "# Per-minute rate prior: height bucket vs hierarchical",
        "",
        "Validation seasons only. The bucket prior shrinks each player's 5/3/2-weighted per-minute rates toward",
        "his height bucket's mean; the hierarchical prior toward a regression on height bucket, age, experience",
        "and draft slot, fitted by marginal likelihood (features/rates.py). `hierarchical_scaled` also puts the",
        "prior's strength on the scale of the weighted evidence (the bucket prior and `hierarchical` count a",
        "minute of last season five times against a prior fitted on single seasons).",
        "",
        "## 1. Next season's per-minute rate (error per 36 minutes, weighted by next season's minutes)",
        f"Players with {MIN_NEXT:.0f}+ minutes next season. `gain_x` = bucket error minus x's (positive = x better).",
        "",
        "### Mean over the seven stats",
        md(tot),
        "",
        "### By stat",
        md(wide.sort("group", "stat")),
        "",
        "## 2. Through the whole model (rolling-origin validation, components blend with the minutes budget)",
        md(per),
        "",
        "### Cohorts (validation mean)",
        md(cohort),
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
    ]
    path = out_dir / "rate_prior_report.md"
    path.write_text("\n".join(text))
    (out_dir / "rate_prior_summary.json").write_text(json.dumps(summary, indent=2))
    return path
