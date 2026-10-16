"""Why is a projection what it is? Grouped TreeSHAP on the gradient-boosted models.

LightGBM computes exact TreeSHAP values itself (``pred_contrib=True``). Contributions
are summed within feature groups so correlated features (three seasons of minutes, say)
do not split one story into several misleading pieces. This explains the projection,
not the pick; the pick is explained by the recommender's own components.
"""

from __future__ import annotations

import numpy as np
import polars as pl

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "recent production": ("fp_pg_1", "fp_pg_2", "fp_pg_3", "fp_pg_w", "fp36_w", "bonus_pg_w", "dd_rate", "fp50_rate"),
    "game-to-game spread": ("fp_sd", "fp_cv_1", "fp_tail_1"),
    "per-minute rates": ("rate_pts", "rate_reb", "rate_ast", "rate_stl", "rate_blk", "rate_tov", "rate_fg3m",
                         "used_per_min_w"),
    "minutes and role": ("gp_1", "gp_2", "gp_3", "mpg_1", "mpg_2", "mpg_3", "min_w", "mpg_w", "mpg_trend",
                         "min_last20", "last20_minus_mpg"),
    "usage": ("usage_1", "usage_2"),
    "availability history": ("share_1", "share_2", "share_3", "share_w", "longest_absence", "q_short"),
    "age and experience": ("age", "age2", "experience", "log_draft_pick", "height_in"),
    "team change": ("new_team", "team_pace_1", "vacated_mpg"),
}


def grouped_contributions(model, features: list[str], df: pl.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(per-row group contributions, group names, per-row base value)."""
    x = df.select(features).to_numpy().astype(float)
    contrib = model.predict(x, pred_contrib=True)
    groups = [g for g, cols in FEATURE_GROUPS.items() if any(c in features for c in cols)]
    out = np.zeros((x.shape[0], len(groups)))
    for j, g in enumerate(groups):
        idx = [features.index(c) for c in FEATURE_GROUPS[g] if c in features]
        out[:, j] = contrib[:, idx].sum(axis=1)
    return out, groups, contrib[:, -1]


def projection_drivers(models, feats: pl.DataFrame) -> pl.DataFrame:
    """Per player: grouped contributions to the direct FP/G model and the games-share model."""
    cols = {"player_id": feats["player_id"]}
    for label, name in (("fp", "fp_direct"), ("share", "share")):
        m = models.projection.models[name]
        vals, groups, base = grouped_contributions(m.lgbm, m.spec.features, feats)
        cols[f"drv_{label}_base"] = base
        for j, g in enumerate(groups):
            cols[f"drv_{label}:{g}"] = vals[:, j]
    return pl.DataFrame(cols)


def top_drivers(row: dict, label: str = "fp", n: int = 3) -> list[tuple[str, float]]:
    """Largest absolute group contributions for one board row."""
    prefix = f"drv_{label}:"
    items = [(k[len(prefix):], v) for k, v in row.items() if k.startswith(prefix) and v is not None]
    return sorted(items, key=lambda kv: -abs(kv[1]))[:n]
