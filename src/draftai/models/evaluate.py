"""Metrics, bootstrap intervals and blend selection for rolling-origin evaluation."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from draftai.models.components import target_specs, training_rows

BLEND_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DRAFTABLE = 200


def draftable_mask(df: pl.DataFrame, n: int = DRAFTABLE) -> np.ndarray:
    """Top ``n`` per season by a pre-season value proxy (Marcel-style FP/G x games share),
    known at the cut-off. Metrics weighted toward players who are actually drafted."""
    proxy = (pl.col("fp_pg_w").fill_null(0) * pl.col("share_w").fill_null(0)).alias("proxy")
    rank = proxy.rank("ordinal", descending=True).over("season")
    return df.select((rank <= n).alias("m")).to_series().to_numpy()


def mae(pred: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(np.average(np.abs(pred - y), weights=w))


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def paired_bootstrap(err_a: np.ndarray, err_b: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of |err_a| - |err_b| over players, with a 95% bootstrap interval.

    Positive values mean model B has the lower error.
    """
    rng = np.random.default_rng(seed)
    d = np.abs(err_a) - np.abs(err_b)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def spearman(pred: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(pred, y).statistic)


def choose_blends(oof: pl.DataFrame, parts: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Blend weight on LightGBM per target, minimising weighted MAE on out-of-fold rows."""
    blends = {}
    for spec in target_specs():
        g, r = parts[spec.name]
        mask_df = oof.with_row_index("_i")
        rows = training_rows(mask_df, spec)
        i = rows["_i"].to_numpy()
        y = rows[spec.target].to_numpy().astype(float)
        w = rows[spec.weight].to_numpy().astype(float) if spec.weight else None
        best = min(BLEND_GRID, key=lambda b: mae(np.clip(b * g[i] + (1 - b) * r[i], *spec.clip), y, w))
        blends[spec.name] = best
    return blends
