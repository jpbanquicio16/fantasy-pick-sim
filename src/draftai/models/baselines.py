"""Baselines every report includes. The bar the learned models must clear."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MARCEL_K = 100.0  # regression toward the mean, in 5/3/2-weighted games (~20 last-season games)


def _pool_mean(df: pl.DataFrame) -> float:
    """Games-weighted mean of last-season FP/G across the pool: known at the cut-off."""
    d = df.filter(pl.col("gp_1") > 0)
    return float((d["fp_pg_1"] * d["gp_1"]).sum() / d["gp_1"].sum())


def last_season(df: pl.DataFrame) -> np.ndarray:
    mean = _pool_mean(df)
    return df.select(pl.coalesce("fp_pg_1", "fp_pg_w", pl.lit(mean))).to_series().to_numpy()


def marcel(df: pl.DataFrame) -> np.ndarray:
    """5/3/2 weights by games, regressed toward the pool mean, simple age adjustment."""
    mean = _pool_mean(df)
    base = (pl.col("fp_wsum") + MARCEL_K * mean) / (pl.col("gp_w") + MARCEL_K)
    age = pl.col("age").fill_null(27.0)
    adj = 1.0 + 0.01 * (27.0 - age).clip(-8, 6)
    return df.select((base * adj).alias("m")).to_series().to_numpy()


RIDGE_BASELINE_FEATURES = ["fp36_1", "mpg_1", "share_1", "age"]


class RidgeBaseline:
    """Ridge on last-season FP/36, minutes, games share and age."""

    def __init__(self, alpha: float = 10.0):
        self.model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=alpha))

    @staticmethod
    def _x(df: pl.DataFrame) -> np.ndarray:
        d = df.with_columns((36.0 * pl.col("fp_1") / pl.col("min_1").clip(1.0)).alias("fp36_1"))
        return d.select(RIDGE_BASELINE_FEATURES).to_numpy().astype(float)

    def fit(self, df: pl.DataFrame) -> "RidgeBaseline":
        d = df.filter(pl.col("y_gp") >= 1)
        self.model.fit(self._x(d), d["y_fp_pg"].to_numpy(), ridge__sample_weight=d["y_gp"].to_numpy())
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self.model.predict(self._x(df))
