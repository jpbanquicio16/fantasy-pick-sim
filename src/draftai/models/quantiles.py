"""Season-level and availability uncertainty: LightGBM quantile models with conformal
calibration (asymmetric conformalised quantile regression).

With an ``anchor`` the trees model quantiles of the residual ``target - anchor`` and the
interval is re-centred on the point projection: shallow trees cannot extrapolate to the
handful of elite players, but they can learn how wide each player's range is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import polars as pl

from draftai.models.components import recency_weights


@dataclass
class QuantileModel:
    target: str
    features: list[str]
    lgbm_params: dict
    half_life: float
    seed: int
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    min_gp: int = 0
    clip: tuple[float, float] = (0.0, np.inf)
    anchor: str | None = None
    adj_lo: float = 0.0
    adj_hi: float = 0.0
    models: dict[float, lgb.LGBMRegressor] = field(default_factory=dict)

    def rows(self, df: pl.DataFrame) -> pl.DataFrame:
        d = df.filter(pl.col(self.target).is_not_null())
        return d.filter(pl.col("y_gp") >= self.min_gp) if self.min_gp else d

    def fit(self, df: pl.DataFrame) -> "QuantileModel":
        d = self.rows(df)
        x = d.select(self.features).to_numpy().astype(float)
        y = d[self.target].to_numpy().astype(float) - self._anchor(d)
        w = recency_weights(d["season"].to_numpy(), self.half_life)
        params = {k: v for k, v in self.lgbm_params.items() if k not in ("objective",)}
        for q in self.quantiles:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=self.seed, deterministic=True,
                                  force_row_wise=True, **params)
            self.models[q] = m.fit(x, y, sample_weight=w)
        return self

    def _anchor(self, df: pl.DataFrame) -> np.ndarray:
        if self.anchor is None:
            return np.zeros(df.height)
        return df[self.anchor].fill_null(0.0).to_numpy().astype(float)

    def predict_raw(self, df: pl.DataFrame) -> np.ndarray:
        """Raw quantiles (of the residual, when anchored), sorted so they never cross."""
        x = df.select(self.features).to_numpy().astype(float)
        preds = np.column_stack([self.models[q].predict(x) for q in self.quantiles])
        return np.sort(preds, axis=1)

    def centre(self, raw: np.ndarray, point: np.ndarray | None) -> np.ndarray:
        """Anchored models: shift the residual quantiles so the median sits on ``point``."""
        if self.anchor is None:
            return raw.copy()
        mid = raw.shape[1] // 2
        return point[:, None] + (raw - raw[:, [mid]])

    def adjust(self, centred: np.ndarray) -> np.ndarray:
        p = centred.copy()
        p[:, 0] -= self.adj_lo
        p[:, -1] += self.adj_hi
        return np.clip(np.sort(p, axis=1), *self.clip)

    def predict(self, df: pl.DataFrame, point: np.ndarray | None = None) -> np.ndarray:
        return self.adjust(self.centre(self.predict_raw(df), point))

    def calibrate(self, raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Set the outer-quantile corrections from held-out (centred) predictions."""
        lo_q, hi_q = self.quantiles[0], self.quantiles[-1]
        n = len(y)
        lvl_lo = min(1.0, (1 - lo_q) * (1 + 1 / n))
        lvl_hi = min(1.0, hi_q * (1 + 1 / n))
        self.adj_lo = float(np.quantile(raw[:, 0] - y, lvl_lo))
        self.adj_hi = float(np.quantile(y - raw[:, -1], lvl_hi))
        return self.adj_lo, self.adj_hi


def coverage(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y >= pred[:, 0]) & (y <= pred[:, -1])))


def pinball(pred: np.ndarray, y: np.ndarray, quantiles: tuple[float, ...]) -> list[float]:
    out = []
    for j, q in enumerate(quantiles):
        e = y - pred[:, j]
        out.append(float(np.mean(np.maximum(q * e, (q - 1) * e))))
    return out


def crps_from_quantiles(pred: np.ndarray, y: np.ndarray, quantiles: tuple[float, ...]) -> float:
    """CRPS approximated by twice the average pinball loss over a quantile grid.

    With only three quantiles this is coarse; it is reported for comparison between
    models on the same grid, not as an absolute number.
    """
    return float(2 * np.mean(pinball(pred, y, quantiles)))
