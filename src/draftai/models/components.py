"""Component projection models: minutes, per-minute rates, games share, bonuses.

Each target is a blend of LightGBM (monotone-constrained, shallow) and ridge. Both are
always fitted; the blend weight is chosen on rolling-origin validation seasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from draftai.features.build import COMPONENT_FEATURES, COMPONENTS, FP_FEATURES
from draftai.scoring import ScoringSettings
from draftai.scoring.settings import STAT_KEYS


@dataclass(frozen=True)
class TargetSpec:
    name: str
    target: str
    weight: str | None
    min_weight: float
    features: list[str]
    monotone: dict[str, int]
    clip: tuple[float, float]


def target_specs() -> list[TargetSpec]:
    specs = [
        TargetSpec("mpg", "y_mpg", "y_gp", 1, COMPONENT_FEATURES,
                   {"mpg_1": 1, "mpg_w": 1, "min_last20": 1}, (0.0, 44.0)),
        TargetSpec("share", "y_share", None, 0, COMPONENT_FEATURES, {"share_w": 1, "share_1": 1}, (0.0, 1.0)),
    ]
    for c in COMPONENTS:
        specs.append(
            TargetSpec(f"rate_{c}", f"y_rate_{c}", "y_min", 50, COMPONENT_FEATURES, {f"rate_{c}": 1}, (0.0, 2.0))
        )
    specs.append(TargetSpec("bonus", "y_bonus_pg", "y_gp", 1, FP_FEATURES, {}, (0.0, 20.0)))
    specs.append(TargetSpec("fp_direct", "y_fp_pg", "y_gp", 1, FP_FEATURES, {"fp_pg_w": 1}, (0.0, 90.0)))
    return specs


def recency_weights(seasons: np.ndarray, half_life: float) -> np.ndarray:
    latest = seasons.max()
    return 0.5 ** ((latest - seasons) / half_life)


def training_rows(df: pl.DataFrame, spec: TargetSpec) -> pl.DataFrame:
    d = df.filter(pl.col(spec.target).is_not_null() & pl.col(spec.target).is_finite())
    if spec.weight is not None:
        d = d.filter(pl.col(spec.weight) >= spec.min_weight)
    return d


@dataclass
class BlendModel:
    spec: TargetSpec
    lgbm_params: dict
    ridge_alpha: float
    half_life: float
    seed: int
    blend: float = 0.5  # weight on LightGBM
    lgbm: lgb.LGBMRegressor | None = None
    ridge: object | None = None

    def _x(self, df: pl.DataFrame) -> np.ndarray:
        return df.select(self.spec.features).to_numpy().astype(np.float64)

    def fit(self, df: pl.DataFrame) -> "BlendModel":
        d = training_rows(df, self.spec)
        x, y = self._x(d), d[self.spec.target].to_numpy().astype(float)
        w = recency_weights(d["season"].to_numpy(), self.half_life)
        if self.spec.weight is not None:
            w = w * d[self.spec.weight].to_numpy().astype(float)
        mono = [self.spec.monotone.get(f, 0) for f in self.spec.features]
        self.lgbm = lgb.LGBMRegressor(
            objective="regression_l2", monotone_constraints=mono, random_state=self.seed,
            deterministic=True, force_row_wise=True, **self.lgbm_params,
        )
        self.lgbm.fit(x, y, sample_weight=w)
        self.ridge = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=self.ridge_alpha)
        )
        self.ridge.fit(x, y, ridge__sample_weight=w)
        return self

    def predict_parts(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = self._x(df)
        return self.lgbm.predict(x), self.ridge.predict(x)

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        g, r = self.predict_parts(df)
        return np.clip(self.blend * g + (1 - self.blend) * r, *self.spec.clip)


def linear_fp(rates: dict[str, np.ndarray], mpg: np.ndarray, settings: ScoringSettings) -> np.ndarray:
    """Expected linear fantasy points per game from per-minute component rates."""
    total = np.zeros_like(mpg, dtype=float)
    for key, w in settings.weights.items():
        col = STAT_KEYS.get(key)
        if col in rates:
            total += w * rates[col] * mpg
    return total


@dataclass
class ProjectionModel:
    lgbm_params: dict
    ridge_alpha: float
    half_life: float
    seed: int
    blends: dict[str, float] = field(default_factory=dict)
    models: dict[str, BlendModel] = field(default_factory=dict)

    def fit(self, df: pl.DataFrame) -> "ProjectionModel":
        for spec in target_specs():
            m = BlendModel(spec, self.lgbm_params, self.ridge_alpha, self.half_life, self.seed,
                           blend=self.blends.get(spec.name, 0.5))
            self.models[spec.name] = m.fit(df)
        return self

    def predict_parts(self, df: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {name: m.predict_parts(df) for name, m in self.models.items()}

    def set_blends(self, blends: dict[str, float]) -> None:
        self.blends = dict(blends)
        for name, m in self.models.items():
            m.blend = blends.get(name, m.blend)

    def predict(self, df: pl.DataFrame, settings: ScoringSettings) -> pl.DataFrame:
        p = {name: m.predict(df) for name, m in self.models.items()}
        rates = {c: p[f"rate_{c}"] for c in COMPONENTS}
        lin = linear_fp(rates, p["mpg"], settings)
        return df.select("player_id", "season").with_columns(
            pl.Series("mpg_hat", p["mpg"]),
            pl.Series("share_hat", p["share"]),
            *[pl.Series(f"rate_{c}_hat", rates[c]) for c in COMPONENTS],
            pl.Series("bonus_hat", p["bonus"]),
            pl.Series("fp_pg_hat", lin + p["bonus"]),
            pl.Series("fp_direct_hat", p["fp_direct"]),
        )


def blended_from_parts(
    parts: dict[str, tuple[np.ndarray, np.ndarray]], blends: dict[str, float], settings: ScoringSettings
) -> dict[str, np.ndarray]:
    """Point predictions for every target from saved LightGBM/ridge parts and blend weights."""
    specs = {s.name: s for s in target_specs()}
    out = {}
    for name, (g, r) in parts.items():
        b = blends.get(name, 0.5)
        out[name] = np.clip(b * g + (1 - b) * r, *specs[name].clip)
    rates = {c: out[f"rate_{c}"] for c in COMPONENTS}
    out["fp_pg_hat"] = linear_fp(rates, out["mpg"], settings) + out["bonus"]
    return out
