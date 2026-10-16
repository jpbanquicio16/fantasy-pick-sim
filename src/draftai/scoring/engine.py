"""Deterministic Sleeper fantasy scoring: one game's box score -> fantasy points.

Points are computed as integers in the settings' smallest unit (half-points for the
default league) so sums never drift in floating point. Three entry points share one
implementation:

* ``score_arrays`` - NumPy, used by the simulator (millions of simulated games);
* ``score_frame`` - Polars, used for historical game logs;
* ``score_line`` - one game as a dict, used by tests and explanations.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import polars as pl

from draftai.scoring.settings import (
    BONUS_KEYS,
    BOX_COLUMNS,
    DD_CATEGORIES,
    MISS_KEYS,
    STAT_KEYS,
    ScoringSettings,
)

ArrayMap = Mapping[str, np.ndarray]


def _get(cols: ArrayMap, name: str, n: int) -> np.ndarray:
    v = cols.get(name)
    if v is None:
        return np.zeros(n, dtype=np.int64)
    return np.asarray(v).astype(np.int64, copy=False)


def _length(cols: ArrayMap) -> int:
    for v in cols.values():
        return int(np.size(v))
    return 0


def bonus_indicators(cols: ArrayMap, stacking: bool) -> dict[str, np.ndarray]:
    """0/1 arrays for each threshold bonus."""
    n = _length(cols)
    pts = _get(cols, "pts", n)
    cats = sum((_get(cols, c, n) >= 10).astype(np.int64) for c in DD_CATEGORIES)
    b40 = pts >= 40 if stacking else (pts >= 40) & (pts < 50)
    return {
        "dd": (cats >= 2).astype(np.int64),
        "td": (cats >= 3).astype(np.int64),
        "bonus_pt_40p": b40.astype(np.int64),
        "bonus_pt_50p": (pts >= 50).astype(np.int64),
        "bonus_ast_15p": (_get(cols, "ast", n) >= 15).astype(np.int64),
        "bonus_reb_20p": (_get(cols, "reb", n) >= 20).astype(np.int64),
    }


def score_units(cols: ArrayMap, settings: ScoringSettings) -> np.ndarray:
    """Fantasy points in integer units of 1/settings.units_per_point."""
    n = _length(cols)
    total = np.zeros(n, dtype=np.int64)
    weights = settings.unit_weights()
    bonuses = None
    for key, w in weights.items():
        if key in STAT_KEYS:
            total += w * _get(cols, STAT_KEYS[key], n)
        elif key in MISS_KEYS:
            att, made = MISS_KEYS[key]
            total += w * (_get(cols, att, n) - _get(cols, made, n))
        elif key in BONUS_KEYS:
            if bonuses is None:
                bonuses = bonus_indicators(cols, settings.bonus_stacking)
            total += w * bonuses[key]
        else:  # combination stat such as pts_reb_ast
            total += w * sum(_get(cols, part, n) for part in key.split("_"))
    return total


def bonus_units(cols: ArrayMap, settings: ScoringSettings) -> np.ndarray:
    """The threshold-bonus part of ``score_units`` (double-doubles, 40+ points, ...)."""
    n = _length(cols)
    ind = bonus_indicators(cols, settings.bonus_stacking)
    total = np.zeros(n, dtype=np.int64)
    for key, w in settings.unit_weights().items():
        if key in BONUS_KEYS:
            total += w * ind[key]
    return total


def score_arrays(cols: ArrayMap, settings: ScoringSettings) -> np.ndarray:
    """Fantasy points as float64 (exact multiples of the settings' resolution)."""
    return score_units(cols, settings) / settings.units_per_point


def score_line(line: Mapping[str, float], settings: ScoringSettings) -> float:
    cols = {k: np.array([v]) for k, v in line.items() if v is not None}
    return float(score_arrays(cols, settings)[0])


def score_frame(
    df: pl.DataFrame, settings: ScoringSettings, out: str = "fp"
) -> pl.DataFrame:
    """Add ``out`` (float points) and ``out_units`` (integer units) columns."""
    cols = {c: df[c].fill_null(0).to_numpy() for c in BOX_COLUMNS if c in df.columns}
    units = score_units(cols, settings)
    bonus = bonus_units(cols, settings)
    res = settings.units_per_point
    return df.with_columns(
        pl.Series(f"{out}_units", units, dtype=pl.Int64),
        pl.Series(out, units / res, dtype=pl.Float64),
        pl.Series(f"{out}_bonus", bonus / res, dtype=pl.Float64),
    )


def component_contributions(
    expected: Mapping[str, float], settings: ScoringSettings
) -> dict[str, float]:
    """Linear contribution of each box-score stat to expected fantasy points.

    Used for explanations ("22% of value from steals and blocks"); bonuses are reported
    separately by the simulator because they are not linear.
    """
    out: dict[str, float] = {}
    for key, w in settings.weights.items():
        if key in STAT_KEYS and STAT_KEYS[key] in expected:
            col = STAT_KEYS[key]
            out[col] = out.get(col, 0.0) + w * expected[col]
    return out
