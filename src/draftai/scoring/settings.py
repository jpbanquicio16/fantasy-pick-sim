"""League scoring settings, built from the Sleeper league's ``scoring_settings`` object.

Sleeper's basketball stat keys were read from its stats feed (the same keys appear in
league ``scoring_settings``). Unknown keys are kept and reported, never silently dropped.
"""

from __future__ import annotations

import math
import warnings
from typing import Mapping

from pydantic import BaseModel, Field, model_validator

# Sleeper stat key -> canonical box-score column (or a derived quantity).
STAT_KEYS: dict[str, str] = {
    "pts": "pts",
    "reb": "reb",
    "oreb": "oreb",
    "dreb": "dreb",
    "ast": "ast",
    "stl": "stl",
    "blk": "blk",
    "to": "tov",
    "tpm": "fg3m",
    "tpa": "fg3a",
    "fgm": "fgm",
    "fga": "fga",
    "ftm": "ftm",
    "fta": "fta",
    "pf": "pf",
    "tf": "tf",
    "ff": "ff",
}
# Misses are derived: attempts minus makes.
MISS_KEYS: dict[str, tuple[str, str]] = {
    "fgmi": ("fga", "fgm"),
    "ftmi": ("fta", "ftm"),
    "tpmi": ("fg3a", "fg3m"),
}
# Threshold bonuses. Double- and triple-doubles count PTS, REB, AST, STL and BLK.
BONUS_KEYS = ("dd", "td", "bonus_pt_40p", "bonus_pt_50p", "bonus_ast_15p", "bonus_reb_20p")
DD_CATEGORIES = ("pts", "reb", "ast", "stl", "blk")
# Combination stats Sleeper exposes (e.g. "pts_reb_ast") are sums of their parts.
COMBO_PARTS = {"pts", "reb", "ast", "stl", "blk"}
# Keys present in Sleeper stat payloads that are not scoring categories.
NON_SCORING_KEYS = {"gp", "gs", "sp", "plus_minus", "pts_std", "pts_std_dfs", "rank_std", "pos_rank_std"}

# Sleeper's default basketball scoring (the design's league).
SLEEPER_DEFAULT: dict[str, float] = {
    "pts": 0.5,
    "reb": 1.0,
    "ast": 1.0,
    "stl": 2.0,
    "blk": 2.0,
    "to": -1.0,
    "tpm": 0.5,
    "dd": 1.0,
    "td": 2.0,
    "bonus_pt_40p": 2.0,
    "bonus_pt_50p": 2.0,
    "tf": -2.0,
    "ff": -2.0,
}

# Box-score columns every scoring call understands. tf/ff default to zero when absent.
BOX_COLUMNS = (
    "pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov",
    "fg3m", "fg3a", "fgm", "fga", "ftm", "fta", "pf", "tf", "ff",
)


def _is_combo(key: str) -> bool:
    parts = key.split("_")
    return len(parts) >= 2 and all(p in COMBO_PARTS for p in parts)


class ScoringSettings(BaseModel):
    """Scoring weights plus the one rule Sleeper does not document: bonus stacking.

    ``bonus_stacking=True`` pays both the 40+ and the 50+ bonus for a 50-point game.
    Verified against Sleeper's own scored games (see tests/fixtures/golden).
    """

    weights: dict[str, float] = Field(default_factory=lambda: dict(SLEEPER_DEFAULT))
    bonus_stacking: bool = True
    unknown_keys: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _split_unknown(self) -> "ScoringSettings":
        known, unknown = {}, dict(self.unknown_keys)
        for k, v in self.weights.items():
            if v == 0:
                continue
            if k in STAT_KEYS or k in MISS_KEYS or k in BONUS_KEYS or _is_combo(k):
                known[k] = float(v)
            elif k not in NON_SCORING_KEYS:
                unknown[k] = float(v)
        self.weights, self.unknown_keys = known, unknown
        return self

    @classmethod
    def from_sleeper(
        cls, scoring_settings: Mapping[str, float] | None, bonus_stacking: bool | None = None
    ) -> "ScoringSettings":
        weights = dict(SLEEPER_DEFAULT if scoring_settings is None else scoring_settings)
        s = cls(weights=weights, bonus_stacking=True if bonus_stacking is None else bonus_stacking)
        if s.unknown_keys:
            warnings.warn(
                f"Unrecognised Sleeper scoring keys ignored: {sorted(s.unknown_keys)}", stacklevel=2
            )
        return s

    @property
    def units_per_point(self) -> int:
        """Smallest integer resolution that represents every weight exactly (2 = half-points)."""
        for res in (1, 2, 10, 100):
            if all(math.isclose(w * res, round(w * res), abs_tol=1e-9) for w in self.weights.values()):
                return res
        raise ValueError(f"Scoring weights need finer than 0.01 resolution: {self.weights}")

    def unit_weights(self) -> dict[str, int]:
        res = self.units_per_point
        return {k: int(round(v * res)) for k, v in self.weights.items()}

    @property
    def scores_fouls(self) -> bool:
        return bool(self.weights.get("tf") or self.weights.get("ff"))

    def describe(self) -> list[tuple[str, float, str]]:
        """(key, weight, meaning) rows for display."""
        meaning = {
            "dd": "double-double", "td": "triple-double",
            "bonus_pt_40p": "40+ points", "bonus_pt_50p": "50+ points",
            "bonus_ast_15p": "15+ assists", "bonus_reb_20p": "20+ rebounds",
            "tf": "technical foul", "ff": "flagrant foul",
        }
        rows = []
        for k, v in sorted(self.weights.items()):
            if k in STAT_KEYS:
                m = meaning.get(k, STAT_KEYS[k])
            elif k in MISS_KEYS:
                m = f"{MISS_KEYS[k][0]} - {MISS_KEYS[k][1]}"
            elif k in BONUS_KEYS:
                m = meaning[k]
            else:
                m = " + ".join(k.split("_"))
            rows.append((k, v, m))
        return rows
