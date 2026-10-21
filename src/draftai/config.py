"""Configuration: paths and the three YAML configs (league, train, sim).

Seasons are identified by their start year: 2025 means the 2025-26 season.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[2]


class Paths(BaseModel):
    root: Path = REPO_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def normalised(self) -> Path:
        return self.data / "normalised"

    @property
    def features(self) -> Path:
        return self.data / "features"

    @property
    def artifacts(self) -> Path:
        return self.data / "artifacts"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.normalised, self.features, self.artifacts):
            p.mkdir(parents=True, exist_ok=True)
        return self


class LeagueConfig(BaseModel):
    """Everything the league API cannot tell us, plus offline fallbacks for what it can."""

    season: int = 2026
    league_id: str | None = None
    draft_id: str | None = None
    username: str | None = None
    draft_slot: int | None = None
    # None means "use the default for Sleeper basketball", which was verified against
    # Sleeper's own scored games: 40+ and 50+ point bonuses stack.
    bonus_stacking: bool | None = None
    playoff_weight: float = 1.5
    include_cup_final: bool = False
    # Offline fallbacks, used only when no league is connected.
    teams: int = 12
    rounds: int = 13
    draft_type: str = "snake"
    reversal_round: int = 0
    playoff_week_start: int = 22
    playoff_teams: int = 6
    roster_positions: list[str] = Field(
        default_factory=lambda: [
            "PG", "SG", "G", "SF", "PF", "F", "C", "C", "UTIL", "UTIL", "BN", "BN", "BN", "IL",
        ]
    )
    scoring_settings: dict[str, float] | None = None

    @field_validator("league_id", "draft_id", "username", mode="before")
    @classmethod
    def _as_string(cls, v):
        """Sleeper ids are numeric strings; unquoted in YAML they arrive as ints."""
        return None if v is None else str(v)


class TrainConfig(BaseModel):
    ingest_seasons: list[int] = Field(default_factory=lambda: list(range(2013, 2026)))
    first_target_season: int = 2016
    validation_seasons: list[int] = Field(default_factory=lambda: [2021, 2022, 2023, 2024])
    test_season: int = 2025
    target_season: int = 2026
    recency_half_life_seasons: float = 4.0
    seed: int = 7
    lgbm: dict = Field(
        default_factory=lambda: {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 40,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "verbose": -1,
        }
    )
    ridge_alpha: float = 10.0
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    # V5. Rookies: "model" (ridge + comparables, models/rookies.py) or "baseline" (draft slot).
    rookie_model: str = "model"
    rookie_k: int = 20  # comparables per rookie
    rookie_alpha: float = 10.0
    # V5. Per-minute rate prior: "hierarchical" (regression prior, features/rates.py), "bucket"
    # (height-bucket mean) or "hierarchical_scaled"; chosen by `draftai rates-experiment`.
    rate_prior: str = "hierarchical"


class SimConfig(BaseModel):
    seed: int = 11
    game_samples: int = 4000
    min_donor_games: int = 40
    donor_neighbours: int = 8
    adp_sigma_a: float = 1.5
    adp_sigma_b: float = 0.12
    candidate_pool: int = 40
    shortlist: int = 12
    risk_lambda: float = 0.0
    poll_seconds: float = 3.0
    # V4: Monte Carlo rollouts and the opponent model.
    rollout_worlds: int = 600
    rollout_shortlist: int = 8
    rollout_race_after: int = 200
    waiver_slots: int = 1  # waiver claims a simulated roster can start in a week
    opponent_window: int = 50


class LTRConfig(BaseModel):
    """The learning-to-rank experiment (V5, design section 17; backtest/ltr.py)."""

    leagues: int = 24  # simulated leagues per season in which queries are collected
    docs: int = 24  # candidates per query (the greedy recommender's top)
    worlds: int = 150  # rollout worlds per candidate for the simulator's labels
    epsilon: float = 0.25  # exploration while collecting: a random top-5 candidate instead of the best
    levels: int = 10  # relevance grades per query
    sim_rounds: int = 300  # boosting rounds on simulator labels (the distilled surrogate)
    real_rounds: int = 100  # further rounds on realised labels (fine-tuning)
    real_learning_rate: float = 0.03
    seed: int = 17


class BacktestConfig(BaseModel):
    seasons: list[int] = Field(default_factory=lambda: [2021, 2022, 2023, 2024, 2025])
    strategies: list[str] = Field(
        default_factory=lambda: ["adp", "points", "vor", "greedy_survival", "rollout", "oracle"])
    leagues: int = 96  # simulated leagues per season (your seat rotates through every slot)
    expensive_leagues: int = 24  # leagues that also run greedy_survival and rollout
    schedules: int = 40  # head-to-head schedules averaged per league
    waiver_claims: int = 2
    rollout_worlds: int = 300
    rollout_shortlist: int = 6
    rollout_race_after: int = 150
    game_samples: int = 2000
    seed: int = 5
    jobs: int = 5
    replay_schedules: int = 200
    # Simulated rooms: "calibrated" drafts around the market with the configured ADP spread;
    # "fitted" uses the model fitted on your real drafts (noisier: it reads informed
    # deviations from the proxy market as noise, so its teams are weaker than real ones).
    room: str = "calibrated"
    ltr: LTRConfig = Field(default_factory=LTRConfig)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_league_config(path: Path | None = None) -> LeagueConfig:
    paths = Paths()
    if path is None:
        path = paths.configs / "league.yaml"
        if not path.exists():
            path = paths.configs / "league.example.yaml"
    return LeagueConfig(**_load_yaml(path))


def load_train_config(path: Path | None = None) -> TrainConfig:
    return TrainConfig(**_load_yaml(path or Paths().configs / "train.yaml"))


def load_sim_config(path: Path | None = None) -> SimConfig:
    return SimConfig(**_load_yaml(path or Paths().configs / "sim.yaml"))


def season_label(season: int) -> str:
    """2025 -> '2025-26' (nba_api's season format)."""
    return f"{season}-{(season + 1) % 100:02d}"


def season_of(d: date) -> int:
    """NBA season (start year) that a date belongs to; the season turns over on 1 August."""
    return d.year if d.month >= 8 else d.year - 1


def load_backtest_config(path: Path | None = None) -> BacktestConfig:
    return BacktestConfig(**_load_yaml(path or Paths().configs / "backtest.yaml"))
