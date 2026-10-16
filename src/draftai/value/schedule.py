"""Sleeper matchup weeks from the NBA schedule.

Sleeper weeks run Monday to Sunday; week 1 is the partial week from opening night to
the first Sunday (checked against Sleeper's 2025-26 stats feed). Playoff weeks come from
the league's ``playoff_week_start`` and ``playoff_round_type``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_of(d: date, opener: date) -> int:
    return 1 + (_monday(d) - _monday(opener)).days // 7


def add_weeks(schedule: pl.DataFrame) -> pl.DataFrame:
    """Add ``week`` and ``b2b2`` (second night of a back-to-back) to a team-game schedule."""
    opener = schedule["game_date"].min()
    weeks = [week_of(d, opener) for d in schedule["game_date"].to_list()]
    return (
        schedule.with_columns(pl.Series("week", weeks, dtype=pl.Int32))
        .sort("team", "game_date")
        .with_columns(
            ((pl.col("game_date") - pl.col("game_date").shift(1).over("team")).dt.total_days() == 1)
            .fill_null(False)
            .alias("b2b2")
        )
    )


@dataclass(frozen=True)
class WeekPlan:
    weeks: list[int]  # every Sleeper week with games
    playoff_weeks: list[int]
    weights: dict[int, float]  # week -> weight (1 for regular season, w for playoffs, 0 unused)

    @property
    def regular_weeks(self) -> list[int]:
        return [w for w in self.weeks if w not in self.playoff_weeks and self.weights[w] > 0]


def playoff_weeks(start: int, teams: int, round_type: int = 0) -> list[int]:
    """Rounds needed for ``teams`` (byes for the top seeds), one week per round unless
    ``round_type`` is 1 (two-week final) or 2 (two weeks per round)."""
    rounds = 1
    while 2**rounds < teams:
        rounds += 1
    weeks, w = [], start
    for r in range(rounds):
        span = 2 if round_type == 2 or (round_type == 1 and r == rounds - 1) else 1
        weeks.extend(range(w, w + span))
        w += span
    return weeks


def week_plan(schedule: pl.DataFrame, playoff_start: int, playoff_teams: int, playoff_weight: float,
              round_type: int = 0) -> WeekPlan:
    weeks = sorted(schedule["week"].unique().to_list())
    po = [w for w in playoff_weeks(playoff_start, playoff_teams, round_type) if w in weeks]
    last = max(po) if po else max(weeks)
    weights = {w: (playoff_weight if w in po else (1.0 if w < playoff_start else 0.0)) for w in weeks}
    weights = {w: (v if w <= last else 0.0) for w, v in weights.items()}
    return WeekPlan(weeks=weeks, playoff_weeks=po, weights=weights)


def team_week_patterns(schedule: pl.DataFrame) -> dict[tuple[str, int], tuple[bool, ...]]:
    """(team, week) -> back-to-back flags of that week's games, in order."""
    out: dict[tuple[str, int], tuple[bool, ...]] = {}
    for (team, week), g in schedule.sort("game_date").group_by(["team", "week"], maintain_order=True):
        out[(team, int(week))] = tuple(g["b2b2"].to_list())
    return out
