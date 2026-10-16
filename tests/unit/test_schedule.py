from datetime import date

import polars as pl

from draftai.value.schedule import add_weeks, playoff_weeks, team_week_patterns, week_of, week_plan

OPENER_2025 = date(2025, 10, 21)  # a Tuesday


def test_sleeper_weeks_match_sleeper_stats_feed():
    """Week numbers checked against games in Sleeper's 2025-26 stats feed."""
    assert week_of(date(2025, 10, 21), OPENER_2025) == 1
    assert week_of(date(2025, 10, 26), OPENER_2025) == 1  # Sunday closes the partial first week
    assert week_of(date(2025, 10, 27), OPENER_2025) == 2  # Monday starts week 2
    assert week_of(date(2025, 12, 26), OPENER_2025) == 10
    assert week_of(date(2025, 12, 28), OPENER_2025) == 10
    assert week_of(date(2026, 1, 3), OPENER_2025) == 11
    assert week_of(date(2026, 1, 17), OPENER_2025) == 13
    assert week_of(date(2026, 4, 12), OPENER_2025) == 25


def test_playoff_weeks():
    assert playoff_weeks(22, 6) == [22, 23, 24]
    assert playoff_weeks(22, 4) == [22, 23]
    assert playoff_weeks(22, 6, round_type=1) == [22, 23, 24, 25]
    assert playoff_weeks(20, 6, round_type=2) == [20, 21, 22, 23, 24, 25]


def test_patterns_and_plan():
    days = [date(2025, 10, 21), date(2025, 10, 22), date(2025, 10, 24), date(2025, 10, 28)]
    s = add_weeks(pl.DataFrame({"game_id": ["a", "b", "c", "d"], "game_date": days, "team": ["X"] * 4,
                                "opp": ["Y"] * 4, "home": [True] * 4}))
    pats = team_week_patterns(s)
    assert pats[("X", 1)] == (False, True, False)
    assert pats[("X", 2)] == (False,)
    plan = week_plan(s, playoff_start=2, playoff_teams=2, playoff_weight=1.5)
    assert plan.weights == {1: 1.0, 2: 1.5}
