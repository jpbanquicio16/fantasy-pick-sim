import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import CFG, ROSTER, S, make_league, target_schedule  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def league():
    return make_league()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def training(league):
    from draftai.features.build import build_training_set
    from draftai.models.train import train_all

    ts = build_training_set(league["player_games"], league["team_games"], league["bio"], league["draft"], S,
                            list(range(2018, 2023)))
    return ts, train_all(ts, S, CFG, log=lambda *_: None)


def _with_ids(proj: pl.DataFrame, league) -> pl.DataFrame:
    return proj.with_columns(
        pl.col("player_id").cast(pl.Utf8).alias("sleeper_id"),
        pl.format("Player {}", pl.col("player_id")).alias("name"),
        pl.col("player_id").replace_strict(league["positions"], return_dtype=pl.List(pl.Utf8)).alias("positions"),
        pl.lit(None, dtype=pl.Utf8).alias("injury_status"),
    )


def _adp(b, seed: int = 0):
    rng = np.random.default_rng(seed)
    adp = (-b.players["season_value"].to_numpy() + rng.normal(0, 40, b.players.height)).argsort().argsort() + 1
    b.players = b.players.with_columns(pl.Series("adp", adp.astype(float)))
    return b


@pytest.fixture(scope="session")
def board(league, training):
    """The synthetic draft board for 2023 (the target season; no games played yet)."""
    from draftai.models.project import season_projections
    from draftai.value.board import build_board
    from draftai.value.schedule import add_weeks, week_plan

    _, r = training
    roster = pl.DataFrame({"player_id": list(league["current"]), "team": list(league["current"].values())})
    proj = season_projections(league["player_games"], league["team_games"], league["bio"], league["draft"],
                              r.models_final, S, 2023, date(2023, 10, 19), roster, pool=roster.select("player_id"))
    sched = add_weeks(target_schedule())
    plan = week_plan(sched, playoff_start=6, playoff_teams=4, playoff_weight=1.5)
    b = build_board(_with_ids(proj, league), league["player_games"], sched, plan, S, ROSTER, teams=8, b2b_mult=1.2,
                    target_season=2023, game_samples=1500)
    return _adp(b)


@pytest.fixture(scope="session")
def past_season(league, training):
    """A backtest season on the synthetic league: 2022 as it looked on opening night, and
    what then happened. (The synthetic models saw 2022; this exercises the mechanics only.)"""
    from draftai.backtest.history import LeagueRules, SeasonData, market_proxy, realise
    from draftai.draft import adp as adp_mod
    from draftai.features.build import roster_from_first_game
    from draftai.models.project import season_projections
    from draftai.normalise.tables import schedule_from_team_games
    from draftai.value.board import build_board
    from draftai.value.schedule import add_weeks, week_plan

    _, r = training
    pg, tg = league["player_games"], league["team_games"]
    cut = tg.filter(pl.col("season") == 2022)["game_date"].min()
    roster = roster_from_first_game(pg, 2022)
    proj = season_projections(pg, tg, league["bio"], league["draft"], r.models_test, S, 2022, cut, roster,
                              pool=roster.select("player_id"))
    sched = add_weeks(schedule_from_team_games(tg.filter(pl.col("season") == 2022)))
    rules = LeagueRules(teams=6, rounds=12, roster_positions=ROSTER[:-1], scoring=S, playoff_week_start=6,
                        playoff_teams=4)
    plan = week_plan(sched, rules.playoff_week_start, rules.playoff_teams, rules.playoff_weight)
    hist = pg.filter(pl.col("game_date") < cut)
    b = build_board(_with_ids(proj, league), hist, sched, plan, S, rules.roster_positions, rules.teams, 1.2, 2022,
                    game_samples=1000)
    b.players = adp_mod.attach(b.players, market_proxy(b.players))
    real = realise(b, 2022, pg, tg, S, 1.2, rules.playoff_week_start)
    return SeasonData(2022, b, real, 1.2, {}), rules
