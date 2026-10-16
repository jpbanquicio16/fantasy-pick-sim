from datetime import date

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from draftai.features.asof import AsOf, LeakageError, season_cutoff
from draftai.features.build import (
    availability_stats,
    build_features,
    build_training_set,
    eb_prior,
    roster_from_first_game,
)
from draftai.scoring import ScoringSettings

S = ScoringSettings()


def _features(league, Y, pg):
    tg = league["team_games"]
    cut = season_cutoff(tg, Y)
    roster = roster_from_first_game(league["player_games"], Y)
    return build_features(AsOf.view(pg, tg, cut), Y, roster, league["bio"], league["draft"], S)


def test_perturbation_post_cutoff_rows_do_not_change_features(league):
    """Scramble, drop and inflate every row dated on or after the cut-off."""
    pg = league["player_games"]
    Y = 2020
    cut = season_cutoff(league["team_games"], Y)
    base = _features(league, Y, pg)
    rng = np.random.default_rng(0)
    after = pg.filter(pl.col("game_date") >= cut)
    scrambled = after.with_columns(
        *[pl.Series(c, rng.permutation(after[c].to_numpy())) for c in ("pts", "reb", "ast", "stl", "blk", "min")],
        (pl.col("tov") * 3).alias("tov"),
    ).sample(fraction=0.7, seed=1)
    perturbed = pl.concat([pg.filter(pl.col("game_date") < cut), scrambled])
    assert_frame_equal(base, _features(league, Y, perturbed))


def test_asof_view_refuses_future_rows(league):
    pg, tg = league["player_games"], league["team_games"]
    with pytest.raises(LeakageError):
        AsOf(as_of=date(2018, 1, 1), player_games=pg, team_games=tg)
    v = AsOf.view(pg, tg, date(2018, 1, 1))
    assert v.player_games["game_date"].max() < date(2018, 1, 1)


def test_bio_for_target_season_is_not_used(league):
    """A bio row for season Y would reveal that the player played in Y."""
    Y = 2020
    pg = league["player_games"]
    base = _features(league, Y, pg)
    bio = league["bio"].with_columns(
        pl.when(pl.col("season") >= Y).then(99.0).otherwise(pl.col("age")).alias("age"))
    tg = league["team_games"]
    alt = build_features(AsOf.view(pg, tg, season_cutoff(tg, Y)), Y, roster_from_first_game(pg, Y), bio,
                         league["draft"], S)
    assert_frame_equal(base, alt)


def test_training_pool_includes_players_who_missed_the_season(league):
    ts = build_training_set(league["player_games"], league["team_games"], league["bio"], league["draft"], S,
                            [2019, 2020])
    assert (ts["y_gp"] == 0).sum() > 0  # retirements and season-long absences kept as zeros
    assert ts.select(pl.struct("player_id", "season").is_unique().all()).item()


def test_availability_runs_and_shares():
    d = [date(2020, 10, 20 + i) for i in range(10)]
    tg = pl.DataFrame({"season": [2020] * 10, "team": ["A"] * 10, "game_id": [f"g{i}" for i in range(10)],
                       "game_date": d})
    played = [0, 1, 2, 3, 8, 9]  # misses 4-7 (a 4-game absence); no short misses
    pg = pl.DataFrame({"player_id": [1] * 6, "season": [2020] * 6, "team": ["A"] * 6,
                       "game_id": [f"g{i}" for i in played], "game_date": [d[i] for i in played]})
    a = availability_stats(pg, tg)
    r = a.row(0, named=True)
    assert r["n_sched"] == 10 and r["longest_absence"] == 4 and r["long_missed"] == 4 and r["short_missed"] == 0
    pg2 = pg.filter(pl.col("game_id") != "g2")  # one isolated rest day
    r2 = availability_stats(pg2, tg).row(0, named=True)
    assert r2["short_missed"] == 1 and r2["q_short"] == pytest.approx(1 / 6)


def test_eb_prior_shrinks_noisy_stats_harder():
    rng = np.random.default_rng(0)
    minutes = rng.uniform(200, 2500, 400)
    true_reb = rng.normal(0.2, 0.06, 400).clip(0.02)
    true_stl = rng.normal(0.03, 0.003, 400).clip(0.005)
    _, k_reb = eb_prior(rng.poisson(true_reb * minutes).astype(float), minutes)
    _, k_stl = eb_prior(rng.poisson(true_stl * minutes).astype(float), minutes)
    assert k_stl > k_reb


def test_rolling_windows_never_include_target_season(league):
    ts = build_training_set(league["player_games"], league["team_games"], league["bio"], league["draft"], S, [2021])
    pg = league["player_games"]
    last = pg.filter(pl.col("season") == 2020).group_by("player_id").agg(pl.len().alias("gp20"))
    j = ts.join(last, on="player_id", how="inner")
    assert (j["gp_1"] == j["gp20"]).all()
