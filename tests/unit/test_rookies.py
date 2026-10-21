"""The rookie model (V5): draft-day inputs, monotone draft slot, comparables, projections."""

from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pytest
from synthetic import S

from draftai.features.roles import bucket_weighted, team_bucket_net
from draftai.models.rookies import (
    MIN_GP,
    RookieModel,
    age_at_bio_date,
    class_rows,
    evaluate_rookies,
)


@pytest.fixture(scope="module")
def rows(league):
    return class_rows(league["player_games"], league["team_games"], league["bio"], league["draft"], S, range(2017, 2023))


@pytest.fixture(scope="module")
def model(rows):
    return RookieModel(k=10).fit(rows.filter(pl.col("season") < 2022), S)


@pytest.mark.parametrize("birth, season, age", [
    (date(2004, 1, 4), 2023, 20.0),    # Wembanyama: the NBA's bio says 20 in 2023-24
    (date(2002, 5, 1), 2023, 22.0),    # Holmgren: 22
    (date(2006, 12, 21), 2025, 19.0),  # Flagg: 19
    (date(2000, 7, 6), 2019, 19.0),    # Williamson: 19 (birthday just after 30 June)
])
def test_live_age_matches_the_nba_bio_reference_date(birth, season, age):
    assert age_at_bio_date(birth, season) == age


def test_class_rows_cover_every_pick_with_outcomes(rows, league):
    drafted = league["draft"].filter(pl.col("draft_year").is_between(2017, 2022)).height
    assert rows.height == drafted
    assert rows["y_share"].is_between(0, 1).all()
    assert (rows.filter(pl.col("y_gp") == 0)["y_fp"] == 0).all()
    for c in ("log_pick", "second_round", "intl", "team_win_1", "team_opp_mpg", "team_opp_pos"):
        assert rows[c].null_count() == 0, c


def test_later_picks_never_project_more(model, rows):
    row = rows.filter(pl.col("season") == 2022).head(1)
    grid = pl.concat([row.with_columns(pl.lit(p).alias("draft_pick"), pl.lit(float(np.log(p))).alias("log_pick"),
                                       pl.lit(float(p > 30)).alias("second_round")) for p in (1, 5, 15, 30, 31, 60)])
    pred = model.predict(grid, S)
    for c in ("share_hat", "mpg_hat"):
        v = pred[c].to_numpy()
        assert np.all(np.diff(v) <= 1e-9), (c, v)


def test_predictions_carry_ranges_donors_and_comparables(model, rows):
    hold = rows.filter(pl.col("season") == 2022)
    pred = model.predict(hold, S)
    assert pred.height == hold.height
    assert (pred["fp_p10"] <= pred["fp_pg_hat"] + 1e-9).all() and (pred["fp_pg_hat"] <= pred["fp_p90"] + 1e-9).all()
    assert pred["share_hat"].is_between(0, 1).all()
    assert all(len(d) > 0 for d in pred["donor_ids"].to_list())
    regular = set(rows.filter((pl.col("season") < 2022) & (pl.col("y_gp") >= MIN_GP))["player_id"].to_list())
    assert all(set(d) <= regular for d in pred["donor_ids"].to_list())
    assert all(len(c) > 0 for c in pred["comps"].to_list())


def test_inputs_outside_the_training_range_are_clipped(model, rows):
    row = rows.filter(pl.col("season") == 2022).head(1)
    old = model.predict(row.with_columns(pl.lit(35.0).alias("age")), S)
    edge = model.predict(row.with_columns(pl.lit(model.bounds["age"][1]).alias("age")), S)
    assert np.isclose(old["mpg_hat"][0], edge["mpg_hat"][0])


def test_team_net_is_relative_to_the_average_team():
    prev = pl.DataFrame({"player_id": [1, 2, 3, 4], "team": ["A", "A", "B", "C"], "game_id": ["g1", "g1", "g2", "g3"],
                         "min": [2000.0, 1000.0, 1500.0, 1500.0]})
    roster = pl.DataFrame({"player_id": [1, 2, 3, 4], "team": ["B", "A", "B", "C"]})  # player 1 left A for B
    buckets = pl.DataFrame({"player_id": [1, 2, 3, 4], "hbucket": ["guard", "big", "wing", "wing"]})
    net = team_bucket_net(roster, prev, buckets)
    g = {t: v for t, b, v in net.filter(pl.col("hbucket") == "guard").iter_rows()}
    assert g["A"] > 0 > g["B"]
    assert abs(sum(g.values())) < 1e-9
    # A guard left A: guards gain most, wings half as much, bigs nothing; unknown teams zero.
    pos, flat = bucket_weighted(net, ["A", "A", "A", "Z"], ["guard", "wing", "big", "guard"])
    assert pos[0] > pos[1] > 0
    assert np.isclose(pos[2], net.filter((pl.col("team") == "A") & (pl.col("hbucket") == "big"))["net"][0]
                      + 0.5 * net.filter((pl.col("team") == "A") & (pl.col("hbucket") == "wing"))["net"][0])
    assert pos[3] == 0.0 and flat[3] == 0.0


def test_season_projections_use_the_rookie_model(league, training, model):
    from draftai.models.project import season_projections

    _, r = training
    models = replace(r.models_final, rookies=model)
    roster = pl.DataFrame({"player_id": list(league["current"]), "team": list(league["current"].values())})
    draft = league["draft"].vstack(pl.DataFrame({"draft_year": [2023], "player_id": [99001], "player_name": ["New Kid"],
                                                 "overall_pick": [3], "team": ["T01"]},
                                                schema=league["draft"].schema))
    proj = season_projections(league["player_games"], league["team_games"], league["bio"], draft, models, S, 2023,
                              date(2023, 10, 19), roster.vstack(pl.DataFrame({"player_id": [99001], "team": ["T01"]})),
                              pool=roster.select("player_id").vstack(pl.DataFrame({"player_id": [99001]})),
                              rookie_attrs=pl.DataFrame({"player_id": [99001], "age": [19.0], "height_in": [80.0],
                                                         "weight_lb": [210.0]}))
    kid = proj.filter(pl.col("player_id") == 99001).row(0, named=True)
    assert kid["rookie"] and kid["rookie_source"] == "model"
    assert kid["fp_pg_hat"] > 0 and len(kid["donor_ids"]) > 0 and kid["age"] == 19.0
    assert proj.filter(~pl.col("rookie")).height > 0


def test_rookie_model_round_trips_with_the_artifacts(training, model, tmp_path, rows):
    from synthetic import CFG

    from draftai.models.artifacts import load_models, save_models

    _, r = training
    save_models(replace(r.models_final, rookies=model), tmp_path, CFG, date(2023, 10, 1))
    loaded = load_models(tmp_path)
    hold = rows.filter(pl.col("season") == 2022)
    assert np.allclose(loaded.rookies.predict(hold, S)["fp_pg_hat"].to_numpy(), model.predict(hold, S)["fp_pg_hat"].to_numpy())
    save_models(r.models_final, tmp_path, CFG, date(2023, 10, 1))
    assert load_models(tmp_path).rookies is None


def test_evaluation_compares_against_the_baseline(league):
    ev = evaluate_rookies(league["player_games"], league["team_games"], league["bio"], league["draft"], S,
                          validation=[2020, 2021], test=2022, k=10, log=lambda *_: None)
    models = set(ev.metrics["model"].unique().to_list())
    assert {"rookie_model", "draft_slot_baseline", "comparables_mean"} <= models
    assert ev.metrics.filter(pl.col("split") == "test")["season"].unique().to_list() == [2022]
    assert ev.model_final.classes[-1] == 2022 and ev.model_test.classes[-1] == 2021
